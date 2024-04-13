import logging
from collections import namedtuple
from enum import Enum
from queue import Empty, Queue
from typing import Optional, Tuple, Union

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from scipy.fft import rfft, rfftfreq
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from soundsep.core.models import Project, ProjectIndex, Source, StftIndex
from soundsep.core.ampenv import filter_and_ampenv
from soundsep.core.stft import create_gaussian_window
from soundsep.core.utils import DuplicatedRingBuffer, get_blocks_of_ones


logger = logging.getLogger(__name__)


Selection = namedtuple("Selection", ["x0", "x1", "f0", "f1", "source"])


class StftConfig(dict):
    def __init__(self, window, step):
        self.window = window
        self.step = step


class SelectionService:
    """Keeps track of selection state, a rectangle of time and frequency bounds

    Includes a "fine selection", which is a smaller ROI on top of the main roi
    that can be accessed with get_fine_selection()

    Includes a "segment selection", which is a single segment ID
    """
    def __init__(self, project: Project):
        self.project = project
        self._selection = None
        self._fine_selection = None
        self._segment_selection = []

    def is_set(self) -> bool:
        return self._selection is not None

    def clear(self):
        self._selection = None
        self._fine_selection = None

    def set_selection(
            self,
            x0: ProjectIndex,
            x1: ProjectIndex,
            f0: float,
            f1: float,
            source: Source,
        ):
        self._selection = Selection(x0, x1, f0, f1, source)

    def move_selection(self, dx: int):
        self._selection = Selection(
            self._selection.x0 + dx,
            self._selection.x1 + dx,
            self._selection.f0,
            self._selection.f1,
            self._selection.source
        )

    def scale_selection(self, n: int):
        self._selection = Selection(
            self._selection.x0 - n // 2,
            self._selection.x1 + n // 2,
            self._selection.f0,
            self._selection.f1,
            self._selection.source
        )

    def get_selection(self) -> Optional[Selection]:
        return self._selection

    # Fine selection
    def set_fine_selection(self, x0: ProjectIndex, x1: ProjectIndex):
        if not self._selection:
            raise ValueError("Cannot set a fine selection if no current selection exists")

        self._fine_selection = Selection(x0, x1, self._selection.f0, self._selection.f1, self._selection.source)

    def clear_fine_selection(self):
        self._fine_selection = None

    def get_fine_selection(self) -> Optional[Selection]:
        return self._fine_selection or self._selection

    # Segment selection
    def set_segment_selection( self, segIDs):
        self._segment_selection = segIDs.copy()
    
    def get_segment_selection(self):
        return self._segment_selection.copy()
    
    def clear_segment_selection(self):
        self._segment_selection = []


class SourceService(list):
    """Placeholder service for Source management"""
    def __init__(self, project: Project):
        self.project = project
        self._needs_saving = False
        super().__init__()

    @property
    def _existing_names(self):
        return set([s.name for s in self])

    def create(self, name: str, channel: int) -> Source:
        if name in self._existing_names:
            raise ValueError("Cannot create a Source with a non-unique name")

        new_source = Source(self.project, name, channel, len(self))
        self.append(new_source)
        self._needs_saving = True
        return new_source

    def edit(self, index: int, name: str, channel: int) -> Source:
        self[index].name = name
        self[index].channel = channel
        self._needs_saving = True
        return self[index]

    def delete(self, index: int) -> "SourceService":
        del self[index]
        for source in self[index:]:
            source.index -= 1
        self._needs_saving = True
        return self

    def set_needs_saving(self, flag: bool):
        self._needs_saving = flag

    def needs_saving(self) -> bool:
        return self._needs_saving

    def create_template_source(self) -> 'Source':
        """Creates a new source with a generic name on the next available channel
        """
        if len(self):
            next_channel = int(np.max([s.channel for s in self])) + 1
        else:
            next_channel = 0

        if next_channel >= self.project.channels:
            next_channel = 0

        existing_names = self._existing_names
        # Try names until we get a unique one
        name_idx = next_channel
        try_name = "New Source {}".format(name_idx)
        while try_name in existing_names:
            name_idx += 1
            try_name = "New Source {}".format(name_idx)

        return self.create(try_name, next_channel)


class AmpenvService:
    """Simple service that cache's the last computation

    This is useful because often you might want to get data multiple times
    """

    def __init__(self, project: Project):
        self.project = project
        self._last_computation = None

    # TODO: as part of api's get_signal etc cleanup, make AmpenvService do its own caching
    # So it can take start, stop values instead of
    def filter_and_ampenv(
            self,
            signal: np.ndarray,
            f0: float,
            f1: float,
            rectify_lowpass: float,
            ) -> Tuple[np.ndarray, np.ndarray]:

        result = filter_and_ampenv(
            signal,
            self.project.sampling_rate,
            f0,
            f1,
            rectify_lowpass
        )

        return result


class StftWorker(QThread):
    """Background thread for processing STFT

    Emits resultReady(StftIndex, channel, data) when the fft at a particular
    StftIndex and channel is completed.

    Pushing tuples of (StftIndex, StftIndex) to the StftWorker.queue will
    tell it to compute the fft at every index in that range. If

    Attributes
    ----------
    queue : queue.Queue
        Queue that the worker pulls requests from. Tuples of StftIndex start
        and stop pairs trigger the worker to processing FFTs for every time
        point in that range.
    config : StftConfig
        The configuration dictionary containing the stft.window and stft.step
        parameters
    """

    _CLEAR = object()
    END = object()
    resultReady = pyqtSignal(StftIndex, np.ndarray)

    def __init__(self, queue: Queue, project: Project, config: StftConfig):
        super().__init__()
        self.queue = queue
        self.project = project
        self.config = config

        self.gaussian_window = create_gaussian_window(self.config.window, nstd=6)
        self.gaussian_window = self.gaussian_window.reshape((-1, 1))  # for broadcasting
        self._cancel_flag = False

    def cancel(self):
        """Tell the program to stop processing and clear the queue"""
        self._cancel_flag = True
        while not self.queue.empty():
            try:
                self.queue.get(False)
            except Empty:
                continue

        # This tells the worker it can start processing events again
        self.queue.put(StftWorker._CLEAR)

    def run(self):
        """Continuously process requests received on a Queue, but close when specific objects are received"""
        while True:
            q = self.queue.get()
            if q is StftWorker._CLEAR:
                self._cancel_flag = False
            elif q is StftWorker.END:
                break
            else:
                self.process_request(q[0], q[1])

    def process_request(self, start: StftIndex, stop: StftIndex):
        """
        """
        n_fft = 2 * self.config.window + 1
        phantom_start = start.to_project_index() - self.config.window
        phantom_stop = stop.to_project_index() + self.config.window
        arr = self.project[max(ProjectIndex(self.project, 0), phantom_start):phantom_stop]

        if int(phantom_start) < 0:
            pad_start = int(np.abs(phantom_start))
        else:
            pad_start = 0

        if int(phantom_stop) > self.project.frames:
            pad_stop = int(phantom_stop - self.project.frames)
        else:
            pad_stop = 0

        arr = np.pad(arr, ((pad_start, pad_stop), (0, 0)), mode="reflect")

        # Produce a view of shape (steps, samples, channels)
        strides = sliding_window_view(arr, n_fft, axis=0)[::self.config.step].swapaxes(1, 2)
        for window_center_stft, window in zip(StftIndex.range(start, stop), strides):
            if self._cancel_flag:
                return
            window_data = window * self.gaussian_window
            window_fft = np.abs(rfft(window_data, n=n_fft, axis=0))
            self.resultReady.emit(window_center_stft, window_fft)


class StftCache(QObject):
    """A service for caching stft values in and around an active data range

    The StftCache defines a range of data called the "active range" which
    is its primary read region. In the main application this corresponds to the
    region of data visible to the user. This region has width `self.n_active` in
    units of the stft step, and starts at `self._pos`.

    The StftCache also defines a range of data surrounding the "active range"
    called the "cache range". This typically consists of the active range plus
    `self.pad` windows before and after. When the active range is close to the
    endpoints of the project, the endpoints of the cache align with the edge
    of the project such that the size of the "cache range" remains constant.

    Attributes
    ----------
    project : Project
    pad : int
        Requested width of one side of the cache range
    n_active : int
        Requested width of the active region. Will only be shorter if the project
        is not long enough to accomodate it, in which case the entire project
        will lie in the active region
    n_cache_total : int
        The total width of the cache region.
    config : StftConfig
        The stft config parameters including stft.window and stft.width
    _pos : StftIndex
        The index in the project that the start of the active region corresponds to
    _start_ptr : StftIndex
        The index in the project that the start of the cache region corresponds to
    """
    def __init__(self, project, n_active: int, pad: int, stft_config: StftConfig):
        super().__init__()
        self.project = project
        self._project_stft_steps = self.project.frames // stft_config.step + 1

        self.pad = pad
        # Don't allow an active range larger than the total number of samples in project
        self.n_active = min(n_active, self._project_stft_steps)
        self.n_cache_total = min(self.n_active + 2 * self.pad, self._project_stft_steps - self.n_active)

        self.config = stft_config

        # What index in the project the start of the active region corresponds to
        self._pos = StftIndex(project, stft_config.step, 0)
        # What index should the active window start on (relative to project)
        self._start_ptr = StftIndex(project, stft_config.step, 0)

        self._positive_freqs_filter = self._all_freqs() > 0
        self.positive_freqs = self._all_freqs()[self._positive_freqs_filter]

        self._max_freq = (self.project.sampling_rate * 0.5) * (1 - (1 / (2 * self.config.window + 1)))
        self._data = DuplicatedRingBuffer(np.zeros(
            (self.n_cache_total, self.project.channels, len(self.positive_freqs)),
            dtype=np.float32
        ))
        self._stale = DuplicatedRingBuffer(np.ones(len(self._data), dtype=np.bool))

        self.queue = Queue()
        self._worker = StftWorker(self.queue, project, self.config)
        self._worker.resultReady.connect(self._on_worker_result)
        self._worker.start()

        self._trigger_jobs()  # Force it to populate the initial section

    def close(self):
        self._worker.cancel()

    def max_freq(self):
        """Return the largest frequency value in all the land"""
        return self._max_freq

    def _all_freqs(self):
        """Return the frequency range of the ffts, this includes negative frequencies"""
        return rfftfreq(2 * self.config.window + 1, 1 / self.project.sampling_rate)

    def get_cache_range_from_active_position(self, pos: StftIndex) -> Tuple[StftIndex, StftIndex]:
        """Get the cache range from the active range's start position
        """
        potential_start = pos - self.pad
        start = max(potential_start, StftIndex(self.project, self.config.step, 0))

        if start + self.n_cache_total > StftIndex(self.project, self.config.step, self._project_stft_steps):
            stop = StftIndex(self.project, self.config.step, self._project_stft_steps)
            start = stop - self.n_cache_total
        else:
            stop = start + self.n_cache_total

        return (start, stop)

    def set_active_size(self, n):
        if n > self.n_cache_total:
            raise RuntimeError("Cannot increase active size of cache greater than the cache was initaialized with")
        if n < 0:
            raise RuntimeError("Cannot set active size to zero")

        self.pad = (self.n_cache_total - n) // 2
        self.n_active = self.n_cache_total - 2 * self.pad

    def get_active_range_from_active_position(self, pos: StftIndex) -> Tuple[StftIndex, StftIndex]:
        """Get the active range from the active range's start position
        """
        return (pos, pos + self.n_active)

    def set_position(self, pos: StftIndex):
        """Move the left edge of the active range to the given position
        """
        if int(pos) < 0 or int(pos) > self._project_stft_steps - self.n_active:
            pos = StftIndex(self._pos.project, self._pos.step, self._project_stft_steps - self.n_active)

        if pos == self._pos:
            return

        # Determine how far to slide the cached data over
        new_cache_bounds = self.get_cache_range_from_active_position(pos)
        offset = new_cache_bounds[0] - self._start_ptr

        self._data.roll(offset, fill=0)
        self._stale.roll(offset, fill=True)

        self._start_ptr = new_cache_bounds[0]
        self._pos = pos

        self._trigger_jobs()

    @pyqtSlot(StftIndex, np.ndarray)
    def _on_worker_result(self, idx: StftIndex, fft_result):
        if 0 <= int(idx - self._start_ptr) < len(self._data):
            self._data[idx - self._start_ptr] = fft_result[self._positive_freqs_filter].T
            self._stale[idx - self._start_ptr] = False

    def _trigger_jobs(self):
        """Determine what regions of data are stale and queue jobs for the worker to fill them in
        """
        # First clear out the worker's queue
        self._worker.cancel()

        jobs = []

        # First within the active range
        active_bounds = self.get_active_range_from_active_position(self._pos)
        cache_bounds = self.get_cache_range_from_active_position(self._pos)

        request_ranges = [
            (active_bounds[0], active_bounds[1]),
            (active_bounds[1], cache_bounds[1]),
            (cache_bounds[0], active_bounds[0]),
        ]

        for start, stop in request_ranges:
            for a, b in get_blocks_of_ones(
                    self._stale[start - self._start_ptr:stop - self._start_ptr]
                    ):
                jobs.append((
                    start + int(a),
                    start + int(b)
                ))

        for job in jobs:
            self._worker.queue.put((
                StftIndex(self.project, self.config.step, job[0]),
                StftIndex(self.project, self.config.step, job[1])
            ))

    def read(self, start: Optional[StftIndex] = None, stop: Optional[StftIndex] = None):
        """Read data from start (inclusive) to stop (exclusive)

        Arguments
        ---------
        start : Optional[StftIndex] (default None)
        stop : Optional[StftIndex] (default None)

        Returns
        -------
        data : np.ndarray[float]
            The 2D array of the stft values from start to stop. Values not in the cache are return as 0
        stale : np.ndarray[bool]
            Boolean indicator of whether a returned index is valid or not computed yet
        """
        x0, x1 = self.get_active_range_from_active_position(self._pos)
        start = start or x0
        stop = stop or x1

        cache_bounds = self.get_cache_range_from_active_position(self._pos)

        logger.debug("Reading cache from {} to {} out of {}".format(start, stop, cache_bounds))

        if start < cache_bounds[0] or stop > cache_bounds[1]:
            raise ValueError("Attempting read outside of current Cache values. Call StftCache.set_position first?")

        start_idx = start - self._start_ptr
        stop_idx = stop - self._start_ptr

        return self._data[start_idx:stop_idx], self._stale[start_idx:stop_idx]


class Workspace(QObject):
    """Representation of the current working time range in StftIndex units

    The workspace is represented by a start index (inclusive) and end index (non-inclusive)
    """

    class Alignment(Enum):
        Left = "left"
        Center = "center"
        Right = "right"

    def __init__(self, start: StftIndex, stop: StftIndex):
        super().__init__()
        if start.project != stop.project:
            raise TypeError("Cannot instantiate Workspace with StftIndex values from different projects")
        if start.step != stop.step:
            raise TypeError("Cannot instantiate Workspace with StftIndex values with different step sizes")

        self.start = start
        self.stop = stop
        self.set_position(start, stop)

    def __repr__(self):
        return "Workspace<{}, {}>".format(self.start, self.stop)

    @property
    def project(self) -> Project:
        """The project the Workspace is referencing"""
        return self.start.project

    @property
    def step(self) -> int:
        """The step size of the Workspace's StftIndex units"""
        return self.start.step

    @property
    def max_size(self) -> int:
        """Total number of StftIndex frames available in project"""
        return (self.project.frames // self.step) + 1

    @property
    def min_index(self) -> StftIndex:
        return StftIndex(self.project, self.step, 0)

    @property
    def max_index(self) -> StftIndex:
        return StftIndex(self.project, self.step, self.max_size)

    @property
    def size(self) -> int:
        """Return the size of the Workspace in StftIndex units"""
        return self.stop - self.start

    def move_to(self, start: StftIndex):
        """Move the starting point of the Workspace to the given index, preseving size

        Movement will stop when an endpoint is reached

        Arguments
        ---------
        start : StftIndex
        """
        dx = start - self.start
        return self.move_by(dx)

    def move_by(self, dx: int):
        """Move the starting point of the Workspace by the given amount, preserving size

        Movement will stop when an endpoint is reached

        Arguments
        ---------
        dx : int
        """
        self.set_position(self.start + dx, self.stop + dx, preserve_requested_size=True)

    def scale(self, n: int):
        """Increase or decrease the extent of this Workspace by n StftIndex units

        The increments are made alternating endpoints (starting with self.stop) and
        increasing/decreasing the size of the Workspace until the size has changed by n StftIndex
        units. If the endpoints reach 0 or the end of the project, the remainder is added
        to the other end.

        Arguments
        ---------
        n : int
        """
        if n < 0:
            n = max(n, 1 - self.size)
        else:
            n = min(n, self.max_size - self.size)

        start = int(self.start)
        stop = int(self.stop)
        sign = np.sign(n)

        for i in range(abs(n)):
            if i % 2 == 0:
                if int(stop) < self.max_size:
                    stop += sign
                else:
                    start -= sign
            else:
                if int(start) > 0:
                    start -= sign
                else:
                    stop += sign

        self.set_position(StftIndex(self.project, self.step, start), StftIndex(self.project, self.step, stop))

    def get_lim(self, as_: Union[ProjectIndex, StftIndex]) -> Tuple:
        if as_ == ProjectIndex:
            return (self.start.to_project_index(), self.stop.to_project_index())
        elif as_ == StftIndex:
            return (self.start, self.stop)
        else:
            raise TypeError

    def set_position(self, start: StftIndex, stop: StftIndex, preserve_requested_size: bool = False):
        """Attempt to set a new start and stop position

        Arguments
        ---------
        start : StftIndex
        stop : StftIndex
        preserve_requested_size : bool (default False)
            If False, will truncate the endpoints if they flow beyond the ends of the project.
            If True, if the Workspace would overflow the bounds of the project, will adjust the start
            or stop points to guarantee the Workspace size equals the requested stop - start.
        """
        new_start = self.min_index if start < self.min_index else start
        new_stop = self.max_index if stop > self.max_index else stop

        if stop - start < 1:
            raise ValueError("Workspace stop must be after start: got {} to {}".format(start, stop))

        if preserve_requested_size:
            requested_size = stop - start
            if new_stop - new_start == requested_size:
                self.start = new_start
                self.stop = new_stop
            elif new_start == self.min_index:
                self.start = new_start
                self.stop = min(new_start + requested_size, self.max_index)
            elif new_stop == self.max_index:
                self.start = max(new_stop - requested_size, self.min_index)
                self.stop = new_stop
        else:
            self.start = new_start
            self.stop = new_stop
