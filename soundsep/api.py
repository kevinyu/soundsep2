"""Thin wrapper around the main app controller
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np

from soundsep.app.services import Selection, Workspace
from soundsep.core.models import ProjectIndex, StftIndex, Source


logger = logging.getLogger(__name__)


class SignalTooShort(Exception):
    """Signal was too short to compute ampenv"""


class Api(QObject):

    # These signals are propogated up to the Loader
    _closeProject = pyqtSignal()
    _switchProject = pyqtSignal(Path)

    # These signals are public
    projectLoaded = pyqtSignal()
    projectDataLoaded = pyqtSignal()
    sourcesChanged = pyqtSignal()
    configChanged = pyqtSignal(object)  # Not implemented yet
    workspaceChanged = pyqtSignal()
    selectionChanged = pyqtSignal()
    fineSelectionChanged = pyqtSignal()
    segmentSelectionChanged = pyqtSignal()
    segmentDeleted = pyqtSignal(int)
    segmentCreated = pyqtSignal(int)
    closingProgram = pyqtSignal()

    def __init__(self, app: 'soundsep.app.app.SoundsepApp'):
        super().__init__()
        self._app = app
        self._cache = {}
        self._app.configChanged.connect(self.configChanged)

    @property
    def paths(self):
        return self._app.paths

    @property
    def project(self):
        return self._app.project

    @property
    def plugins(self):
        return self._app.plugins

    @property
    def config(self):
        return self._app.config

    def _close(self):
        self.closingProgram.emit()
        self._app.close()

    def switch_project(self, project_dir: Path):
        """Attempt to switch project to a new directory

        Arguments
        ---------
        project_dir : pathlib.Path

        Emits
        -----
        _switchProject
        """
        self._switchProject.emit(project_dir)

    def get_mut_datastore(self) -> dict:
        """Return the mutable datastore dictionary

        Returns
        -------
        datastore : dict
        """
        return self._app.datastore

    def make_project_index(self, i: Union[int, float]) -> ProjectIndex:
        """Shortcut for generating a ProjectIndex by rounding a normal int or float

        Arguments
        ---------
        i : int or float

        Returns
        -------
        pidx : ProjectIndex
        """
        if isinstance(i, float):
            i = np.round(i)

        return ProjectIndex(self._app.project, int(i))

    def create_blank_source(self) -> Source:
        """Shorthand for creating a new blank source on the next available channel

        Shorthand for api.create_source("New Source {ch}", ch), where ch is the next
        channel that does not have a source yet, or channel 0 if all channels
        have been used.

        Returns
        -------
        new_source : Source

        Emits
        -----
        sourcesChanged
        """
        new_source = self._app.datastore["sources"].create_template_source()
        self.sourcesChanged.emit()
        return new_source

    def create_source(self, source_name: str, source_channel: int) -> Source:
        """Append a new source

        Arguments
        ---------
        source_name : str
        source_channel : int

        Returns
        -------
        new_source : Source

        Emits
        -----
        sourcesChanged
        """
        new_source = self._app.datastore["sources"].create(source_name, source_channel)
        self.sourcesChanged.emit()
        return new_source
    
    def get_source(self, source_name: str, source_channel: int) -> Source:
        """Get an existing source

        Arguments
        ---------
        source_name : str
        source_channel : int

        Returns
        -------
        source : Source
        """
        sources = self.get_sources()
        for source in sources:
            if source.name == source_name and source.channel == source_channel:
                return source
        return None

    def edit_source(self, source_index: int, source_name: str, source_channel: int) -> Source:
        """Edit an existing source by index

        Arguments
        ---------
        source_index : int
        source_name : str
        source_channel : int

        Returns
        -------
        modified_source : Source

        Emits
        -----
        sourcesChanged
        """
        modified_source = self._app.datastore["sources"].edit(source_index, source_name, source_channel)
        self.sourcesChanged.emit()
        return modified_source

    def delete_source(self, source_index: int):
        """Delete an existing source by index

        Arguments
        ---------
        source_index : int

        Emits
        -----
        sourcesChanged
        """
        self._app.datastore["sources"].delete(source_index)
        self.sourcesChanged.emit()

    def get_sources(self) -> List[Source]:
        """Return a list of sources

        Returns
        -------
        sources : List[Source]
        """
        return list(self._app.datastore["sources"])

    def convert_project_index_to_stft_index(self, idx: ProjectIndex) -> StftIndex:
        """Converts a ProjectIndex to StftIndex based on the current Stft config

        Arguments
        ---------
        idx : ProjectIndex

        Returns
        -------
        stft_idx : StftIndex
        """
        return StftIndex(
            self._app.project,
            self._app.services["stft"].params.hop,
            idx // self._app.services["stft"].params.hop
        )
    
    def create_stftindex(self, i: int) -> StftIndex:
        """Create a new StftIndex

        Arguments
        ---------
        i : int

        Returns
        -------
        stft_idx : StftIndex
        """
        return StftIndex(self._app.project, self._app.services["stft"].params.hop, i)

    def workspace_move_to(self, start: StftIndex, alignment: Workspace.Alignment = None):
        """Move the current workspace to start at the given index in StftIndex

        Arguments
        ---------
        start : StftIndex
        alignment : Workspace.Alignment (default Workspace.Alignment.Left)

        Emits
        -----
        workspaceChanged
        """
        prev_position = self._app.state["workspace"].start
        self._app.state["workspace"].move_to(start)
        # self._app.services["stft"].set_position(self._app.state["workspace"].start)
        self._app.services["stft"].set_central_range(self._app.state["workspace"].start, self._app.state["workspace"].stop)
        self._cache["get_workspace_signal"] = None
        if prev_position != start:
            self.workspaceChanged.emit()

    def workspace_move_by(self, dx: int):
        """Move the current workspace by the given amount in StftIndex units

        Arguments
        ---------
        dx : int
            Positive numbers move the workspace to the right, negative numbers to
            the left

        Emits
        -----
        workspaceChanged
        """
        self._app.state["workspace"].move_by(dx)
        self._app.services["stft"].set_central_range(self._app.state["workspace"].start, self._app.state["workspace"].stop)
        self._cache["get_workspace_signal"] = None
        if dx != 0:
            self.workspaceChanged.emit()

    def workspace_scale(self, n: int):
        """Scale the current workspace by the given number of StftIndex units

        TODO: Add a parameters so scaling can center on a position

        Arguments
        ---------
        n : int

        Emits
        -----
        workspaceChanged
        """
        max_size = self._app.services["stft"].n_cache_total
        if self._app.state["workspace"].size + n >= max_size:
            n = max_size - self._app.state["workspace"].size

        self._app.state["workspace"].scale(n)
        self._app.services["stft"].set_central_range(self._app.state["workspace"].start, self._app.state["workspace"].stop)
        self._cache["get_workspace_signal"] = None
        if n != 0:
            self.workspaceChanged.emit()

    def workspace_set_position(self, start: StftIndex, stop: StftIndex):
        """Set workspace by epxlicitly setting the start and stop coordinates

        Used if you need more precision than workspace_scale() offers

        Arguments
        ---------
        start : StftIndex
        stop : StftIndex

        Emits
        -----
        workspaceChanged
        """
        old_position = self._app.state["workspace"].get_lim(StftIndex)
        max_size = self._app.services["stft"].n_cache_total
        if stop - start >= max_size:
            stop = start + max_size

        self._app.state["workspace"].set_position(start, stop)
        self._app.services["stft"].set_central_range(self._app.state["workspace"].start, self._app.state["workspace"].stop)

        self._cache["get_workspace_signal"] = None
        if old_position[0] != start or old_position[1] != stop:
            self.workspaceChanged.emit()

    def workspace_get_lim(self) -> Tuple:
        """Get the limits of the current workspace in units of StftIndex

        Returns
        -------
        start : StftIndex
        stop : StftIndex
        """
        return self._app.state["workspace"].get_lim(StftIndex)

    def read_stft(self, start: StftIndex, stop: StftIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the STFT values of the current workspace

        Example
        -------
        >>> start, stop = api.workspace_get_lim()
        >>> stft, stale, freqs = api.read_stft(start, stop)

        Returns
        -------
        t : np.ndarray[int]
            Time axis of data in ProjectIndex units
        result : np.ndarray[float]
            2D array of stft data for the requested range
        stale : np.ndarray[bool]
            Which indices of the requested range have not been filled in
        freqs : np.ndarray[float]
            Frequency axis of data
        """
        t, data, stale = self._app.services["stft"].read(start, stop)
        t = np.array([ProjectIndex(self._app.project, t_ * start.step) for t_ in t])

        return t, data, stale, self._app.services["stft"].positive_freqs

    def get_workspace_signal(self) -> Tuple[list, np.ndarray]:
        """Return the xrange and signal of the current workspace

        Returns
        -------
        time_axis : List[ProjectIndex]
        result : np.ndarray[float]
        """
        # TODO: clean this up into a service maybe?
        # this is a short quick hacky way to stop reading from disk too much
        if self._cache.get("get_workspace_signal"):
            return self._cache["get_workspace_signal"]

        return self._cache_workspace_signal()

    def _cache_workspace_signal(self) -> Tuple[list, np.ndarray]:
        # TODO: with a smarter caching strategy, we dont't need to load
        # the entire workspace signal at at time, just what we don't already
        # have...
        xlim = self._app.state["workspace"].get_lim(ProjectIndex)
        data = self._app.project[xlim[0]:xlim[1]].astype(np.float32)
        result = list(ProjectIndex.range(xlim[0], xlim[1])), data
        self._cache["get_workspace_signal"] = result
        return result

    def get_signal(self, x0: ProjectIndex, x1: ProjectIndex) -> Tuple[np.ndarray, np.ndarray]:
        """Return the signal in the given range

        Arguments
        ---------
        x0 : ProjectIndex
        x1 : ProjectIndex

        Returns
        -------
        time_axis : np.ndarray[float]
        result : np.ndarray[float]
        """
        xlim = self._app.state["workspace"].get_lim(ProjectIndex)
        if x0 >= xlim[0] and x1 <= xlim[1]:
            if not self._cache.get("get_workspace_signal"):
                self._cache_workspace_signal()
            cached_t, cached_data = self._cache["get_workspace_signal"]
            i0 = x0 - xlim[0]
            i1 = x1 - xlim[0]
            return cached_t[i0:i1], cached_data[i0:i1]

        data = self._app.project[x0:x1].astype(np.float32)
        return list(ProjectIndex.range(x0, x1)), data

    def set_selection(
            self,
            x0: ProjectIndex,
            x1: ProjectIndex,
            f0: float,
            f1: float,
            source: Source,
        ):
        """Set the current selection to a bound in time and frequency

        Arguments
        ---------
        x0 : ProjectIndex
        x1 : ProjectIndex
        f0 : float
            Lower frequency bound
        f1 : float
            Upper frequency bound
        source : Source

        Emits
        -----
        selectionChanged
        """
        self._app.state["selection"].set_selection(x0, x1, f0, f1, source)
        self.selectionChanged.emit()
    
    def set_segment_selection(
            self,
            segIds
        ):
        """Set the current selection to a segment"""
        self._app.state["selection"].set_segment_selection(segIds)
        self.segmentSelectionChanged.emit()

    def set_fine_selection(self, x0: ProjectIndex, x1: ProjectIndex):
        """Set the current fine selection's time bounds, inheriting the existing selections frequency band

        Arguments
        ---------
        x0 : ProjectIndex
        x1 : ProjectIndex

        Emits
        -----
        fineSelectionChanged
        """
        self._app.state["selection"].set_fine_selection(x0, x1)
        self.fineSelectionChanged.emit()

    def clear_selection(self):
        """Clear the current selection

        Emits
        -----
        selectionChanged
        fineSelectionChanged
        """
        self._app.state["selection"].clear()
        self.selectionChanged.emit()
        self.fineSelectionChanged.emit()

    def clear_fine_selection(self):
        """Clear the current fine selection

        Emits
        -----
        fineSelectionChanged
        """
        self._app.state["selection"].clear_fine_selection()
        self.fineSelectionChanged.emit()

    def clear_segment_selection(self):
        """Clear the current segment selection"""
        self._app.state["selection"].clear_segment_selection()
        self.segmentSelectionChanged.emit()

    def get_selection(self) -> 'Optional[Selection]':
        """Get the current selection

        Returns
        -------
        selection : Optional[soundsep.app.services.Selection]
        """
        if self._app.state["selection"].is_set():
            return self._app.state["selection"].get_selection()
        else:
            return None

    def get_fine_selection(self):
        """Get the current fine selection

        Returns
        -------
        selection : Optional[soundsep.app.services.Selection]
        """
        if self._app.state["selection"].is_set():
            return self._app.state["selection"].get_fine_selection()
        else:
            return None

    def get_segment_selection(self):
        """Get the current segment selection"""
        return self._app.state["selection"].get_segment_selection()

    def segment_deleted(self, segId):
        """Notify that a segment has been deleted"""
        self.segmentDeleted.emit(segId)
    
    def segment_created(self, segId):
        """Notify that a segment has been created"""
        self.segmentCreated.emit(segId)

    def needs_saving(self) -> bool:
        """Returns True if sources or any plugins have unsaved changes

        Returns
        -------
        needs_saving : bool
        """
        if self._app.datastore["sources"].needs_saving():
            logger.info("Unsaved changes to sources")
            return True

        for p in self._app.plugins.values():
            if p.needs_saving():
                logger.info("Unsaved changes in plugin {}".format(p.__class__.__name__))
                return True

        return False

    def save(self):
        """Save sources and call save hooks on all plugins

        Returns
        -------
        save_successful : bool
        """
        try:
            self._app.save_sources()
            for name, plugin in self._app.plugins.items():
                plugin.save()
        except Exception as e:
            logger.exception("Exception while saving")
            self._app.panic_save(e)
            return False
        else:
            logger.info("Saved data successfully")
            return True

    def filter_and_ampenv(
            self,
            signal: np.ndarray,
            f0: float,
            f1: float
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter and compute ampenv of a signal

        Arguments
        ---------
        signal : np.ndarray
            A 1-D signal of size N samples or a 2-D signal of size (N, K) where
            K is the number of channels in the signal.
        f0 : float
            Lower frequency to filter signal at
        f1 : float
            Upper frequency to filter signal at

        Returns
        -------
        filtered : np.ndarray
            The bandpass filtered signal of the same shape as the input signal
        ampenv : np.ndarray
            The amplitude envelope computed from the filtered signal after
            rectification and lowpass
        """
        # TODO: seems a bit weird to include this in api but maybe its okay
        # TODO: also, where should the rectivy lowpass be configured?
        f0 = min(max(f0, self._app.config["filter.low"]), self._app.config["filter.high"])
        f1 = max(min(f1, self._app.config["filter.high"]), self._app.config["filter.low"])
        if f0 > f1:
            raise ValueError("Cannot filter with f0 > f1")

        if len(signal) <= 33:
            raise SignalTooShort

        return self._app.services["ampenv"].filter_and_ampenv(
                signal, f0, f1, self._app.config["filter.ampenv_rectify"]
        )
