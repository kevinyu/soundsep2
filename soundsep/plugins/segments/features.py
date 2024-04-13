import logging
import json
import time
from multiprocessing import Queue, Process, Event
import threading
from functools import partial
from typing import List, Tuple
import concurrent.futures
from soundsig.sound import BioSound
import soundsig.sound as sound
import warnings
from scipy.signal import firwin, filtfilt



import PyQt6.QtWidgets as widgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QObject
from PyQt6 import QtGui

from soundsep.core.base_plugin import BasePlugin
from soundsep.core.models import Source, ProjectIndex, StftIndex
from soundsep.core.segments import Segment
from soundsep.core.utils import hhmmss

# TODO move to core or something
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
class WorkerSignals(QObject):
    done_adding = pyqtSignal()
    finished = pyqtSignal()
    progress = pyqtSignal(int)


class FeatureGenerationPanel(widgets.QWidget):
    segmentSelectionChanged = pyqtSignal(object)


    def __init__(self, parent=None,features=None):
        super().__init__(parent)

        self.FEATUREDICT=features        
        self.indiv_features = []
        for k,v in features.items():
            self.indiv_features.extend(v)
        self.init_ui()

    
    def init_ui(self):

        

        layout = widgets.QVBoxLayout()
        # add a generate button
        self.generate_button = widgets.QPushButton("Generate")
        layout.addWidget(self.generate_button)
        self.generate_button.clicked.connect(self.on_button_press)
        
        # Add checkboxes for each feature class
        self.feature_checkboxes = {}
        checkbox_layout = widgets.QHBoxLayout()
        for k in self.FEATUREDICT.keys():
            self.feature_checkboxes[k] = widgets.QCheckBox(k)
            self.feature_checkboxes[k].setChecked(True)

            checkbox_layout.addWidget(self.feature_checkboxes[k])
        layout.addLayout(checkbox_layout)

        # add the feature table
        self.feature_to_column = dict(zip(self.indiv_features, range(1, len(self.indiv_features)+1))) 
        self.table = widgets.QTableWidget(0, len(self.indiv_features)+1)
        self.table.setEditTriggers(widgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.table.setColumnHidden(0, True)
        self.table.setHorizontalHeaderLabels(['segID'] + self.indiv_features)
        header = self.table.horizontalHeader()
        for i in range(1, len(self.indiv_features)+1):
            header.setSectionResizeMode(i, widgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.setLayout(layout)

        # init actions
        self.table.itemSelectionChanged.connect(self.on_click)

    
    def set_feature_selection(self, feature_selections):
        for k,v in feature_selections.items():
            if self.feature_checkboxes[k].isChecked() != v:
                self.feature_checkboxes[k].setChecked(v)
            for feature in self.FEATUREDICT[k]:
                self.table.setColumnHidden(self.feature_to_column[feature], not v)

    def on_button_press(self):
        # Add a progress bar into the qvboxlayout
        self.progress = widgets.QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.layout().insertWidget(1, self.progress)

    def on_progress_signal(self,progress):
        self.progress.setValue(progress)
    
    def on_finished_signal(self):
        self.progress.deleteLater()

    def add_or_edit_row(self, feature_row):
        ind = self._find_segment_row_by_segID(feature_row.name)
        if ind is not None:
            self.table.setSortingEnabled(False)
            # Edit the row
            for i, feature in enumerate(self.indiv_features):
                self.table.setItem(ind, i+1, widgets.QTableWidgetItem(str("%.2f"%feature_row[feature])))
            self.table.setSortingEnabled(True)
            return
        else:
            self.add_row(feature_row)
    def add_row(self, feature_row):
        # Add the row
        self.table.setSortingEnabled(False)
        ind = self.table.rowCount()
        self.table.insertRow(self.table.rowCount())
        self.table.setItem(ind, 0, widgets.QTableWidgetItem(str(feature_row.name)))
        for i, feature in enumerate(self.indiv_features):
            self.table.setItem(ind, i+1, widgets.QTableWidgetItem(str("%.2f"%feature_row[feature])))
        self.table.setSortingEnabled(True)

    # Selection functions
    def on_click(self):
        selection = self.get_selection()
        self.segmentSelectionChanged.emit(selection)

    def on_selection_changed(self, selection):
        self.table.itemSelectionChanged.disconnect(self.on_click)
        self.set_selection(selection)
        self.table.itemSelectionChanged.connect(self.on_click)



    def _find_segment_row_by_segID(self, seg_id):
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).text() == str(seg_id):
                return i
        return None
    
    def remove_row_by_segID(self, seg_id):
        ind = self._find_segment_row_by_segID(seg_id)
        if seg_id in self.get_selection():
            self.table.clearSelection()
        if ind is not None:
            self.table.removeRow(ind)
        else:
            raise ValueError("Cannot remove Segment ID {}: not found in table".format(seg_id))


    def get_selection(self):
        selection = []
        ranges = self.table.selectedRanges()
        for selection_range in ranges:
            selection += list(range(selection_range.topRow(), selection_range.bottomRow() + 1))
        # Get IDS for each selected ROW
        ids = [int(self.table.item(row, 0).text()) for row in selection]
        return sorted(ids)

    def set_selection(self, selection):
        self.table.clearSelection()
            
        for seg_id in selection:
            table_ind = self._find_segment_row_by_segID(seg_id)
            if table_ind is not None:
                self.table.selectRow(table_ind)
            else:
                return
            
    def set_data(self, feature_db):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.table.setRowCount(len(feature_db))
        ix = 0
        for row, feature_row in feature_db.iterrows():
            self.table.setItem(ix, 0, widgets.QTableWidgetItem(str(row)))
            for i, feature in enumerate(self.indiv_features):
                self.table.setItem(ix, i+1, widgets.QTableWidgetItem(str("%.2f"%feature_row[feature])))
            ix += 1
        self.table.setSortingEnabled(True)

class FeaturePlugin(BasePlugin):

    SAVE_FILENAME = "features.csv"
    FEATUREDICT = dict({
        "Amplitude":["rms","meantime","stdT","skewT","kurtT","entT","maxAmp"],
        "Fundamental":["fund","sal","fund2","sal2","maxfund","minfund","cvfund","cvfund2","devfund"],
        "Formant":["F1","F2","F3"],
        "Spectrum": ["meanS","stdS","skewS","kurtS","entS","q1","q2","q3"]
    })
    @property
    def featurelist(self):
        features = self.FEATUREDICT
        indiv_features = []
        for k,v in features.items():
            indiv_features.extend(v)
        return indiv_features
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.worker_signals = WorkerSignals()
        self.config = dict()
        self.reset_config()
        self.panel = FeatureGenerationPanel(features=self.FEATUREDICT)
        self.feature_selections = { k: True for k in self.FEATUREDICT.keys()}

        self.connect_events()

        self._needs_saving = False

    def feature_selection_changed(self, feature_class, state):
        self.feature_selections[feature_class] = state == 2
        self.panel.set_feature_selection(self.feature_selections)

    def connect_events(self):
        # TODO Connect panel events
        self.panel.generate_button.clicked.connect(self.launch_feature_generation_async)
        for k in self.FEATUREDICT.keys():
            self.panel.feature_checkboxes[k].stateChanged.connect(partial(self.feature_selection_changed, k))
        self.panel.segmentSelectionChanged.connect(self.api.set_segment_selection)
        
        self.worker_signals.finished.connect(self.panel.on_finished_signal)
        self.worker_signals.progress.connect(self.panel.on_progress_signal)
        
        # connect to api
        self.api.segmentSelectionChanged.connect(self.on_segment_selection_changed)
        self.api.projectLoaded.connect(self.on_project_ready)
        self.api.projectDataLoaded.connect(self.on_project_data_loaded)


    def on_segment_selection_changed(self):
        selection = self.api.get_segment_selection()
        self.panel.on_selection_changed(selection)

    def plugin_panel_widget(self):
        return [self.panel]
    # def add_plugin_menu(self, menu_parent):
    #     menu = menu_parent.addMenu("&Segments")
    #     menu.addAction(self.generate_all_features)
    #     menu.addAction(self.delete_selection_action)
    #     menu.addAction(self.merge_selection_action)
    #     return menu
    def reset_config(self):
        self.config['ncores'] = 6
        self.config['overwrite'] = False

    @property
    def _datastore(self):
        return self.api.get_mut_datastore()

    @property
    def _feature_datastore(self):
        datastore = self._datastore
        if 'features' not in datastore:
            datastore['features'] = pd.DataFrame(columns= self.featurelist)
        return datastore['features']

    @_feature_datastore.setter
    def _feature_datastore(self, value):
        # check that the value is a pandas dataframe
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Feature datastore must be a pandas dataframe")
        # check that it has the requisite columns
        if not all([c in value.columns for c in self.featurelist]):
            raise ValueError("Featire datastore must have columns %s" % (self.featurelist))
        self._datastore["features"] = value

    def needs_saving(self):
        return self._needs_saving

    def on_project_ready(self):
        """Called once"""
        save_file = self.api.paths.save_dir / self.SAVE_FILENAME
        if not save_file.exists():
            # initialize feature_datastore to an empty dataframe
            # with the correct columns
            self._feature_datastore = pd.DataFrame(columns=self.featurelist)
            return
        
        self._feature_datastore = pd.read_csv(save_file, index_col=0)

    def on_project_data_loaded(self):
        """Called each time project data is loaded"""
        self.panel.set_data(self._feature_datastore)
    
    def launch_feature_generation_async(self):
        # launch feature generation in a separate thread
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.worker.submit(self.generate_all_features)
    
    def get_segment_audio(self, segmentID, lowpass=6000, highpass=200):
        """Get the audio data for a segment"""
        seg = self._datastore['segments'].loc[segmentID]
        sr = self.api.project.sampling_rate
        t, audio = self.api.get_signal(seg.StartIndex, seg.StopIndex)
        audio = audio[:,seg.Source.channel]

        # Apply filtering
        # high pass filter the signal
        nfilt = 1024
        soundLen = len(audio)
        highpassFilter = firwin(nfilt-1, 2.0*highpass/sr, pass_zero=False)
        padlen = min(soundLen-10, 3*len(highpassFilter))
        soundIn = filtfilt(highpassFilter, [1.0], audio, padlen=padlen)

        # low pass filter the signal
        lowpassFilter = firwin(nfilt, 2.0*lowpass/sr)
        padlen = min(soundLen-10, 3*len(lowpassFilter))
        soundIn = filtfilt(lowpassFilter, [1.0], audio, padlen=padlen)

        return audio, sr

    def generate_all_features(self, overwrite=False):
        """Generate features for all segments"""
        # first go through all segments and identify the segments
        # that have not been processed yet
        # then generate features for those segments
        # and update the feature_datastore
        
        all_segIDs = self._datastore['segments'].index
        if overwrite:
            processed_segIDs = []
        else:
            processed_segIDs = self._feature_datastore.index

        # get segIDs that have not been processed
        unprocessed_segIDs = [segID for segID in all_segIDs if segID not in processed_segIDs]

        # TODO CHeck if the other ones have had the feautres i want extracted
        # if not, then add them to the unprocessed_segIDs
        for segID in processed_segIDs:
            for feature_cat in self.feature_selections.keys():
                if self.feature_selections[feature_cat]:
                    if all([np.isnan(self._feature_datastore.at[segID, feature]) for feature in self.FEATUREDICT[feature_cat]]):
                        unprocessed_segIDs.append(segID)
                        break   

        nworkers = 6
        done_prep_event = Event()
        audio_queue = Queue()
        feature_queue = Queue()

        feature_processes = []
        for i in range(nworkers):
            feature_processes.append(FeatureExtractionProcess(audio_queue, feature_queue, done_prep_event, self.feature_selections))
            feature_processes[-1].start()
        
        loading_thread = threading.Thread(target=load_all_audio, args=(self.get_segment_audio, unprocessed_segIDs, audio_queue, 4*nworkers, done_prep_event))
        loading_thread.start()

        n_complete = 0
        while np.any([p.is_alive() for p in feature_processes]) or not feature_queue.empty():
            try:
                segmentID, all_features = feature_queue.get(timeout=.5)

                if segmentID not in self._feature_datastore.index:
                    self._feature_datastore.loc[segmentID] = pd.Series()
                for k,v in all_features.items():
                    # Outer is Amplitude etc
                    for kk,vv in v.items():
                        # inner is columns
                        self._feature_datastore.at[segmentID, kk] = vv
                n_complete += 1
                # This should be on a signal probably
                self.panel.add_or_edit_row(self._feature_datastore.loc[segmentID])
                self.worker_signals.progress.emit(n_complete / len(unprocessed_segIDs) * 100)
            except Exception as e:
                continue
        print("OUT O HERE")
        self.worker_signals.finished.emit()
        self._needs_saving = True
        # do the cleanup
        for p in feature_processes:
            p.join()
        loading_thread.join()


    def save(self):
        save_file = self.api.paths.save_dir / self.SAVE_FILENAME
        self._feature_datastore.to_csv(save_file, index_label='SegmentID')
        self._needs_saving = False

def progress_updater(progress_queue_in, progress_signal_out, n):
    n_done = 0
    # loop until done or sigabbrt
    while n_done < n:
        seg_id_done = progress_queue_in.get()
        n_done += 1
        progress_signal_out.emit(n_done / n * 100)

def load_all_audio(load_func, seg_ids, queue, max_queue_size, done_event):
    for seg_id in seg_ids:
        audio, sr = load_func(seg_id)
        while queue.qsize() > max_queue_size:
            time.sleep(.1)
        queue.put((seg_id, audio, sr))
    done_event.set()
class FeatureExtractionProcess(Process):
    def __init__(self, in_queue, out_queue, stop_signal, feature_selections):
        super().__init__()
        self.input_queue = in_queue
        self.stop_signal = stop_signal
        self.output_queue = out_queue
        self.feature_selections = feature_selections

    def run(self):
        print("Beginning Feature Extraction Process")
        while not self.stop_signal.is_set() or not self.input_queue.empty():
            try:
                segmentID, audio, sr = self.input_queue.get(timeout=.5)
                features = generate_audio_features(audio, sr, segmentID, self.feature_selections)
                self.output_queue.put(features)
            except Exception as e:
                continue
        print("Exiting feature extraction process")
#from numba import jit
#@jit(nogil=True)
def _extract_features( audio: np.ndarray, sr: int, feature_selections: dict, normalize: bool) -> List[float]:
    """Extract features from audio"""
    if normalize:
        audio = audio / np.max(audio)
    output_features = dict() 
    if feature_selections is None or feature_selections['Amplitude']:
        output_features['Amplitude'] = features_ampenv(audio, sr)
    if feature_selections is None or feature_selections['Fundamental']:
        output_features['Fundamental'] = features_fundamental(audio, sr)
    if feature_selections is None or feature_selections['Formant']:
        output_features['Formant'] = features_formants(audio, sr)
    if feature_selections is None or feature_selections['Spectrum']:
        output_features['Spectrum'] = features_spectrum(audio, sr)
    return output_features


# from numba import jit
# @jit(nogil=True)
def generate_audio_features(audio, sr, segmentID, feature_selections = None, normalize=True):
    """Generate features for audio data"""
    return segmentID, _extract_features(audio, sr, feature_selections = feature_selections, normalize=normalize)

def generate_segment_features(segmentID, seg_datastore, api):
    """Generate features for a single segment"""
    # get the segment
    seg = seg_datastore.loc[segmentID]
    sr = api.project.sampling_rate
    # get the audio data for the segment
    # TODO could pad small segments here
    t, audio = api.get_signal(seg.StartIndex, seg.StopIndex)
    audio = audio[:,seg.Source.channel]
    # get the sampling rate
    # get the features
    features = _extract_features(audio, sr, segmentID=segmentID, normalize=True)
    return features


# TODO Move this to a utils file
def features_ampenv(audio, sr, cutoff_freq = 20, amp_sample_rate = 1000):
    # Calculates the amplitude enveloppe and related parameters
    (amp, tdata)  = sound.temporal_envelope(audio, sr, cutoff_freq=cutoff_freq, resample_rate=amp_sample_rate)
    
    # Here are the parameters
    ampdata = amp/np.sum(amp)
    meantime = np.sum(tdata*ampdata)
    stdtime = np.sqrt(np.sum(ampdata*((tdata-meantime)**2)))
    skewtime = np.sum(ampdata*(tdata-meantime)**3)
    skewtime = skewtime/(stdtime**3)
    kurtosistime = np.sum(ampdata*(tdata-meantime)**4)
    kurtosistime = kurtosistime/(stdtime**4)
    indpos = np.where(ampdata>0)[0]
    entropytime = -np.sum(ampdata[indpos]*np.log2(ampdata[indpos]))/np.log2(np.size(indpos))
    return dict({
        "rms": audio.std(),
        "meantime": meantime,
        "stdT": stdtime,
        "skewT": skewtime,
        "kurtT": kurtosistime,
        "entT": entropytime,
        "maxAmp": max(amp),
    })

def features_fundamental(audio, sr, maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, minSaliency = 0.5, method='HPS'):
    funds_salience = sound.fundEstOptim(audio, sr, maxFund = maxFund, minFund = minFund, nofilt=True, lowFc = lowFc, highFc = highFc, minSaliency = minSaliency, method = method)
    f0 = funds_salience[:,0]
    f0_2 = funds_salience[:,1]
    sal = funds_salience[:,2]
    sal_2 = funds_salience[:,3]
    fund = np.nanmean(funds_salience[:,0])
    meansal = np.nanmean(funds_salience[:,2])
    fund2 = np.nanmean(funds_salience[:,1])
    meansal2 = np.nanmean(funds_salience[:,3])
    maxfund = np.nanmax(f0)
    minfund = np.nanmin(f0)
    cvfund = np.nanstd(f0)/fund
    cvfund2 = np.nanstd(f0_2)/fund2
    devfund = np.nanmean(np.diff(f0))
    return dict({
        "fund":fund,
        "sal":meansal,
        "fund2":fund2,
        "sal2":meansal2,
        "maxfund":maxfund,
        "minfund":minfund,
        "cvfund":cvfund,
        "cvfund2":cvfund2,
        "devfund":devfund
        })
    #self.voice2percent = np.nanmean(funds_salience[:,4])*100

def features_formants(audio, sr,  lowFc = 200, highFc = 6000, minFormantFreq = 500, maxFormantBW = 500, windowFormant = 0.1):
    formants = sound.formantEstimator(audio, sr, nofilt=True, lowFc=lowFc, highFc=highFc, windowFormant = windowFormant,
                                    minFormantFreq = minFormantFreq, maxFormantBW = maxFormantBW )
    F1 = formants[:,0]
    F2 = formants[:,1]
    F3 = formants[:,2]
    # Take the time average formants 
    meanF1 = np.nanmean(F1)
    meanF2 = np.nanmean(F2)
    meanF3 = np.nanmean(F3)
    return dict({
        "F1":meanF1,
        "F2":meanF2,
        "F3":meanF3
    })


def features_spectrum(audio, sr, f_high = 10000):
    Pxx, Freqs = sound.mlab.psd(audio,Fs=sr,NFFT=1024, noverlap=512)
    
    # Find quartile power
    cum_power = np.cumsum(Pxx)
    tot_power = np.sum(Pxx)
    quartile_freq = np.zeros(3, dtype = 'int')
    quartile_values = [0.25, 0.5, 0.75]
    nfreqs = np.size(cum_power)
    iq = 0
    for ifreq in range(nfreqs):
        if (cum_power[ifreq] > quartile_values[iq]*tot_power):
            quartile_freq[iq] = ifreq
            iq = iq+1
            if (iq > 2):
                break
                
    # Find skewness, kurtosis and entropy for power spectrum below f_high
    ind_fmax = np.where(Freqs > f_high)[0][0]

    # Description of spectral shape
    spectdata = Pxx[0:ind_fmax]
    freqdata = Freqs[0:ind_fmax]
    spectdata = spectdata/np.sum(spectdata)
    meanspect = np.sum(freqdata*spectdata)
    stdspect = np.sqrt(np.sum(spectdata*((freqdata-meanspect)**2)))
    skewspect = np.sum(spectdata*(freqdata-meanspect)**3)
    skewspect = skewspect/(stdspect**3)
    kurtosisspect = np.sum(spectdata*(freqdata-meanspect)**4)
    kurtosisspect = kurtosisspect/(stdspect**4)
    entropyspect = -np.sum(spectdata*np.log2(spectdata))/np.log2(ind_fmax)

    # Storing the values       
    meanspect = meanspect
    stdspect = stdspect
    skewspect = skewspect
    kurtosisspect = kurtosisspect
    entropyspect = entropyspect
    q1 = Freqs[quartile_freq[0]]
    q2 = Freqs[quartile_freq[1]]
    q3 = Freqs[quartile_freq[2]]

    return dict({
        "meanS":meanspect,
        "stdS":stdspect,
        "skewS":skewspect,
        "kurtS":kurtosisspect,
        "entS":entropyspect,
        "q1":q1,
        "q2":q2,
        "q3":q3
    })