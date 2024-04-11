import logging
import json
from functools import partial
from typing import List, Tuple
import concurrent.futures
from soundsig.sound import BioSound
import soundsig.sound as sound

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
class WorkerSignals(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)


class FeatureGenerationPanel(widgets.QWidget):
    segmentSelectionChanged = pyqtSignal(object)


    def __init__(self, parent=None,features=None):
        super().__init__(parent)

        self.FEATURELIST=features
        self.init_ui()

    
    def init_ui(self):
        features = self.FEATURELIST

        layout = widgets.QVBoxLayout()
        # add a generate button
        self.generate_button = widgets.QPushButton("Generate")
        layout.addWidget(self.generate_button)
        self.generate_button.clicked.connect(self.on_button_press)
        

        # add the feature table
        self.table = widgets.QTableWidget(0, len(features)+1)
        self.table.setEditTriggers(widgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.table.setColumnHidden(0, True)
        self.table.setHorizontalHeaderLabels(['segHASH'] + self.FEATURELIST)
        header = self.table.horizontalHeader()
        for i in range(1, len(features)+1):
            header.setSectionResizeMode(i, widgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.setLayout(layout)

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




class FeaturePlugin(BasePlugin):

    SAVE_FILENAME = "features.csv"

    FEATURELIST = [
        "fund", "devfund", "cvfund", "maxfund", "minfund", "F1", "F2", "F3", 
        "sal", "rms", "maxAmp", "meanS", "stdS", "skewS", "kurtS", "entS", 
        "q1", "q2", "q3", "meanT", "stdT", "skewT", "kurtT", "entT"
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.worker_signals = WorkerSignals()
        self.config = dict()
        self.reset_config()
        self.panel = FeatureGenerationPanel(features=self.FEATURELIST)


        self.connect_events()

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
            datastore['features'] = pd.DataFrame(columns= self.FEATURELIST)
        return datastore['features']

    @_feature_datastore.setter
    def _feature_datastore(self, value):
        # check that the value is a pandas dataframe
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Feature datastore must be a pandas dataframe")
        # check that it has the requisite columns
        if not all([c in value.columns for c in self.FEATURELIST]):
            raise ValueError("Featire datastore must have columns %s" % (self.FEATURELIST))
        self._datastore["features"] = value

    def connect_events(self):
        # TODO Connect panel events
        self.panel.generate_button.clicked.connect(self.launch_feature_generation_async)
        self.worker_signals.finished.connect(self.panel.on_finished_signal)
        self.worker_signals.progress.connect(self.panel.on_progress_signal)
        self.api.projectLoaded.connect(self.on_project_ready)

    def on_project_ready(self):
        """Called once"""
        save_file = self.api.paths.save_dir / self.SAVE_FILENAME
        if not save_file.exists():
            # initialize feature_datastore to an empty dataframe
            # with the correct columns
            self._feature_datastore = pd.DataFrame(columns=self.FEATURELIST)
            return
        
        self._feature_datastore = pd.read_csv(save_file)

    def on_project_data_loaded(self):
        """Called each time project data is loaded"""
        self.panel.set_data(self._feature_datastore)
    
    def launch_feature_generation_async(self):
        # launch feature generation in a separate thread
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.worker.submit(self.generate_all_features)
    
    def get_segment_audio(self, segmentID):
        """Get the audio data for a segment"""
        seg = self._datastore['segments'].loc[segmentID]
        sr = self.api.project.sampling_rate
        t, audio = self.api.get_signal(seg.StartIndex, seg.StopIndex)
        audio = audio[:,seg.Source.channel]
        return audio, sr

    def generate_all_features(self, overwrite=False):
        """Generate features for all segments"""
        # first go through all segments and identify the segments
        # that have not been processed yet
        # then generate features for those segments
        # and update the feature_datastore
        
        all_segIDs = self._datastore['segments'].index
        if overwrite:
            processed_segIDs = self._feature_datastore.index
        else:
            processed_segIDs = []

        # get segIDs that have not been processed
        unprocessed_segIDs = [segID for segID in all_segIDs if segID not in processed_segIDs]

        new_segs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            print("Initializing processes")
            for segID in unprocessed_segIDs:
                audio, sr = self.get_segment_audio(segID)
                #futures.append(executor.submit(generate_audio_features, audio, sr, segID))
            print("Processes initialized")
            for future in concurrent.futures.as_completed(futures):
                new_segs.append(future.result())
                self.worker_signals.progress.emit(len(new_segs) / len(unprocessed_segIDs) * 100)
            new_segs = pd.DataFrame(new_segs)
        # for segID in unprocessed_segIDs:
        #     audio, sr = self.get_segment_audio(segID)
        #     new_segs.append(generate_audio_features(audio, sr, segID))
        #     self.worker_signals.progress.emit(len(new_segs) / len(unprocessed_segIDs) * 100)
        new_segs = pd.DataFrame(new_segs)
        new_segs.set_index('SegmentID', inplace=True)
        self._feature_datastore[new_segs.index] = new_segs
        self.worker_signals.finished.emit()
        #self._feature_datastore = pd.concat([self._feature_datastore, new_segs])

#from numba import jit
#@jit(nogil=True)
def _extract_features( audio: np.ndarray, sr: int, segmentID: int, normalize: bool) -> List[float]:
    """Extract features from audio"""
    if normalize:
        audio = audio / np.max(audio)


    f_amp = features_ampenv(audio, sr)
    f_fund = features_fundamental(audio, sr)
    f_formants = features_formants(audio, sr)
    f_spectrum = features_spectrum(audio, sr)
    # combine all these dictionaries
    output_features = {**f_amp, **f_fund, **f_formants, **f_spectrum}
    return output_features

    # return dict({ 'SegmentID': segmentID,
    #                 "fund": fund, "devfund": devfund,
    #                 "cvfund": cvfund, "maxfund": maxfund, "minfund": minfund,
    #                 "F1": meanF1, "F2": meanF2, "F3":meanF3,
    #                 "sal": float(myBioSound.meansal), 
    #                 "rms": float(myBioSound.rms), 
    #                 "maxAmp": float(myBioSound.maxAmp),
    #                 "meanS": float(myBioSound.meanspect), "stdS": float(myBioSound.stdspect),
    #                 "skewS": float(myBioSound.skewspect), "kurtS": float(myBioSound.kurtosisspect), 
    #                 "entS": float(myBioSound.entropyspect),
    #                 "q1": float(myBioSound.q1), "q2": float(myBioSound.q2), "q3": float(myBioSound.q3),                  
    #                 "meanT": float(myBioSound.meantime), "stdT": float(myBioSound.stdtime),
    #                 "skewT": float(myBioSound.skewtime), "kurtT": float(myBioSound.kurtosistime),
    #                 "entT": float(myBioSound.entropytime)})


# from numba import jit
# @jit(nogil=True)
def generate_audio_features(audio, sr, segmentID, normalize=True):
    """Generate features for audio data"""
    return _extract_features(audio, sr, segmentID=segmentID, normalize=normalize)

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
        'rms': audio.std(),
        "meantime": meantime,
        "stdtime": stdtime,
        "skewtime": skewtime,
        "kurtosistime": kurtosistime,
        "entropytime": entropytime,
        "maxAmp": max(amp),
    })

def features_fundamental(audio, sr, maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, minSaliency = 0.5, method='HPS'):
    funds_salience = sound.fundEstOptim(audio, sr, maxFund = maxFund, minFund = minFund, lowFc = lowFc, highFc = highFc, minSaliency = minSaliency, method = method)
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
    formants = sound.formantEstimator(audio, sr, lowFc=lowFc, highFc=highFc, windowFormant = windowFormant,
                                    minFormantFreq = minFormantFreq, maxFormantBW = maxFormantBW )
    F1 = formants[:,0]
    F2 = formants[:,1]
    F3 = formants[:,2]
    # Take the time average formants 
    meanF1 = np.nanmean(F1)
    meanF2 = np.nanmean(F2)
    meanF3 = np.nanmean(F3)
    return dict({
        "meanF1":meanF1,
        "meanF2":meanF2,
        "meanF3":meanF3
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
        "meanspect":meanspect,
        "stdspect":stdspect,
        "skewspect":skewspect,
        "kurtosisspect":kurtosisspect,
        "entropyspect":entropyspect,
        "q1":q1,
        "q2":q2,
        "q3":q3
    })