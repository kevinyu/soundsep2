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



class VisualizationPanel(widgets.QWidget):
    segmentSelectionChanged = pyqtSignal(object)
    def __init__(self, parent=None, api=None):
        super().__init__(parent)
        self.api=api
        self.init_ui()
        self.init_actions()
        self.npoints = 0

    def init_ui(self):
        # setup a 2d plot
        layout = widgets.QVBoxLayout()
        # Add two drop down menus for x and y axis features
        self.x_axis = widgets.QComboBox()
        self.y_axis = widgets.QComboBox()
        x_y_layout = widgets.QHBoxLayout()
        x_y_layout.addWidget(widgets.QLabel("X-Axis:"))
        x_y_layout.addWidget(self.x_axis)
        x_y_layout.addWidget(widgets.QLabel("Y-Axis:"))
        x_y_layout.addWidget(self.y_axis)
        layout.addLayout(x_y_layout)
        # Add the plot
        self.plot = pg.plot()
        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)
        #layout = widgets.QGridLayout()
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def add_features_to_dropdown(self, features):
        self.x_axis.addItems(features)
        self.y_axis.addItems(features)
    
    def init_actions(self):
        self.x_axis.currentIndexChanged.connect(self.on_x_axis_change)
        self.y_axis.currentIndexChanged.connect(self.on_y_axis_change)
        self.scatter.sigClicked.connect(self.on_click)
        return
    
    def on_click(self, plot, points):
        if len(points) > 0:
            # TODO what to do for multiselect
            self.segmentSelectionChanged.emit([points[0].data()])
        return

    def on_x_axis_change(self, ind):
        self.update_spots()

    def on_y_axis_change(self, ind):
        self.update_spots()

    def on_selection_changed(self, selection):
        sizes = np.ones(self.npoints) * 10
        spot_inds = [spot['data'] for spot in self.scatter.data]
        sel_inds = []
        for s in selection:
            if s in spot_inds:
                sel_inds.append(spot_inds.index(s))
        if len(sel_inds) > 0:
            sizes[sel_inds] = 20
        self.scatter.setSize(sizes)
    
    def set_data(self, features, func_get_color=None):
        spots = []
        for ix,feat_row in features.iterrows():
            tags = self.api.plugins['segments'].get_tags_for_segment(ix)
            if func_get_color and len(s_row['Tags']) > 0:
                c = func_get_color(list(s_row['Tags'])[0])
            else:
                c = 'r'
            if s_row['Coords'] != None and len(s_row['Coords']) >= 2:
                spots.append(dict({
                    'pos': s_row['Coords'][:2],
                    'data': ix,
                    'brush': pg.mkBrush(c),
                    'size': 10
                }))
        
        self.npoints = len(spots)
        self.scatter.setData(
            spots=spots,
            hoverSize=20,
            hoverable=True
        )
    
    def remove_spots(self, segIDs):
        visibilities = self.scatter.data['visible']
        spot_seg_IDs = [spot['data'] for spot in self.scatter.data]
        for segID in segIDs:
            if segID in spot_seg_IDs:
                visibilities[spot_seg_IDs.index(segID)] = False
                #spot_inds.append(seg_IDs.index(segID))
        self.scatter.setPointsVisible(visibilities)

    def add_spot(self, segID, coords, color='r'):
        self.scatter.addPoints(
            pos=[coords],
            data=segID,
            brush=pg.mkBrush(color),
            size=10
        )
        self.npoints += 1

    def update_spots(self, func_get_color=None):
        spot_seg_IDs = self.scatter.data['data']
        spot_brushes = [spot['brush'] for spot in self.scatter.data]
        mut_ds = self.api.get_mut_datastore()
        seg_db = mut_ds['segments']
        feat_db = mut_ds['features']
        # confirm that the x and y axis are in the features
        x_axis = self.x_axis.currentText()
        y_axis = self.y_axis.currentText()

        if x_axis == "" or y_axis == "":
            return

        if x_axis not in feat_db.columns or y_axis not in feat_db.columns:
            return

        # only take segments that are in the feature db
        segments = seg_db.loc[feat_db.index]
        data = self.scatter.data
        if spot_seg_IDs != []:
            data['x'] = feat_db[x_axis].loc[spot_seg_IDs]
            data['y'] = feat_db[y_axis].loc[spot_seg_IDs]
            self.scatter.updateSpots()
            vb = self.scatter.getViewBox()
            xrange = self.scatter.dataBounds(0)
            vb.setXRange(xrange[0], xrange[1])
            yrange = self.scatter.dataBounds(1)
            vb.setYRange(yrange[0], yrange[1])
            # self.scatter.invalidate()
        segs_to_add = []
        segments_not_present = segments[~segments.index.isin(spot_seg_IDs)]
        for ix, s_row in segments_not_present.iterrows():
            if feat_db.loc[ix][x_axis] is not np.nan and feat_db.loc[ix][y_axis] is not np.nan:
                segs_to_add.append((s_row, [feat_db.loc[ix][x_axis], feat_db.loc[ix][y_axis]]))

        # now add the ones that were not present
        for s_row, coords in segs_to_add:
            self.add_spot(s_row.name, coords)# TODO func_get_color

class DimensionalityReductionWizard(widgets.QWidget):
    """ Window for selecting features to include in PCA"""
    feature_generation_signal = pyqtSignal(object, str, str)
    def __init__(self,parent=None, features=None, feature_percents=None, feat_check_callback=None):
        super().__init__(parent)
        self.features = features
        self.feat_check_callback = feat_check_callback
        self.feature_percents = feature_percents
        self.init_ui()
    
    def init_ui(self):
        layout = widgets.QVBoxLayout()
        # Add checkboxes for each feature class
        self.feature_checkboxes = {}
        checkbox_layout = widgets.QHBoxLayout()
        sub_layouts = {}
        for k in self.features.keys():
            sub_layouts[k] = widgets.QVBoxLayout()
            sub_layouts[k].addWidget(widgets.QLabel(k))
            for ix,kk in enumerate(self.features[k]):
                self.feature_checkboxes[kk] = widgets.QCheckBox(kk)
                sub_layouts[k].addWidget(self.feature_checkboxes[kk])
                pcen = self.feature_percents[k][ix]
                sub_layouts[k].addWidget(widgets.QLabel("NaNs: %.2f" % (pcen*100)))
                self.feature_checkboxes[kk].setChecked(pcen < .1)
                self.feature_checkboxes[kk].stateChanged.connect(self.on_box_checked)
            checkbox_layout.addLayout(sub_layouts[k])
        layout.addLayout(checkbox_layout)

        # add a label for total number of stims
        n_good, n_total = self.feat_check_callback(self.get_selected_features())
        self.n_stims_label = widgets.QLabel("Number of stims: %d/%d" % (n_good, n_total))
        layout.addWidget(self.n_stims_label)

        # add a drop down for type of dimensionality reduction
        self.dim_reduction_type = widgets.QComboBox()
        self.dim_reduction_type.addItems(["PCA", "UMAP"])
        layout.addWidget(self.dim_reduction_type)

        # add a clear all boxes button
        self.clear_all_button = widgets.QPushButton("Uncheck All")
        self.clear_all_button.clicked.connect(self.clear_checkboxes)
        layout.addWidget(self.clear_all_button)

        # TODO can add some params here

        # Add a generate button with a field for the name of the new feature
        generation_layout = widgets.QHBoxLayout()
        self.feat_name = widgets.QLineEdit("DimRed1")
        generation_layout.addWidget(widgets.QLabel("Feature Name:"))
        generation_layout.addWidget(self.feat_name)
        self.generate_button = widgets.QPushButton("Generate")
        self.generate_button.clicked.connect(self.on_generate_button_press)
        generation_layout.addWidget(self.generate_button)
        layout.addLayout(generation_layout)
        self.setLayout(layout)

    def on_box_checked(self, state):
        # check how many stims are good for these features
        if self.feat_check_callback:
            n_good, n_total = self.feat_check_callback(self.get_selected_features())
            self.n_stims_label.setText("Number of stims: %d/%d" % (n_good, n_total))
    def clear_checkboxes(self):
        for k,v in self.feature_checkboxes.items():
            v.setChecked(False)
    def get_selected_features(self):
        selected_features = []
        for k,v in self.feature_checkboxes.items():
            if v.isChecked():
                selected_features.append(k)
        return selected_features

    def on_generate_button_press(self):
        selected_features = self.get_selected_features()
        dim_reduction_type = self.dim_reduction_type.currentText()
        new_feature_name = self.feat_name.text()
        self.feature_generation_signal.emit(selected_features, dim_reduction_type, new_feature_name)

class FeatureGenerationPanel(widgets.QWidget):
    segmentSelectionChanged = pyqtSignal(object)
    def __init__(self, parent=None,features=None):
        super().__init__(parent)

        self.FEATUREDICT=features        
        self.indiv_features = []
        if features is not None:
            for k,v in features.items():
                self.indiv_features.extend(v)
        self.init_ui()

    
    def init_ui(self):
        layout = widgets.QVBoxLayout()
        # add a generate button
        self.generate_button = widgets.QPushButton("Generate")
        layout.addWidget(self.generate_button)
        self.generate_button.clicked.connect(self.on_generate_button_press)
        
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

        # Dim reduction generation button
        self.generate_DR_button = widgets.QPushButton("Generate Dimensionality Reduction")
        layout.addWidget(self.generate_DR_button)

        self.setLayout(layout)

        # init actions
        self.table.itemSelectionChanged.connect(self.on_click)

    
    def set_feature_selection(self, feature_selections):
        for k,v in feature_selections.items():
            if self.feature_checkboxes[k].isChecked() != v:
                self.feature_checkboxes[k].setChecked(v)
            for feature in self.FEATUREDICT[k]:
                self.table.setColumnHidden(self.feature_to_column[feature], not v)

    def on_generate_button_press(self):
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

    def add_feature(self, feature_name, df_feature_data):
        self.indiv_features.append(feature_name)
        self.table.insertColumn(self.table.columnCount())
        self.table.setHorizontalHeaderItem(self.table.columnCount()-1, widgets.QTableWidgetItem(feature_name))
        self.table.setColumnHidden(self.table.columnCount()-1, True)
        self.feature_to_column[feature_name] = self.table.columnCount()-1
        for i in range(self.table.rowCount()):
            segID = int(self.table.item(i, 0).text())
            if segID in df_feature_data.index:
                self.table.setItem(i, self.table.columnCount()-1, widgets.QTableWidgetItem(str("%.2f"%df_feature_data.at[segID])))

        

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

from sklearn.decomposition import PCA
import umap
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

    def current_featurelist(self):
        feature_selections = self.feature_selections
        features = self.FEATUREDICT
        indiv_features = []
        for k,v in features.items():
            if feature_selections[k]:
                indiv_features.extend(v)
        return indiv_features

    def get_custom_features(self):
        feature_list = self.current_featurelist()
        # now get the custom features
        col_names = self._datastore['features'].columns
        custom_features = []
        for feature in col_names:
            if feature not in feature_list:
                custom_features.append(feature)
        return custom_features
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.worker_signals = WorkerSignals()
        self.config = dict()
        self.reset_config()
        self.panel = FeatureGenerationPanel(features=self.FEATUREDICT)
        self.vis_panel = VisualizationPanel(api=self.api)
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
        self.panel.generate_DR_button.clicked.connect(self.on_dim_reduce_button_press)
        self.vis_panel.segmentSelectionChanged.connect(self.api.set_segment_selection)

        self.worker_signals.finished.connect(self.panel.on_finished_signal)
        self.worker_signals.progress.connect(self.panel.on_progress_signal)
        
        # connect to api
        self.api.segmentSelectionChanged.connect(self.on_segment_selection_changed)
        self.api.projectLoaded.connect(self.on_project_ready)
        self.api.projectDataLoaded.connect(self.on_project_data_loaded)
        self.api.segmentDeleted.connect(self.on_segment_deleted)


    def on_segment_selection_changed(self):
        selection = self.api.get_segment_selection()
        self.panel.on_selection_changed(selection)
        self.vis_panel.on_selection_changed(selection)

    def plugin_panel_widget(self):
        return [self.panel, self.vis_panel]
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
        self.vis_panel.add_features_to_dropdown(self.featurelist)
        self.vis_panel.update_spots()
    
    def get_feat_percent(self, feature):
        feature_db = self._feature_datastore[feature]
        return feature_db.isnull().mean()

    def get_number_of_stim_for_selection(self, features):
        feature_db = self._feature_datastore[features]
        feature_db_tmp = feature_db.dropna()
        return len(feature_db_tmp),len(feature_db)

    def on_dim_reduce_button_press(self):
        # Make a popup window to select the features to include
        # Then generate the PCA features
        features_to_include = self.FEATUREDICT.copy()
        custom_feats = self.get_custom_features()
        if len(custom_feats) > 0:
            features_to_include['Custom'] = self.get_custom_features()
        feat_percents = {}
        for k in features_to_include.keys():
            feat_percents[k] = [self.get_feat_percent(f) for f in features_to_include[k]]

        self.dim_red_window = DimensionalityReductionWizard(features=features_to_include, feature_percents=feat_percents, feat_check_callback = self.get_number_of_stim_for_selection)
        self.dim_red_window.feature_generation_signal.connect(self.on_dim_reduction_generate)
        self.dim_red_window.show()
    
    def on_dim_reduction_generate(self, features, dim_reduction_type, new_feature_name):
        # TODO generate the new feature
        print(features, dim_reduction_type, new_feature_name)
        if dim_reduction_type == "PCA":
            self.generate_PCA_Feature(new_feature_name,features)
        self.dim_red_window.close()

    def generate_PCA_Feature(self, feat_name, features):
        """Generates PCA Feature based on currently visible features"""
        feature_db = self._feature_datastore[features]
        feature_db = feature_db.dropna()
        data = feature_db.to_numpy()
        # todo could balance across channels
        Zdata = (data - data.mean(axis=0))/ np.std(data,axis=0,ddof=1)

        # PCA the data
        pca = PCA(n_components=10, svd_solver='full')
        Z_PCA_DATA = pca.fit_transform(Zdata)
        # Add the PCA data to the feature db
        for i in range(Z_PCA_DATA.shape[1]):
            self._feature_datastore.loc[feature_db.index, feat_name + str(i)] = Z_PCA_DATA[:,i]
            self.panel.add_feature(feat_name + str(i),  self._feature_datastore[feat_name + str(i)])
            self.vis_panel.add_features_to_dropdown([feat_name + str(i)])

    def generate_UMAP_Feature(self, feat_name, features):
        pass


# FEATURE GENERATION
    def launch_feature_generation_async(self):
        # launch feature generation in a separate thread
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.worker.submit(self.generate_all_features)

    def on_segment_deleted(self, segmentID):
        self._feature_datastore.drop(segmentID, inplace=True)
        self._needs_saving = True
        self.panel.remove_row_by_segID(segmentID)
        self.vis_panel.remove_spots([segmentID])
    
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
    if np.isnan(f0).all():
        fund = np.nan
        maxfund = np.nan
        minfund = np.nan
        cvfund = np.nan
        devfund = np.nan
    else:
        meansal = np.nanmean(sal)
        fund = np.nanmean(f0)
        maxfund = np.nanmax(f0)
        minfund = np.nanmin(f0)
        cvfund = np.nanstd(f0)/fund
        devfund = np.nanmean(np.diff(f0))
    if np.isnan(f0_2).all():
        fund2 = np.nan
        meansal2 = np.nan
        cvfund2 = np.nan
    else:
        fund2 = np.nanmean(f0_2)
        cvfund2 = np.nanstd(f0_2)/fund2
        meansal2 = np.nanmean(sal_2)
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

def features_formants(audio, sr,  lowFc = 200, highFc = 6000, minFormantFreq = 500, maxFormantBW = 1000, windowFormant = 0.1):
    formants = sound.formantEstimator(audio, sr, nofilt=True, lowFc=lowFc, highFc=highFc, windowFormant = windowFormant,
                                    minFormantFreq = minFormantFreq, maxFormantBW = maxFormantBW )
    F1 = formants[:,0]
    F2 = formants[:,1]
    F3 = formants[:,2]

    # Take the time average formants only if there are some non nan numbers
    if np.sum(~np.isnan(F1)) > 0:
        meanF1 = np.nanmean(F1)
    else:
        meanF1 = np.nan
    if np.sum(~np.isnan(F2)) > 0:
        meanF2 = np.nanmean(F2)
    else:
        meanF2 = np.nan
    if np.sum(~np.isnan(F3)) > 0:
        meanF3 = np.nanmean(F3)
    else:
        meanF3 = np.nan

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