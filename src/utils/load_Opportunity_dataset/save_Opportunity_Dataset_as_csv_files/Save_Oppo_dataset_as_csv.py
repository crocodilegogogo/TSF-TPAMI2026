"""Opportunity dataset can be Downloaded at https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"""
"""Put the downloaded dataset into the current folder"""
"""Run this py file"""

import os
import csv
import numpy as np
import pandas as pd
import yaml
from pandas import Series

def readOpportunityFiles(filelist, cols, mid_label_to_id, hi_label_to_id, loco_label_to_id):
    data = []
    mid_labels = []
    hi_labels = []
    loco_labels = []
    subject_mapping = []
    base_path = os.path.join('OpportunityUCIDataset', 'datasets')
    assert os.path.exists(base_path), "Please download the dataset first using the script"

    for i, filename in enumerate(filelist):
        
        data_mid            = []
        
        with open(os.path.join(base_path, filename), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            subject_info = [int(filename[1]), filename[6:7]]
            for line in reader:
                elem = []
                for ind in cols:
                    elem.append(line[ind])
                
                data_mid.append([float(x) for x in elem[:-3]])
                mid_labels.append(mid_label_to_id[elem[-1]])
                hi_labels.append(hi_label_to_id[elem[-2]])
                loco_labels.append(loco_label_to_id[elem[-3]])
                subject_mapping.append(subject_info)
        
        data_mid_array = np.array(data_mid).T
        data_mid_array = np.array([Series(i).interpolate() for i in data_mid_array]).T
        data_mid_array[np.isnan(data_mid_array)] = 0
        data_mid = data_mid_array.tolist()
        
        data.extend(data_mid)
        print(i)
        
    return np.asarray(data), np.asarray(mid_labels, dtype=int), np.asarray(hi_labels, dtype=int), np.asarray(loco_labels, dtype=int), np.asarray(subject_mapping)

def prepare_opp_data(save_npy_dir):
    metadata_file = open('metadata.yaml', mode='r')
    metadata = yaml.load(metadata_file, Loader=yaml.FullLoader)['opp_preprocess']
    file_list_nodrill = metadata['file_list']
    
    mid_label_to_id = metadata['mid_label_to_id']
    hi_label_to_id = metadata['hi_label_to_id']
    loco_label_to_id = metadata['loco_label_to_id']

    cols = metadata['columns_list']

    selected_cols = np.asarray(cols)-1
    
    data, mid_labels, hi_labels, loco_labels, subject_mapping = readOpportunityFiles(file_list_nodrill, selected_cols, mid_label_to_id, hi_label_to_id, loco_label_to_id)
    
    shp = data.shape[0]
    combined = pd.DataFrame(np.hstack((data.astype(np.float32), loco_labels.reshape((shp, 1)), mid_labels.reshape((shp, 1)), hi_labels.reshape((shp, 1)), subject_mapping)))
    combined.to_csv(os.path.join(save_npy_dir, 'clean_opp.csv'), index=False)

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory
save_npy_dir = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(CUR_DIR)[0])[0])[0])[0],
                            'datasets','Opportunity')

FEATURES = [str(i) for i in range(97)]
LOCO_LABEL_COL = 97
MID_LABEL_COL = 98
HI_LABEL_COL = 99
SUBJECT_ID = 100
RUN_ID = 101

save_npy_dir_flag = create_directory(save_npy_dir)
prepare_opp_data(save_npy_dir)