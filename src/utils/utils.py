"""Collection of utility functions"""
from datetime import datetime
from logging import getLogger, Formatter, FileHandler, DEBUG, WARNING
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from scipy.fftpack import fft
from thop import profile
from thop import clever_format
from utils.constants import *
from utils.load_HAPT_dataset.load_HAPT_dataset import load_HAPT_raw_data
from utils.load_Motion_Sense_dataset.load_Motion_Sense_dataset import load_Motion_Sense_raw_data
from utils.load_SHL2018_dataset.load_SHL2018_dataset import load_SHL2018_raw_data
from utils.load_HHAR_dataset.load_HHAR_dataset import load_HHAR_raw_data
from utils.load_MobiAct_dataset.load_MobiAct_dataset import load_MobiAct_raw_data
from utils.load_Opportunity_dataset.load_Opportunity_dataset import load_Opportunity_data
from utils.load_Pamap2_dataset.load_Pamap2_dataset import load_Pamap2_data
from utils.load_RealWorld_dataset.load_RealWorld_dataset import load_RealWorld_data
from utils.load_DSADS_dataset.load_DSADS_dataset import load_DSADS_data
from utils.load_SHO_dataset.load_SHO_dataset import load_SHO_data
from utils.hf_downloader import ensure_dataset_available

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import time
import random

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# Avoid parsing command-line arguments at import time.  The main entry point
# calls configure_runtime(args) after parse_args().
logger = getLogger(__name__)

class _RuntimeArgs:
    INFERENCE_DEVICE = os.environ.get("TSF_INFERENCE_DEVICE", "TEST_CUDA")
    seed = int(os.environ.get("TSF_SEED", "6"))

args = _RuntimeArgs()

def configure_runtime(parsed_args):
    """Register CLI arguments for utility functions without import-time argparse side effects."""
    global args
    args = parsed_args
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_random_seed(getattr(parsed_args, "seed", 6))

def get_runtime_device():
    """Return the active torch device while gracefully falling back to CPU."""
    use_cuda = getattr(args, "INFERENCE_DEVICE", "TEST_CUDA") == "TEST_CUDA" and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")

def set_random_seed(seed: int = 6):
    """Set common random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def round_float(f: float, r: float = 0.000001) -> float:
    return float(Decimal(str(f)).quantize(Decimal(str(r)), rounding=ROUND_HALF_UP))


def round_list(l: List[float], r: float = 0.000001) -> List[float]:
    return [round_float(f, r) for f in l]


def round_dict(d: Dict[Any, Any], r: float = 0.000001) -> Dict[Any, Any]:
    return {key: round(d[key], r) for key in d.keys()}


def round(arg: Any, r: float = 0.000001) -> Any:
    if type(arg) == float or type(arg) == np.float64 or type(arg) == np.float32:
        return round_float(arg, r)
    elif type(arg) == list or type(arg) == np.ndarray:
        return round_list(arg, r)
    elif type(arg) == dict:
        return round_dict(arg, r)
    else:
        logger.error(f"Arg type {type(arg)} is not supported")
        return arg

def get_hyperparams(path, dataset_name):
    
    hparam_file     = open(path, mode='r', encoding='utf-8')
    hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)
    BATCH_SIZE      = hyperparameters['BATCH_SIZE'][dataset_name]
    EPOCH           = hyperparameters['EPOCH'][dataset_name]
    LR              = hyperparameters['LR']
    K_FOLDS         = hyperparameters['K_FOLDS']
    
    return BATCH_SIZE, EPOCH, LR, K_FOLDS

def load_all_data(classifer_id, classifier_name, dataset_name, cur_sep_flag, CUR_DIR):
    
    X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
        label2act, act2label, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
        MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM, cal_attitude_angle,\
                STFT_intervals = load_raw_data(dataset_name, CUR_DIR, cur_sep_flag)
    All_data   = np.concatenate((X_train, X_test), axis = 0)
    All_labels = np.concatenate((y_train, y_test))
    All_users  = np.concatenate((User_ids_train, User_ids_test))
    ALL_SUBJECTS_ID = list(set(TRAIN_SUBJECTS_ID + TEST_SUBJECTS_ID))
    
    return All_data, All_labels, All_users, ALL_SUBJECTS_ID, X_train, label2act, \
           ACT_LABELS, ActID, MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM, \
               cal_attitude_angle, STFT_intervals

def get_sep_flags(classifier_name, dataset_name, sep_flags):
    
    if classifier_name in ['IF_ConvTransformer_torch', \
                           'TSF_torch'] and \
        dataset_name in ['HAPT', 'HHAR', 'MobiAct', 'Opportunity',\
                         'Pamap2', 'DSADS', 'RealWorld']:
        cur_sep_flag = True
        sep_flags.append('Sep')
    else:
        cur_sep_flag = False
        sep_flags.append('Unsep')
    return cur_sep_flag, sep_flags

def load_raw_data(dataset_name, CUR_DIR, separate_gravity_flag):
    
    # Optional Hugging Face data pipeline. This is intentionally placed at the
    # beginning of the legacy data loading entry so every dataset loader benefits
    # without embedding download logic into each dataset-specific module.
    if getattr(args, "auto_download_data", True):
        ensure_dataset_available(
            dataset_name,
            project_root=os.path.dirname(CUR_DIR),
            repo_id=getattr(args, "hf_dataset_repo", "crocodilegogogo/TSF-Datasets"),
            revision=getattr(args, "hf_revision", None),
            cache_dir=getattr(args, "hf_cache_dir", None),
            token=getattr(args, "hf_token", None),
            force=getattr(args, "force_hf_download", False),
        )

    # load raw data
    if dataset_name == 'HAPT':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID,\
        TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM,\
        cal_attitude_angle, STFT_intervals\
            = get_HAPT_dataset_param(CUR_DIR,dataset_name,separate_gravity_flag)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_HAPT_raw_data(DATA_DIR, TRAIN_SUBJECTS_ID, ActID,
                                                      WINDOW_SIZE, OVERLAP,separate_gravity_flag,
                                                      cal_attitude_angle)
    
    elif dataset_name == 'Motion_Sense':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
        TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, TRIAL_CODES,\
        WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM, to_NED_flag,\
        cal_attitude_angle, STFT_intervals\
                = get_Motion_Sense_dataset_param(CUR_DIR,dataset_name)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_Motion_Sense_raw_data(DATA_DIR, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,
                                                              ACT_LABELS, TRIAL_CODES, WINDOW_SIZE, OVERLAP, 
                                                              to_NED_flag, cal_attitude_angle)
    
    elif dataset_name == 'SHL_2018':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
        TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, SPLIT_NUM,\
        DATASET_SIZE, STD_ALL, STD, INPUT_CHANNEL, POS_NUM,\
        save_npy_flag, to_NED_flag, cal_attitude_angle, STFT_intervals\
                = get_SHL2018_dataset_param(CUR_DIR,dataset_name)
        
        X_train, X_test, y_train, y_test, label2act, act2label, User_ids_train, User_ids_test\
            = load_SHL2018_raw_data(DATA_DIR, cal_attitude_angle, STD_ALL, STD, 
                                    ACT_LABELS, ActID, SPLIT_NUM, DATASET_SIZE, 
                                    save_npy_flag, to_NED_flag)
    
    elif dataset_name   == 'HHAR':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, SUBJECTS, TRAIN_SUBJECTS_ID,\
        TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM,\
        cal_attitude_angle, STFT_intervals\
            = get_HHAR_dataset_param(CUR_DIR,dataset_name,separate_gravity_flag)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_HHAR_raw_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID, ACT_LABELS,
                                                      WINDOW_SIZE, OVERLAP, separate_gravity_flag,
                                                      cal_attitude_angle)
    
    elif dataset_name == 'MobiAct':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, SUBJECTS, TRAIN_SUBJECTS_ID,\
        TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM,\
        to_NED_flag, cal_attitude_angle, STFT_intervals\
            = get_MobiAct_dataset_param(CUR_DIR,dataset_name,separate_gravity_flag)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_MobiAct_raw_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID, ACT_LABELS,
                                                         WINDOW_SIZE, OVERLAP, separate_gravity_flag,
                                                         cal_attitude_angle, to_NED_flag)
    
    elif dataset_name == 'Opportunity':
        DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS, TRIALS, SELEC_LABEL,\
        ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID, WINDOW_SIZE, OVERLAP,\
        INPUT_CHANNEL, POS_NUM, to_NED_flag,\
        cal_attitude_angle, STFT_intervals\
                   = get_Opportunity_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_Opportunity_data(DATA_DIR, SUBJECTS, TRIALS, SELEC_LABEL,
                                                         TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID,
                                                         ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP,
                                                         separate_gravity_flag, cal_attitude_angle,
                                                         to_NED_flag)
        
        TEST_SUBJECTS_ID         = list(set(SUBJECTS) ^ set(TRAIN_SUBJECTS_ID))
    
    elif dataset_name == 'Pamap2':
        DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
        TRAIN_SUBJECTS_ID, ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP,\
        INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals\
               = get_Pamap2_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_Pamap2_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID,
                                                    ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP, 
                                                    separate_gravity_flag, cal_attitude_angle)
        
        TEST_SUBJECTS_ID         = list(set(SUBJECTS) ^ set(TRAIN_SUBJECTS_ID))
    
    elif dataset_name == 'RealWorld':
        DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
           TRAIN_SUBJECTS_ID, ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle,\
               STFT_intervals = get_RealWorld_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_RealWorld_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID,
                                                    ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP, 
                                                    separate_gravity_flag, cal_attitude_angle)
        
        TEST_SUBJECTS_ID         = list(set(SUBJECTS) ^ set(TRAIN_SUBJECTS_ID))
    
    elif dataset_name == 'DSADS':
        DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
        TRAIN_SUBJECTS_ID, ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP,\
        INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals\
                = get_DSADS_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label  = load_DSADS_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID, ACT_LABELS,
                                                    ActID, separate_gravity_flag, cal_attitude_angle)
        
        TEST_SUBJECTS_ID         = list(set(SUBJECTS) ^ set(TRAIN_SUBJECTS_ID))
    
    elif dataset_name == 'SHO':
        DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
           TRAIN_SUBJECTS_ID, ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals = get_SHO_dataset_param(CUR_DIR, dataset_name)
        
        X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
            label2act, act2label = load_SHO_data(DATA_DIR, SUBJECTS, TRAIN_SUBJECTS_ID,
                                                 ACT_LABELS, ActID, WINDOW_SIZE, OVERLAP, 
                                                 cal_attitude_angle)
        
        TEST_SUBJECTS_ID         = list(set(SUBJECTS) ^ set(TRAIN_SUBJECTS_ID))
    
    return X_train, X_test, y_train, y_test, User_ids_train, User_ids_test,\
           label2act, act2label,\
           ACT_LABELS, ActID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,\
           MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals

def get_K_folds_subjects(ALL_SUBJECTS_ID, K_FOLDS):
    
    K_folds_subjects = partition(ALL_SUBJECTS_ID, K_FOLDS)
    K_folds_subjects = [[22, 37, 43, 45, 39, 20, 25],
                        [55, 48, 49, 41, 59, 64],
                        [42, 2, 29, 35, 66, 11],
                        [50, 6, 19, 10, 38, 65],
                        [63, 28, 67, 16, 44, 26],
                        [53, 54, 32, 47, 3, 33],
                        [9, 27, 1, 7, 34, 21],
                        [52, 61, 4, 12, 36, 18],
                        [40, 57, 51, 56, 60, 58],
                        [5, 24, 8, 23, 46, 62]]
    
    return K_folds_subjects

def check_class_balance(logger, y_train: np.ndarray, y_test: np.ndarray,
                        label2act: Dict[int, str], n_class: int = 12
) -> None:
    c_train = Counter(y_train)
    c_test = Counter(y_test)

    for c, mode in zip([c_train, c_test], ["train", "test"]):
        logger.debug(f"{mode} labels")
        len_y = sum(c.values())
        for label_id in range(n_class):
            logger.debug(
                f"{label2act[label_id]} ({label_id}): {c[label_id]} samples ({c[label_id] / len_y * 100:.04} %)"
            )

def data_preprocessing(ALL_SUBJECTS_ID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, All_users, All_data, \
                       All_labels, X_train, classifier_name, STFT_intervals, POS_NUM, test_split, 
                       COMPLEMAENTARY_ALPHA=0.8):
    
    start1 = time.time()
    
    if len(ALL_SUBJECTS_ID)!=1:
        X_train, y_train = get_loso_train_test_data(TRAIN_SUBJECTS_ID, All_users, All_data, All_labels)
        X_test, y_test   = get_loso_train_test_data(TEST_SUBJECTS_ID, All_users, All_data, All_labels)
    else:
        X_train          = All_data[:X_train.shape[0],:,:,:]
        y_train          = All_labels[:X_train.shape[0]]
        X_test           = All_data[X_train.shape[0]:,:,:,:]
        y_test           = All_labels[X_train.shape[0]:]
    
    if classifier_name in ['TSF_IMU_CF_torch']:
        for (pos_id,pos) in enumerate(range(POS_NUM)):
            if pos_id == 0:
                X_train_cat = complementary_filter(X_train[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], COMPLEMAENTARY_ALPHA)
                X_test_cat  = complementary_filter(X_test[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], COMPLEMAENTARY_ALPHA)
            else:
                X_train_mid = complementary_filter(X_train[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], COMPLEMAENTARY_ALPHA)
                X_test_mid  = complementary_filter(X_test[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], COMPLEMAENTARY_ALPHA)
                X_train_cat = np.concatenate((X_train_cat, X_train_mid), axis = 2)
                X_test_cat  = np.concatenate((X_test_cat, X_test_mid), axis = 2)
        X_train             = X_train_cat
        X_test              = X_test_cat
    
    if classifier_name in ['DeepSense_torch','AttnSense_torch','GlobalFusion_torch']:
        for (pos_id,pos) in enumerate(range(POS_NUM)):
            if pos_id == 0:
                X_train_cat = STFT_transform(X_train[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], y_train, STFT_intervals, test_split)
                X_test_cat  = STFT_transform(X_test[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], y_test, STFT_intervals, test_split)
            else:
                X_train_mid = STFT_transform(X_train[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], y_train, STFT_intervals, test_split)
                X_test_mid  = STFT_transform(X_test[:,:,(X_train.shape[2]//POS_NUM)*pos_id:(X_train.shape[2]//POS_NUM)*(pos_id+1),:], y_test, STFT_intervals, test_split)
                X_train_cat = np.concatenate((X_train_cat, X_train_mid), axis = 1)
                X_test_cat  = np.concatenate((X_test_cat, X_test_mid), axis = 1)
        X_train             = X_train_cat
        X_test              = X_test_cat
    
    end1 = time.time()
    
    return X_train, y_train, X_test, y_test, start1, end1

def fft_transform(x, X_data, test_split):
    
    x_fft = fft(x)
    # X_data.shape[1]:9, the channel num of all sensor data.||x.shape[2]:8, the lenth of each time_step
    x_fft = x_fft.reshape([-1, X_data.shape[1], x.shape[1]])
    if len(x_fft.shape) > 3:
        x_fft = np.swapaxes(x_fft.squeeze(),1,2)
    else:
        x_fft = np.swapaxes(x_fft,1,2)
    # split the data of different sensors
    x_sensor_fft = np.split(x_fft, int(x_fft.shape[2]/3), axis=2)
    x_all_sensor_fft = []
    # flatten the data of each sensor data, then concat them on the rows
    for i in range(len(x_sensor_fft)):
        shape_flag = x_sensor_fft[i].shape
        if i==0:
            fft_flag = x_sensor_fft[i].reshape([shape_flag[0], shape_flag[1]*shape_flag[2]])
            x_all_sensor_fft = fft_flag
        else:
            fft_flag = x_sensor_fft[i].reshape([shape_flag[0], shape_flag[1]*shape_flag[2]])
            x_all_sensor_fft = np.concatenate((x_all_sensor_fft, fft_flag), axis=1)
    # flatten the batch of real and imag values, get ready to merge them
    x_all_sensor_fft_real = x_all_sensor_fft.real.reshape(x_all_sensor_fft.shape[0]*x_all_sensor_fft.shape[1],1)
    x_all_sensor_fft_imag = x_all_sensor_fft.imag.reshape(x_all_sensor_fft.shape[0]*x_all_sensor_fft.shape[1],1)
    # concat the real and imag values on the columns, then reshape to the batch size to merge the real and imag values:
    x_merge_real_imag = np.concatenate((x_all_sensor_fft_real, x_all_sensor_fft_imag), axis=1)
    # a real value following a imag value
    x_merge_real_imag = x_merge_real_imag.reshape([x_all_sensor_fft.shape[0],-1])
    
    return x_merge_real_imag

def STFT_transform(X_data, y_data, STFT_intervals, test_split):
    
    STFT_result = []
    
    X_data = X_data.squeeze()
    torch_dataset = Data.TensorDataset(torch.FloatTensor(X_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = X_data.shape[0] // test_split,
                                  shuffle = False)
    for step, (x,y) in enumerate(data_loader):
        
        x_split_merge_real_imag = []
        
        with torch.no_grad():
            x = x.cpu().data.numpy()
            if len(x.shape) == 2:
                x = np.expand_dims(x, axis=0)
            x = x.reshape([x.shape[0]*x.shape[1], x.shape[2]])
            if x.shape[1]%STFT_intervals == 0:
                x_split = np.split(x, STFT_intervals, axis=1)
            else:
                print('Please input a STFT_intervals value can be divided evently by data lenth:'+str(x.shape[1]))
            # Do fft transform for every time_step, then concat all the time_steps
            for time_step in range(len(x_split)):
                
                if time_step == 0:
                    x_split_merge_real_imag = fft_transform(x_split[time_step], X_data, test_split)
                    x_split_merge_real_imag = np.expand_dims(x_split_merge_real_imag, axis=1)
                else:
                    merge_cur_time_step = fft_transform(x_split[time_step], X_data, test_split)
                    merge_cur_time_step = np.expand_dims(merge_cur_time_step, axis=1)
                    x_split_merge_real_imag = np.concatenate((x_split_merge_real_imag, merge_cur_time_step), axis=1)
    
        if step == 0:
            STFT_result = x_split_merge_real_imag
        else:
            STFT_result = np.concatenate((STFT_result, x_split_merge_real_imag), axis=0)
    
    return np.expand_dims(STFT_result, axis=1)

def complementary_filter(X, alpha):
    
    X       = X.squeeze()
    x_grav  = X[:,0:3,:]
    x_gyro  = X[:,3:6,:]
    x_acc   = X[:,6:9,:]
    
    attitude = np.expand_dims(x_grav[:,:,0], axis=2)
    for i in range(1,x_gyro.shape[2]):
        # angleAcc = math.degrees(math.atan(-acc[i,0]/math.sqrt(acc[i,1]**2+acc[i,2]**2)))
        new_attitude = (attitude[:,:,i-1] + x_gyro[:,:,i])*alpha + x_grav[:,:,i]*(1-alpha)
        new_attitude = np.expand_dims(new_attitude, axis=2)
        attitude     = np.concatenate((attitude, new_attitude), axis=2)
    
    X = np.concatenate((attitude, x_acc), axis=1)
    X = np.expand_dims(X, axis=1)
    
    return X

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

def create_cuda_classifier(dataset_name, classifier_name, INPUT_CHANNEL, POS_NUM, data_length,
                           train_size, val_size, test_size, nb_classes, STFT_intervals,
                           BATCH_SIZE, INFERENCE_DEVICE, test_split):
    
    classifier, classifier_func = create_classifier(dataset_name, classifier_name, INPUT_CHANNEL, POS_NUM,
                                                    data_length, train_size, val_size, test_size,
                                                    nb_classes, STFT_intervals, BATCH_SIZE, INFERENCE_DEVICE, test_split)
    if INFERENCE_DEVICE == 'TEST_CUDA':
        classifier.to(get_runtime_device())
    print(classifier)
    classifier_parameter = get_parameter_number(classifier)
    
    return classifier, classifier_func, classifier_parameter

def initialize_saving_variables(X_train, X_test, nb_classes):
    
    # for test sets there are predictions for SUBJECT_NUM times
    models = []
    scores: Dict[str, Dict[str, List[Any]]] = {
        "logloss": {"train": [], "valid": [], "test": []},
        "accuracy": {"train": [], "valid": [], "test": []},
        "macro-precision": {"train": [], "valid": [], "test": []},
        "macro-recall": {"train": [], "valid": [], "test": []},
        "macro-f1": {"train": [], "valid": [], "test": []},
        "weighted-f1": {"train": [], "valid": [], "test": []},
        "micro-f1": {"train": [], "valid": [], "test": []},
        "per_class_f1": {"train": [], "valid": [], "test": []},
        "confusion_matrix": {"train": [], "valid": [], "test": []},
    }    
    log_training_duration = []
    
    return models, scores, log_training_duration

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Network_Total_Parameters:', total_num, 'Network_Trainable_Parameters:', trainable_num)
    return {'Total': total_num, 'Trainable': trainable_num}

def training_process(logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                     y_test, nb_classes, classifier_parameter, classifier,
                     classifier_func, MODEL_DIR, PATTERN, EPOCH, BATCH_SIZE, LR,
                     log_training_duration, test_split):
    
    # log train, validation and test dataset info, log the network
    log_redivdataset_network_info(logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                  y_test, nb_classes, classifier_parameter, classifier)
    
    # train the network and save the best validation model
    output_directory_models = os.path.join(MODEL_DIR, 'SUBJECT_'+str(subject_id))
    flag_output_directory_models = create_directory(output_directory_models)
    if PATTERN == 'TRAIN':
        # flag_output_directory_models = PATTERN
        # for each subject, train the network and save the best validation model
        print('SUBJECT_'+str(subject_id)+': start to train')
        history, per_training_duration, log_training_duration = classifier_func.train_op(classifier, EPOCH, BATCH_SIZE, LR,
                                                                                         X_tr, Y_tr, X_val, Y_val, X_test, y_test,
                                                                                         output_directory_models, log_training_duration,
                                                                                         test_split)
    else:
        if not os.path.exists(os.path.join(output_directory_models, 'score.csv')):
            # for each subject, train the network and save the best validation model
            print('SUBJECT_'+str(subject_id)+': start to train')
            history, per_training_duration, log_training_duration = classifier_func.train_op(classifier, EPOCH, BATCH_SIZE, LR,
                                                                                             X_tr, Y_tr, X_val, Y_val, X_test, y_test,
                                                                                             output_directory_models, log_training_duration,
                                                                                             test_split)
        else:
            print('Already_done: '+'SUBJECT_'+str(subject_id))
            # read the training duration of current subject when available. HF-hosted
            # pretrained weights may contain only checkpoint files, so fall back to 0.0.
            score_path = os.path.join(output_directory_models, 'score.csv')
            if os.path.exists(score_path):
                per_training_duration = pd.read_csv(score_path, skiprows=1, nrows=1, header = None)[1][0]
            else:
                per_training_duration = 0.0
            log_training_duration.append(per_training_duration)
        
    return per_training_duration, log_training_duration, output_directory_models

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence  = 1. - smoothing
        logprobs    = F.log_softmax(x, dim=-1)
        nll_loss    = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss    = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss        = confidence * nll_loss + smoothing * smooth_loss
        # return loss.mean()
        return loss.sum()

def get_loso_train_test_data(SUBJECTS_IDS, All_users, All_data, All_labels):
    for (r_id, sub_id) in enumerate(SUBJECTS_IDS):
        if r_id == 0:
            ids = np.where(All_users == sub_id)[0]
        else:
            ids = np.concatenate((ids, np.where(All_users == sub_id)[0]))
    X = All_data[ids,:,:,:]
    y = All_labels[ids]
    return X, y

def predict_tr_val_test(dataset_name, network, nb_classes, LABELS,
                        train_x, val_x, test_x,
                        train_y, val_y, test_y,
                        scores, per_training_duration,
                        run_id, output_directory_models,
                        test_split):
    
    start = time.time()
    
    # generate network objects
    network_obj = network
    # load best saved validation models
    best_validation_model = os.path.join(output_directory_models, 'best_validation_model.pkl')
    network_obj.load_state_dict(torch.load(best_validation_model, map_location=get_runtime_device()))
    network_obj.eval()
    # get outputs of best saved validation models by concat them, input: train_x, val_x, test_x
    pred_train = np.array(model_predict(network_obj, train_x, train_y, test_split)[0])
    pred_valid = np.array(model_predict(network_obj, val_x, val_y, test_split)[0])
    pred_test = np.array(model_predict(network_obj, test_x, test_y, test_split)[0])
    
    end = time.time()
    
    # record the metrics of each subject, initialize the score per CV
    score: Dict[str, Dict[str, List[Any]]] = {
                "logloss": {"train": [], "valid": [], "test": []},
                "accuracy": {"train": [], "valid": [], "test": []},
                "macro-precision": {"train": [], "valid": [], "test": []},
                "macro-recall": {"train": [], "valid": [], "test": []},
                "macro-f1": {"train": [], "valid": [], "test": []},
                "weighted-f1": {"train": [], "valid": [], "test": []},
                "micro-f1": {"train": [], "valid": [], "test": []},
                "per_class_f1": {"train": [], "valid": [], "test": []},
                "confusion_matrix": {"train": [], "valid": [], "test": []},
                }
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    for pred, X, y, mode in zip(
        [pred_train, pred_valid, pred_test], [train_x, val_x, test_x], [train_y, val_y, test_y], ["train", "valid", "test"]
    ):
        loss, acc, weighted_f1 = get_test_loss_acc(network_obj, loss_function, X, y, test_split, test_flag=True)
        pred = pred.argmax(axis=1)
        # y is already the argmaxed category
        scores["logloss"][mode].append(loss)
        scores["accuracy"][mode].append(acc)
        scores["macro-precision"][mode].append(precision_score(y, pred, average="macro"))
        scores["macro-recall"][mode].append(recall_score(y, pred, average="macro"))
        scores["macro-f1"][mode].append(f1_score(y, pred, average="macro"))
        scores["weighted-f1"][mode].append(f1_score(y, pred, average="weighted"))
        scores["micro-f1"][mode].append(f1_score(y, pred, average="micro"))
        scores["per_class_f1"][mode].append(f1_score(y, pred, average=None))
        scores["confusion_matrix"][mode].append(confusion_matrix(y, pred, normalize=None))
        
        # record the metrics of each subject
        score["logloss"][mode].append(loss)
        score["accuracy"][mode].append(acc)
        score["macro-precision"][mode].append(precision_score(y, pred, average="macro"))
        score["macro-recall"][mode].append(recall_score(y, pred, average="macro"))
        score["macro-f1"][mode].append(f1_score(y, pred, average="macro"))
        score["weighted-f1"][mode].append(f1_score(y, pred, average="weighted"))
        score["micro-f1"][mode].append(f1_score(y, pred, average="micro"))
        score["per_class_f1"][mode].append(f1_score(y, pred, average=None))
        score["confusion_matrix"][mode].append(confusion_matrix(y, pred, normalize=None))
    
    if dataset_name != 'MobiAct':
        save_metrics_per_cv(score, per_training_duration,
                            run_id, nb_classes, LABELS,
                            test_y, pred_test,
                            output_directory_models)
    else:
        save_metrics_per_fold(score, per_training_duration,
                              run_id, nb_classes, LABELS,
                              test_y, pred_test, output_directory_models)
    
    return pred_train, pred_valid, pred_test, scores, (end-start)

def get_test_loss_acc(net, loss_function, x_data, y_data, test_split=1, test_flag=False):
    loss_sum_data = torch.tensor(0, device=get_runtime_device())
    true_sum_data = torch.tensor(0, device=get_runtime_device())
    batch_size_split = x_data.shape[0] // test_split
    if batch_size_split == 0:
        batch_size_split = 1
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = batch_size_split,
                                  shuffle = False)
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                x = x.to(get_runtime_device())
                y = y.to(get_runtime_device())
            output_bc = net(x, test_flag)[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            
            out = output_bc
            
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            else:
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            loss_bc = loss_function(output_bc, y)
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_bc
            true_sum_data = true_sum_data + true_num_bc
            
            if step == 0:
                output = out
            else:
                output = torch.cat((output, out), axis=0)
    
    loss = loss_sum_data.data.item()/y_data.shape[0]
    acc = true_sum_data.data.item()/y_data.shape[0]
    
    output      = output.cpu().data.numpy()
    output      = np.array(output).argmax(axis=1)
    weighted_f1 = f1_score(y_data, output, average="weighted")
    
    return loss, acc, weighted_f1

def get_test_loss_acc_dynamic(net, loss_function, x_data, y_data, test_split=1, test_flag=False):
    loss_sum_data = torch.tensor(0, device=get_runtime_device())
    true_sum_data = torch.tensor(0, device=get_runtime_device())
    output = []
    batch_size_split = x_data.shape[0] // test_split
    if batch_size_split == 0:
        batch_size_split = 1
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = batch_size_split,
                                  shuffle = False)
    
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                x = x.to(get_runtime_device())
                y = y.to(get_runtime_device())
            output_bc = net(x, test_flag)[0]
            
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            
            out = output_bc
            
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            else:
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            loss_bc = loss_function(output_bc, y)
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_bc
            true_sum_data = true_sum_data + true_num_bc
            
            if step == 0:
                output = out
            else:
                output = torch.cat((output, out), axis=0)
    
    loss = loss_sum_data.data.item()/y_data.shape[0]
    acc = true_sum_data.data.item()/y_data.shape[0]
    
    output = output.cpu().data.numpy()
    output = np.array(output).argmax(axis=1)
    macro_f1 = f1_score(y_data, output, average="macro")
    
    return loss, acc, macro_f1

def get_test_loss_acc_graph(net, loss_function, x_data, y_data, test_split=1, test_flag=False):
    loss_sum_data = torch.tensor(0, device=get_runtime_device())
    true_sum_data = torch.tensor(0, device=get_runtime_device())
#    output = []
    batch_size_split = x_data.shape[0] // test_split
    if batch_size_split == 0:
        batch_size_split = 1
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader   = Data.DataLoader(dataset = torch_dataset,
                                    batch_size = batch_size_split,
                                    shuffle = False)
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            
            # calculate the output of networks
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                x = x.to(get_runtime_device())
                y = y.to(get_runtime_device())
            output_bc, attns = net(x, test_flag)
            g_matrix_bc      = attns[1]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            out = output_bc
            g_matrix = g_matrix_bc
            if step == 0:
                output = out
                g_matrix_out = g_matrix
            else:
                output = torch.cat((output, out), axis=0)
                g_matrix_out = torch.cat((g_matrix_out, g_matrix), axis=0)
            
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            else:
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            loss_bc = loss_function(output_bc, y)
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_bc
            true_sum_data = true_sum_data + true_num_bc
    
    loss = loss_sum_data.data.item()/y_data.shape[0]
    acc = true_sum_data.data.item()/y_data.shape[0]
    
    output      = output.cpu().data.numpy()
    output      = np.array(output).argmax(axis=1)
    weighted_f1 = f1_score(y_data, output, average="weighted")
    
    return loss, acc, weighted_f1, g_matrix_out

def get_test_loss_acc_dwt(net, loss_function, x_data, y_data, test_split=1, test_flag=False):
    loss_sum_data = torch.tensor(0, device=get_runtime_device())
    true_sum_data = torch.tensor(0, device=get_runtime_device())
    batch_size_split = x_data.shape[0] // test_split
    if batch_size_split == 0:
        batch_size_split = 1
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader   = Data.DataLoader(dataset = torch_dataset,
                                    batch_size = batch_size_split,
                                    shuffle = False)
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            
            # calculate the output of networks
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                x = x.to(get_runtime_device())
                y = y.to(get_runtime_device())
            output_bc, attns = net(x, test_flag)
            DWT_matrix_bc    = attns[2]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            out = output_bc
            DWT_matrix = DWT_matrix_bc
            if step == 0:
                output = out
                DWT_matrix_out = DWT_matrix
            else:
                output = torch.cat((output, out), axis=0)
                DWT_matrix_out = torch.cat((DWT_matrix_out, DWT_matrix), axis=0)
            
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            else:
                pred_bc = torch.max(output_bc, 1)[1].data.squeeze()
            loss_bc = loss_function(output_bc, y)
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_bc
            true_sum_data = true_sum_data + true_num_bc
    
    loss = loss_sum_data.data.item()/y_data.shape[0]
    acc = true_sum_data.data.item()/y_data.shape[0]
    
    output      = output.cpu().data.numpy()
    output      = np.array(output).argmax(axis=1)
    weighted_f1 = f1_score(y_data, output, average="weighted")
    
    return loss, acc, weighted_f1, DWT_matrix_out

def model_predict(net, x_data, y_data, test_split=1, test_flag=True):
    predict = [] 
    output  = []
    IMU_attentions = []
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    batch_size_split = x_data.shape[0] // test_split
    if batch_size_split == 0:
        batch_size_split = 1
    data_loader   = Data.DataLoader(dataset = torch_dataset,
                                  batch_size = batch_size_split,
                                  shuffle = False)
    for step, (x,y) in enumerate(data_loader):
        with torch.no_grad():
            if args.INFERENCE_DEVICE == 'TEST_CUDA':
                x = x.to(get_runtime_device())
            output_bc, attn_bc = net(x, test_flag)
            if type(attn_bc) == list:
                attn_bc = attn_bc[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
                attn_bc.unsqueeze_(dim=0)
            if step == 0:
                output, IMU_attentions = output_bc, attn_bc
            else:
                output = torch.cat((output, output_bc), axis=0)
                IMU_attentions = torch.cat((IMU_attentions, attn_bc), axis=0)
    
    return output.cpu().data.numpy(), IMU_attentions.cpu().data.numpy()

def save_models(net, output_directory_models, 
                loss_train, loss_train_results, 
                accuracy_validation, accuracy_validation_results, 
                start_time, training_duration_logs):   
    
    output_directory_best_val = os.path.join(output_directory_models, 'best_validation_model.pkl')
    if accuracy_validation >= max(accuracy_validation_results):
        torch.save(net.state_dict(), output_directory_best_val)

def logging_settings(classifier_name, CUR_DIR, dataset_name):
    
    # Logging settings
    EXEC_TIME = classifier_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join(CUR_DIR, f"logs", dataset_name, classifier_name, f"{EXEC_TIME}")
    MODEL_DIR = os.path.join(CUR_DIR, f"saved_models", dataset_name, classifier_name)
    create_directory(LOG_DIR) # Create log directory
    
    # create log object with classifier_name
    logger = getLogger(classifier_name)
    # set recording format
    formatter = Formatter("%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s")
    # create FileHandler with current LOG_DIR and format
    fileHandler = FileHandler(f"{LOG_DIR}/{EXEC_TIME}.log")
    fileHandler.setFormatter(formatter)

    mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
    mpl_logger.setLevel(WARNING)
    
    logger.setLevel(DEBUG)
    logger.addHandler(fileHandler)
    
    # important! get current logger with its name (the name is set with classifier_name)
    logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")
    
    return EXEC_TIME, LOG_DIR, MODEL_DIR, logger, fileHandler

def log_dataset_network_info(logger, ALL_SUBJECTS_ID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, 
                             X_train, X_test, y_train, y_test, cal_attitude_angle, 
                             ACT_LABELS, ActID, label2act, BATCH_SIZE, EPOCH, LR):
    
    # log the information of datasets
    log_dataset_info(logger, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, X_train, X_test,
                     y_train, y_test, cal_attitude_angle, ACT_LABELS, ActID)
    
    # check the category imbalance
    nb_classes = len(ACT_LABELS)
    check_class_balance(logger, y_train.flatten(), y_test.flatten(),
                        label2act=label2act, n_class=nb_classes)
    
    # log the hyper-parameters
    log_HyperParameters(logger, BATCH_SIZE, EPOCH, LR, len(ALL_SUBJECTS_ID))
    
    return nb_classes

def log_dataset_info(logger, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, X_train, X_test,
                     y_train, y_test, cal_attitude_angle, ACT_LABELS, ActID):
    logger.debug("---Dataset and preprocessing information---")
    logger.debug(f"TRAIN_SUBJECTS_ID = {TRAIN_SUBJECTS_ID}")
    logger.debug(f"TEST_SUBJECTS_ID = {TEST_SUBJECTS_ID}")
    logger.debug(f"X_train_shape = {X_train.shape}, X_test_shape={X_test.shape}")
    logger.debug(f"Y_train_shape = {y_train.shape}, Y_test.shape={y_test.shape}")
    logger.debug(f"Cal_Attitude_Angle = {cal_attitude_angle}")
    logger.debug(f"ACT_LABELS = {ACT_LABELS}")
    logger.debug(f"ActID = {ActID}")

def log_HyperParameters(logger, BATCH_SIZE, EPOCH, LR, subject_num):
    logger.debug("---HyperParameters---")
    logger.debug(f"BATCH_SIZE : {BATCH_SIZE}, EPOCH : {EPOCH}, LR : {LR}, SUBJECT_NUM : {subject_num}")

def log_redivdataset_network_info(logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                  y_test, nb_classes, classifier_parameter, classifier):
    if subject_id == 1:
        logger.debug("---Redivided dataset and network information---")
        logger.debug(f"X_train_shape={X_tr.shape}, X_validation_shape={X_val.shape}, X_test_shape={X_test.shape}")
        logger.debug(f"Y_train_shape={Y_tr.shape}, Y_validation_shape={Y_val.shape}, y_test_shape={y_test.shape}")
        logger.debug(f"num of categories = {nb_classes}")
        logger.debug(f"num of network parameter = {classifier_parameter}")
        logger.debug(f"the architecture of the network = {classifier}")

def log_final_test_results(logger, log_training_duration, scores, label2act, nb_classes, ALL_SUBJECTS_ID,
                           start1, end1, X_tr, y_train, y_test, time2, dataset_name, classifier_name, MODELS_COMP_LOG_DIR,
                           CLASSIFIERS, classifier, classifier_parameter, fileHandler):
    
    # Log the Test Scores of every subject
    log_every_SUBJECT_score(logger, log_training_duration, scores, label2act, nb_classes, len(ALL_SUBJECTS_ID))
    
    # Log the averaged Score of different subjects
    log_averaged_SUBJECT_scores(logger, log_training_duration, scores, label2act, nb_classes, len(ALL_SUBJECTS_ID))
    
    # Log the inference time including data loading, preprocessing and model inference
    preprocess_time, _ = log_inference_time(start1, end1, y_train, y_test, time2,
                                            dataset_name, classifier_name, len(ALL_SUBJECTS_ID), logger)
    
    # Log the flops and param_num of networks
    flops, inference_time = log_flops(logger, classifier, X_tr)
    
    save_classifiers_comparison(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name,
                                scores, ALL_SUBJECTS_ID, preprocess_time, inference_time,
                                len(ALL_SUBJECTS_ID), flops, classifier_parameter['Trainable'])
    
    # remove current logger
    remove_log(logger, fileHandler)
    
    return 

def log_every_SUBJECT_score(logger, log_training_duration, scores, label2act, nb_classes, SUBJECT_NUM):
    for i in range(SUBJECT_NUM):
        # Log Every Subject Scores
        logger.debug("---Per Subject Scores, Subject"+str(i)+"---")
        
        # log per SUBJECT training time
        logger.debug(f"Training Duration = {log_training_duration[i]}s")
        
        for mode in ["train", "valid", "test"]:
            # log the average of "logloss", "accuracy", "precision", "recall", "f1"
            logger.debug(f"---{mode}---")
            logger.debug(f"logloss={round(scores['logloss'][mode][i])}, accuracy={round(scores['accuracy'][mode][i])},\
                         macro-precision={round(scores['macro-precision'][mode][i])}, macro-recall={round(scores['macro-recall'][mode][i])},\
                         macro-f1={round(scores['macro-f1'][mode][i])}, weighted-f1={round(scores['weighted-f1'][mode][i])},\
                          micro-f1={round(scores['micro-f1'][mode][i])}")

def log_averaged_SUBJECT_scores(logger, log_training_duration, scores, label2act, nb_classes, SUBJECT_NUM):
    # Log Averaged Score of all Subjects
    logger.debug("---Subject Averaged Scores---")    
    # log the average of training time
    logger.debug(f"Averaged Training Duration = {(np.mean(log_training_duration))}s")
    
    for mode in ["train", "valid", "test"]:
        
        # log the average of "logloss", "accuracy", "precision", "recall", "f1"
        logger.debug(f"---{mode}---")
        for metric in ["logloss", "accuracy", "macro-precision", "macro-recall", "macro-f1", "weighted-f1", "micro-f1"]:
            logger.debug(f"{metric}={round(np.mean(scores[metric][mode]))} +- {round(np.std(scores[metric][mode]))}")

def log_inference_time(start1, end1, y_train, y_test, time2,
                       dataset_name, classifier_name, SUBJECT_NUM, logger):
    time1 = (end1 - start1)/(len(y_train)+len(y_test))
    time2 = time2/((len(y_train)+len(y_test))*SUBJECT_NUM)
    # time2 = time2/SUBJECT_NUM
    logger.debug("---Final Inference Time Per Sample-Averaged over Folds---")
    logger.debug(f"preprocessing_time={time1}")
    logger.debug(f"pure_inference_time={time2}")
    logger.debug(f"single_model_prepro_inference_time={time1+time2}")
    logger.debug(f"all_model_prepro_inference_time={time1+time2*SUBJECT_NUM}")
    
    return time1, time2

def log_flops(logger, network, train_x):
    # flops
    inputs_shape = torch.FloatTensor(train_x)[0].unsqueeze(0).shape
    inputs = torch.randn(inputs_shape)
    if args.INFERENCE_DEVICE == 'TEST_CUDA':
        inputs = inputs.to(get_runtime_device())
    flops, paras = profile(network, inputs = (inputs,))
    flops, paras = clever_format([flops, paras], "%.3f")
    print(flops, paras)
    # flops
    logger.debug(f"flops={flops}")
    
    start_time = time.time()
    # inference_time
    for i in range(10000):
        _, _ = network(inputs)
    end_time = time.time()
    inference_time = (end_time - start_time)/10000
    
    return flops, inference_time
    
def remove_log(logger, fileHandler):
    logger.removeHandler(fileHandler)

def plot_confusion_matrix(
    cms: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    path: str = "confusion_matrix.png",
) -> None:
    """Plot confusion matrix"""
    # Cal the ensembled confusion_matrix by averaging them
    cms = [np.mean(cms[mode], axis=0) for mode in ["train", "valid", "test"]]

    fig, ax = plt.subplots(ncols=3, figsize=(20, 7))
    for i, (cm, mode) in enumerate(zip(cms, ["train", "valid", "test"])):
        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            square=True,
            vmin=0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax[i],
        )
        ax[i].set_xlabel("Predicted label")
        ax[i].set_ylabel("True label")
        ax[i].set_title(f"Averaged confusion matrix - {mode}")

    plt.tight_layout()
    fig.savefig(path)
    plt.close()

def log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                loss_validation_results, accuracy_validation_results, loss_test_results, accuracy_test_results,
                output_directory_models):
    
    history = pd.DataFrame(data = np.zeros((EPOCH,7),dtype=np.float), 
                           columns=['train_acc','train_loss','val_acc','val_loss',
                                    'test_acc','test_loss','lr'])
    history['train_acc'] = accuracy_train_results
    history['train_loss'] = loss_train_results
    history['val_acc'] = accuracy_validation_results
    history['val_loss'] = loss_validation_results
    history['test_acc'] = accuracy_test_results
    history['test_loss'] = loss_test_results
    history['lr'] = lr_results
    
    # load saved models, predict, cal metrics and save logs
    history.to_csv(os.path.join(output_directory_models, 'history.csv'), index=False)
    
    return history

def plot_learning_history(EPOCH, history, path):
    """Plot learning curve
    Args:
        fit (Any): History object
        path (str, default="history.png")
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    axL.plot(history["train_loss"], label="train")
    axL.plot(history["val_loss"], label="validation")
    axL.plot(history["test_loss"], label="test")
    axL.set_title("Loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")

    axR.plot(history["train_acc"], label="train")
    axR.plot(history["val_acc"], label="validation")
    axR.plot(history["test_acc"], label="test")
    axR.set_title("Accuracy")
    axR.set_xlabel("epoch")
    axR.set_ylabel("accuracy")
    axR.legend(loc="upper right")

    fig.savefig(os.path.join(path, 'history.png'))
    plt.close()
    
def save_metrics_per_cv(score, per_training_duration,
                        subject_id, nb_classes, LABELS,
                        y_true, y_pred,
                        output_directory_models):
    
    # save training time
    per_training_duration_pd = pd.DataFrame(data = per_training_duration,
                                            index = ["training duration"],
                                            columns = ["SUBJECT_"+str(subject_id)])
    per_training_duration_pd.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True)
    
    # save "logloss", "accuracy", "precision", "recall", "f1"
    score_pd = pd.DataFrame(data = np.zeros((7,3),dtype=np.float),
                            index=["logloss", "accuracy", "macro-precision", "macro-recall",
                                   "macro-f1", "weighted-f1", "micro-f1"], 
                            columns=["train", "valid", "test"])
    for row in score_pd.index:
        for column in score_pd.columns:
            score_pd.loc[row, column] = score[row][column][0]
    score_pd.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True, mode='a+')
    
    # save "per_class_f1"
    pd.DataFrame(["per_class_f1"]).to_csv(os.path.join(output_directory_models, 'score.csv'), index=False, header=False, mode='a+')
    per_class_f1_pd = pd.DataFrame(data = np.zeros((3, nb_classes),dtype=np.str_),
                            index=["train", "valid", "test"], columns=LABELS)
    for row in per_class_f1_pd.index:
        flag = 0
        if row == "test":
            for (i, column) in enumerate(per_class_f1_pd.columns):
                if i in list(np.unique(y_true)):
                    per_class_f1_pd.loc[row, column] = np.str_(score["per_class_f1"][row][0][flag])
                    flag = flag + 1
                else:
                    per_class_f1_pd.loc[row, column] = 'missed_category'
        else:
            for (i, column) in enumerate(per_class_f1_pd.columns):
                per_class_f1_pd.loc[row, column] = score["per_class_f1"][row][0][i]
            
    per_class_f1_pd.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True, mode='a+')
    
    # save confusion_matrix
    for key in score['confusion_matrix'].keys():
        pd.DataFrame(["confusion_matrix_"+key]).to_csv(os.path.join(output_directory_models, 'score.csv'), index=False, header=False, mode='a+')
        each_confusion_matrix = pd.DataFrame(data = np.zeros((nb_classes, nb_classes),dtype=np.float), 
                                             index = LABELS, columns=LABELS)
        # if missing categories exist
        if key == 'test':
            # two loops, one for row and one for column
            flag_cfm_row = 0
            # row loop
            for (i,row) in enumerate(each_confusion_matrix.index):
                if i in list(np.unique(y_true)):
                    flag_cfm_col = 0
                    # column loop
                    for (j,column) in enumerate(each_confusion_matrix.columns):
                        if j in list(np.unique(y_true)):
                            each_confusion_matrix.loc[row, column] = score['confusion_matrix'][key][0][flag_cfm_row][flag_cfm_col]
                            flag_cfm_col = flag_cfm_col + 1
                        else:
                            each_confusion_matrix.loc[row, column] = 'missed_category'
                    flag_cfm_row = flag_cfm_row + 1
                else:
                    for (j,column) in enumerate(each_confusion_matrix.columns):
                        each_confusion_matrix.loc[row, column] = 'missed_category'
        else:
            for (i,row) in enumerate(each_confusion_matrix.index):
                for (j,column) in enumerate(each_confusion_matrix.columns):
                    each_confusion_matrix.loc[row, column] = score['confusion_matrix'][key][0][i][j]
        each_confusion_matrix.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True, mode='a+')
    
    # save the indexes of the false predictions
    y_pred = y_pred.argmax(axis=1)
    false_index = np.where(np.array(y_true)!=np.array(y_pred))[0].tolist()
    y_correct = np.array(y_true)[np.array(y_true)!=np.array(y_pred)].tolist()
    pre_false = np.array(y_pred)[np.array(y_true)!=np.array(y_pred)].tolist()
    false_pres = pd.DataFrame(data = np.zeros((len(false_index),3),dtype=np.int64), 
                              columns=['index','real_category','predicted_category'])
    false_pres['index'] = false_index
    false_pres['real_category'] = y_correct
    false_pres['predicted_category'] = pre_false
    false_pres.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True, mode='a+')

def save_classifiers_comparison(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name,
                                scores, ALL_SUBJECTS_ID, preprocess_time, inference_time,
                                SUBJECT_NUM, flops, paras):
    
    for i in range(len(CLASSIFIERS)):
        if i == 0:
            CLASSIFIERS_names = CLASSIFIERS[0]+'&'
        else:
            CLASSIFIERS_names = CLASSIFIERS_names+CLASSIFIERS[i]+'&'
    classifiers_comparison_log_dir = os.path.join(MODELS_COMP_LOG_DIR, CLASSIFIERS_names + '-comparison' + '.csv')
    
    index_metrics = ["accuracy","macro-precision","macro-recall","macro-f1",
                     "weighted-f1","micro-f1","inference-time","param_num","flops"]
    
    # record Averaged_SUBJECT_scores
    averaged_score_pd = pd.DataFrame(data    = np.zeros((len(index_metrics), 1),dtype=np.str_),
                                     index   = index_metrics,
                                     columns = [classifier_name])
    for row in averaged_score_pd.index:
        for column in averaged_score_pd.columns:
            if row not in ["inference-time","param_num","flops"]:
                averaged_score_pd.loc[row][column] = np.str_(np.mean(scores[row]["test"])) + '+-' + np.str_(np.std(scores[row]["test"]))
            elif row == "inference-time":
                averaged_score_pd.loc[row][column] = np.str_(preprocess_time + inference_time)
            elif row == "flops":
                averaged_score_pd.loc[row][column] = flops
            else:
                averaged_score_pd.loc[row][column] = paras
    
    # record Every_SUBJECT_scores
    for subject_id in ALL_SUBJECTS_ID:
            # record Averaged_SUBJECT_scores
            persub_score_pd = pd.DataFrame(data = np.zeros((3, 1),dtype=np.str_),
                                           index=["sub_"+str(subject_id)+"_accuracy",
                                                  "sub_"+str(subject_id)+"_macro-f1",
                                                  "sub_"+str(subject_id)+"_weighted-f1"],
                                           columns=[classifier_name])
            for row in persub_score_pd.index:
                for column in persub_score_pd.columns:
                    persub_score_pd.loc[row][column] = np.str_(scores[row.replace("sub_"+str(subject_id)+"_", '')]["test"][subject_id-1])
            if subject_id == min(ALL_SUBJECTS_ID):
                persub_score_pd_concat = persub_score_pd
            else:
                persub_score_pd_concat = pd.concat([persub_score_pd_concat, persub_score_pd], axis=0)
    
    if classifier_name == CLASSIFIERS[0]:
        if os.path.exists(classifiers_comparison_log_dir):
            os.remove(classifiers_comparison_log_dir)
        _ = create_directory(MODELS_COMP_LOG_DIR)
        
        # save Averaged_SUBJECT_scores to CSV
        pd.DataFrame(["Averaged_SUBJECT_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        averaged_score_pd.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
        
        # save Every_SUBJECT_scores to CSV
        pd.DataFrame(["Every_SUBJECT_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        persub_score_pd_concat.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')

    else:
        # add averaged_scores of new classifier
        saved_averaged_scores  = pd.read_csv(classifiers_comparison_log_dir, skiprows=1, nrows=len(index_metrics), header=0, index_col=0)
        saved_averaged_scores  = pd.concat([saved_averaged_scores, averaged_score_pd], axis=1)
        
        # add every_subject_scores of new classifier
        saved_everysub_scores = pd.read_csv(classifiers_comparison_log_dir, skiprows=(len(index_metrics)+3), nrows=3*len(ALL_SUBJECTS_ID), header=0, index_col=0)
        saved_everysub_scores = pd.concat([saved_everysub_scores, persub_score_pd_concat], axis=1)
        
        os.remove(classifiers_comparison_log_dir)
        
        # save Averaged_SUBJECT_scores to CSV
        pd.DataFrame(["Averaged_SUBJECT_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        saved_averaged_scores.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
        
        # save Every_SUBJECT_scores to CSV
        pd.DataFrame(["Every_SUBJECT_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        saved_everysub_scores.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
    
############################ The Folowing Functions Are Constructed For MobiAct Datasets (ten folds) ############################
    
def training_process_fold(logger, fold_id, X_tr, X_val, X_test, Y_tr, Y_val,
                          y_test, nb_classes, classifier_parameter, classifier,
                          classifier_func, MODEL_DIR, PATTERN, EPOCH, BATCH_SIZE, LR,
                          log_training_duration, test_split):
    
    # log train, validation and test dataset info, log the network
    log_redivdataset_network_info_fold(logger, fold_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                  y_test, nb_classes, classifier_parameter, classifier)
    
    # train the network and save the best validation model
    output_directory_models = os.path.join(MODEL_DIR, 'FOLD_'+str(fold_id))
    flag_output_directory_models = create_directory(output_directory_models)
    if PATTERN == 'TRAIN':
        flag_output_directory_models = PATTERN
    if flag_output_directory_models is not None:
        if PATTERN == 'TEST':
            best_model_path = os.path.join(output_directory_models, 'best_validation_model.pkl')
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(
                    f"TEST mode requires a pretrained checkpoint at {best_model_path}. "
                    "Enable --auto-download-models or place the checkpoint manually."
                )
            print('Found pretrained model for FOLD_'+str(fold_id))
            per_training_duration = 0.0
            log_training_duration.append(per_training_duration)
        else:
            # for each fold, train the network and save the best validation model
            print('FOLD_'+str(fold_id)+': start to train')
            history, per_training_duration, log_training_duration = classifier_func.train_op(classifier, EPOCH, BATCH_SIZE, LR,
                                                                                             X_tr, Y_tr, X_val, Y_val, X_test, y_test,
                                                                                             output_directory_models, log_training_duration,
                                                                                             test_split)
    else:
        print('Already_done: '+'FOLD_'+str(fold_id))
        # read the training duration of current fold when available. HF-hosted
        # pretrained weights may contain only checkpoint files, so fall back to 0.0.
        score_path = os.path.join(output_directory_models, 'score.csv')
        if os.path.exists(score_path):
            per_training_duration = pd.read_csv(score_path, skiprows=1, nrows=1, header = None)[1][0]
        else:
            per_training_duration = 0.0
        log_training_duration.append(per_training_duration)
    return per_training_duration, log_training_duration, output_directory_models

def save_metrics_per_fold(score, per_training_duration,
                        fold_id, nb_classes, LABELS,
                        y_true, y_pred,
                        output_directory_models):
    
    # save training time
    per_training_duration_pd = pd.DataFrame(data = per_training_duration,
                                            index = ["training duration"],
                                            columns = ["FOLD_"+str(fold_id)])
    per_training_duration_pd.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True)
    
    # save "logloss", "accuracy", "precision", "recall", "f1"
    score_pd = pd.DataFrame(data = np.zeros((7,3),dtype=np.float),
                            index=["logloss", "accuracy", "macro-precision", "macro-recall",
                                   "macro-f1", "weighted-f1", "micro-f1"], 
                            columns=["train", "valid", "test"])
    for row in score_pd.index:
        for column in score_pd.columns:
            score_pd.loc[row, column] = score[row][column][0]
    score_pd.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True, mode='a+')
    
    # save "per_class_f1"
    pd.DataFrame(["per_class_f1"]).to_csv(os.path.join(output_directory_models, 'score.csv'), index=False, header=False, mode='a+')
    per_class_f1_pd = pd.DataFrame(data = np.zeros((3, nb_classes),dtype=np.str_),
                            index=["train", "valid", "test"], columns=LABELS)
    for row in per_class_f1_pd.index:
        flag = 0
        if row == "test":
            for (i, column) in enumerate(per_class_f1_pd.columns):
                if i in list(np.unique(y_true)):
                    per_class_f1_pd.loc[row, column] = np.str_(score["per_class_f1"][row][0][flag])
                    flag = flag + 1
                else:
                    per_class_f1_pd.loc[row, column] = 'missed_category'
        else:
            for (i, column) in enumerate(per_class_f1_pd.columns):
                per_class_f1_pd.loc[row, column] = score["per_class_f1"][row][0][i]
            
    per_class_f1_pd.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True, mode='a+')
    
    # save confusion_matrix
    for key in score['confusion_matrix'].keys():
        pd.DataFrame(["confusion_matrix_"+key]).to_csv(os.path.join(output_directory_models, 'score.csv'), index=False, header=False, mode='a+')
        each_confusion_matrix = pd.DataFrame(data = np.zeros((nb_classes, nb_classes),dtype=np.float), 
                                             index = LABELS, columns=LABELS)
        # if missing categories exist
        if key == 'test':
            # two loops, one for row and one for column
            flag_cfm_row = 0
            # row loop
            for (i,row) in enumerate(each_confusion_matrix.index):
                if i in list(np.unique(y_true)):
                    flag_cfm_col = 0
                    # column loop
                    for (j,column) in enumerate(each_confusion_matrix.columns):
                        if j in list(np.unique(y_true)):
                            each_confusion_matrix.loc[row, column] = score['confusion_matrix'][key][0][flag_cfm_row][flag_cfm_col]
                            flag_cfm_col = flag_cfm_col + 1
                        else:
                            each_confusion_matrix.loc[row, column] = 'missed_category'
                    flag_cfm_row = flag_cfm_row + 1
                else:
                    for (j,column) in enumerate(each_confusion_matrix.columns):
                        each_confusion_matrix.loc[row, column] = 'missed_category'
        else:
            for (i,row) in enumerate(each_confusion_matrix.index):
                for (j,column) in enumerate(each_confusion_matrix.columns):
                    each_confusion_matrix.loc[row, column] = score['confusion_matrix'][key][0][i][j]
        each_confusion_matrix.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True, mode='a+')
    
    # save the indexes of the false predictions
    y_pred = y_pred.argmax(axis=1)
    false_index = np.where(np.array(y_true)!=np.array(y_pred))[0].tolist()
    y_correct = np.array(y_true)[np.array(y_true)!=np.array(y_pred)].tolist()
    pre_false = np.array(y_pred)[np.array(y_true)!=np.array(y_pred)].tolist()
    false_pres = pd.DataFrame(data = np.zeros((len(false_index),3),dtype=np.int64), 
                              columns=['index','real_category','predicted_category'])
    false_pres['index'] = false_index
    false_pres['real_category'] = y_correct
    false_pres['predicted_category'] = pre_false
    false_pres.to_csv(os.path.join(output_directory_models, 'score.csv'), index=True, mode='a+')
    
def partition(list_in, n):
    np.random.seed(66)
    np.random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]
    
def save_classifiers_comparison_fold(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name,
                                     scores, ALL_FOLDS_ID, preprocess_time, inference_time,
                                     FOLD_NUM, flops, paras):
    
    for i in range(len(CLASSIFIERS)):
        if i == 0:
            CLASSIFIERS_names = CLASSIFIERS[0]+'&'
        else:
            CLASSIFIERS_names = CLASSIFIERS_names+CLASSIFIERS[i]+'&'
    classifiers_comparison_log_dir = os.path.join(MODELS_COMP_LOG_DIR, CLASSIFIERS_names + '-comparison' + '.csv')
    
    index_metrics = ["accuracy","macro-precision","macro-recall","macro-f1",
                     "weighted-f1","micro-f1","inference-time","param_num","flops"]
    
    # record Averaged_SUBJECT_scores
    averaged_score_pd = pd.DataFrame(data    = np.zeros((len(index_metrics), 1),dtype=np.str_),
                                     index   = index_metrics,
                                     columns = [classifier_name])
    for row in averaged_score_pd.index:
        for column in averaged_score_pd.columns:
            if row not in ["inference-time","param_num","flops"]:
                averaged_score_pd.loc[row][column] = np.str_(np.mean(scores[row]["test"])) + '+-' + np.str_(np.std(scores[row]["test"]))
            elif row == "inference-time":
                averaged_score_pd.loc[row][column] = np.str_(preprocess_time + inference_time)
            elif row == "flops":
                averaged_score_pd.loc[row][column] = flops
            else:
                averaged_score_pd.loc[row][column] = paras
    
    # record Every_FOLD_scores
    for fold_id in range(len(ALL_FOLDS_ID)):
        
            # record Averaged_FOLD_scores
            persub_score_pd = pd.DataFrame(data = np.zeros((3, 1),dtype=np.str_),
                                           index=["fold_"+str(fold_id)+"_accuracy",
                                                  "fold_"+str(fold_id)+"_macro-f1",
                                                  "fold_"+str(fold_id)+"_weighted-f1"],
                                           columns=[classifier_name])
            for row in persub_score_pd.index:
                for column in persub_score_pd.columns:
                    persub_score_pd.loc[row][column] = np.str_(scores[row.replace("fold_"+str(fold_id)+"_", '')]["test"][fold_id])
            if fold_id == 0:
                persub_score_pd_concat = persub_score_pd
            else:
                persub_score_pd_concat = pd.concat([persub_score_pd_concat, persub_score_pd], axis=0)
    
    if classifier_name == CLASSIFIERS[0]:
        if os.path.exists(classifiers_comparison_log_dir):
            os.remove(classifiers_comparison_log_dir)
        _ = create_directory(MODELS_COMP_LOG_DIR)
        
        # save Averaged_FOLD_scores to CSV
        pd.DataFrame(["Averaged_FOLD_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        averaged_score_pd.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
        
        # save Every_FOLD_scores to CSV
        pd.DataFrame(["Every_FOLD_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        persub_score_pd_concat.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')

    else:
        # add averaged_scores of new classifier
        saved_averaged_scores  = pd.read_csv(classifiers_comparison_log_dir, skiprows=1, nrows=len(index_metrics), header=0, index_col=0)
        saved_averaged_scores  = pd.concat([saved_averaged_scores, averaged_score_pd], axis=1)
        
        # add every_fold_scores of new classifier
        saved_everysub_scores = pd.read_csv(classifiers_comparison_log_dir, skiprows=(len(index_metrics)+3), nrows=3*len(ALL_FOLDS_ID), header=0, index_col=0)
        saved_everysub_scores = pd.concat([saved_everysub_scores, persub_score_pd_concat], axis=1)
        os.remove(classifiers_comparison_log_dir)
        
        # save Averaged_FOLD_scores to CSV
        pd.DataFrame(["Averaged_FOLD_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        saved_averaged_scores.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
        
        # save Every_FOLD_scores to CSV
        pd.DataFrame(["Every_FOLD_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header = False, mode='a+')
        saved_everysub_scores.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
        
def log_dataset_network_info_fold(logger, K_folds_subjects, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, 
                                  X_train, X_test, y_train, y_test, cal_attitude_angle, 
                                  ACT_LABELS, ActID, label2act, BATCH_SIZE, EPOCH, LR):
    
    # log the information of datasets
    log_dataset_info_fold(logger, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, X_train, X_test,
                          y_train, y_test, cal_attitude_angle, ACT_LABELS, ActID)
    
    # check the category imbalance
    nb_classes = len(np.unique(y_train))
    check_class_balance(logger, y_train.flatten(), y_test.flatten(),
                        label2act=label2act, n_class=nb_classes)
    
    # log the hyper-parameters
    log_HyperParameters_fold(logger, BATCH_SIZE, EPOCH, LR, len(K_folds_subjects))
    
    return nb_classes

def log_dataset_info_fold(logger, TRAIN_FOLDS_ID, TEST_FOLDS_ID, X_train, X_test,
                     y_train, y_test, cal_attitude_angle, ACT_LABELS, ActID):
    logger.debug("---Dataset and preprocessing information---")
    logger.debug(f"TRAIN_FOLDS_ID = {TRAIN_FOLDS_ID}")
    logger.debug(f"TEST_FOLDS_ID = {TEST_FOLDS_ID}")
    logger.debug(f"X_train_shape = {X_train.shape}, X_test_shape={X_test.shape}")
    logger.debug(f"Y_train_shape = {y_train.shape}, Y_test.shape={y_test.shape}")
    logger.debug(f"Cal_Attitude_Angle = {cal_attitude_angle}")
    logger.debug(f"ACT_LABELS = {ACT_LABELS}")
    logger.debug(f"ActID = {ActID}")

def log_HyperParameters_fold(logger, BATCH_SIZE, EPOCH, LR, K_FOLDS):
    logger.debug("---HyperParameters---")
    logger.debug(f"BATCH_SIZE : {BATCH_SIZE}, EPOCH : {EPOCH}, LR : {LR}, K_FOLDS_NUM : {K_FOLDS}")

def log_redivdataset_network_info_fold(logger, fold_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                  y_test, nb_classes, classifier_parameter, classifier):
    if fold_id == 0:
        logger.debug("---Redivided dataset and network information---")
        logger.debug(f"X_train_shape={X_tr.shape}, X_validation_shape={X_val.shape}, X_test_shape={X_test.shape}")
        logger.debug(f"Y_train_shape={Y_tr.shape}, Y_validation_shape={Y_val.shape}, y_test_shape={y_test.shape}")
        logger.debug(f"num of categories = {nb_classes}")
        logger.debug(f"num of network parameter = {classifier_parameter}")
        logger.debug(f"the architecture of the network = {classifier}")

def log_inference_time_fold(start1, end1, y_train, y_test, time2,
                       dataset_name, classifier_name, FOLD_NUM, logger):
    time1 = (end1 - start1)/(len(y_train)+len(y_test))
    time2 = time2/((len(y_train)+len(y_test))*FOLD_NUM)
    logger.debug("---Final Inference Time Per Sample-Averaged over Folds---")
    logger.debug(f"preprocessing_time={time1}")
    logger.debug(f"pure_inference_time={time2}")
    logger.debug(f"single_model_prepro_inference_time={time1+time2}")
    logger.debug(f"all_model_prepro_inference_time={time1+time2*FOLD_NUM}")
    
    return time1, time2

def log_final_test_results_fold(logger, log_training_duration, scores, label2act, nb_classes, K_folds_subjects,
                                start1, end1, X_tr, y_train, y_test, time2, dataset_name, classifier_name, MODELS_COMP_LOG_DIR,
                                CLASSIFIERS, classifier, classifier_parameter, fileHandler):
    
    # Log the Test Scores of every fold
    log_every_FOLD_score(logger, log_training_duration, scores, label2act, nb_classes, len(K_folds_subjects))
    
    # Log the averaged Score of different folds
    log_averaged_FOLD_scores(logger, log_training_duration, scores, label2act, nb_classes, len(K_folds_subjects))
    
    # Log the inference time including data loading, preprocessing and model inference
    preprocess_time, _ = log_inference_time_fold(start1, end1, y_train, y_test, time2,
                                                         dataset_name, classifier_name, len(K_folds_subjects), logger)
    
    # Log the flops and param_num of networks
    flops, inference_time = log_flops(logger, classifier, X_tr)
    
    save_classifiers_comparison_fold(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name,
                                     scores, K_folds_subjects, preprocess_time, inference_time,
                                     len(K_folds_subjects), flops, classifier_parameter['Trainable'])
    
    # remove current logger
    remove_log(logger, fileHandler)

def log_every_FOLD_score(logger, log_training_duration, scores, label2act, nb_classes, FOLD_NUM):
    for i in range(FOLD_NUM):
        
        # Log Every Fold Scores
        logger.debug("---Per Fold Scores, Fold"+str(i)+"---")
        
        # log per Fold training time
        logger.debug(f"Training Duration = {log_training_duration[i]}s")
        
        for mode in ["train", "valid", "test"]:
            # log the average of "logloss", "accuracy", "precision", "recall", "f1"
            logger.debug(f"---{mode}---")
            logger.debug(f"logloss={round(scores['logloss'][mode][i])}, \
                         accuracy={round(scores['accuracy'][mode][i])}, \
                         macro-precision={round(scores['macro-precision'][mode][i])}, \
                         macro-recall={round(scores['macro-recall'][mode][i])}, \
                         macro-f1={round(scores['macro-f1'][mode][i])}, \
                         weighted-f1={round(scores['weighted-f1'][mode][i])}, \
                             micro-f1={round(scores['micro-f1'][mode][i])}")

def log_averaged_FOLD_scores(logger, log_training_duration, scores, label2act, nb_classes, FOLD_NUM):
    
    # Log Averaged Score of all Folds
    logger.debug("---Fold Averaged Scores---")
    
    # log the average of training time
    logger.debug(f"Averaged Training Duration = {(np.mean(log_training_duration))}s")
    
    for mode in ["train", "valid", "test"]:
        
        # log the average of "logloss", "accuracy", "precision", "recall", "f1"
        logger.debug(f"---{mode}---")
        for metric in ["logloss", "accuracy", "macro-precision", "macro-recall", "macro-f1", "weighted-f1", "micro-f1"]:
            logger.debug(f"{metric}={round(np.mean(scores[metric][mode]))} +- {round(np.std(scores[metric][mode]))}")