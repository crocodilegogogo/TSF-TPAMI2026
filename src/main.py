import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import time
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
from typing import Any, Dict, List
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from utils.constants import *
from utils.utils import *
from utils.hf_downloader import ensure_tsf_model_weights

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory

class Train_Test(object):
    def __init__(self, args):

        # Initial
        self.args  = args

        for dataset_name in args.DATASETS:
            
            sep_flags = []
            BATCH_SIZE, EPOCH, LR, K_FOLDS = get_hyperparams(os.path.join(CUR_DIR, 'utils', 'hyperparams.yaml'), dataset_name)
            
            for (classifer_id, classifier_name) in enumerate(args.CLASSIFIERS):
                
                ############################### LOAD RAW DATA ################################
                cur_sep_flag, sep_flags = get_sep_flags(classifier_name, dataset_name, sep_flags)
                if classifer_id == 0 or sep_flags[classifer_id] != sep_flags[classifer_id-1]:
                    
                    All_data, All_labels, All_users, ALL_SUBJECTS_ID, X_train, label2act, ACT_LABELS, ActID, \
                       MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals = load_all_data(classifer_id,
                                                                    classifier_name, dataset_name, cur_sep_flag, CUR_DIR)
                
                ########################################## LEAVE ONE SUBJECT OUT ##########################################
                if dataset_name != 'MobiAct':
                    # set logging settings
                    EXEC_TIME, LOG_DIR, MODEL_DIR, logger, fileHandler = logging_settings(classifier_name, CUR_DIR, dataset_name)
                    if args.PATTERN == 'TEST' and getattr(args, "auto_download_models", True):
                        ensure_tsf_model_weights(
                            dataset_name,
                            classifier_name,
                            MODEL_DIR,
                            repo_id=getattr(args, "hf_model_repo", "crocodilegogogo/TSF-Models"),
                            revision=getattr(args, "hf_revision", None),
                            cache_dir=getattr(args, "hf_cache_dir", None),
                            token=getattr(args, "hf_token", None),
                            force=getattr(args, "force_hf_download", False),
                        )
                    time2 = 0
                    
                    for subject_id in ALL_SUBJECTS_ID:
                        # get train and test subjects
                        TEST_SUBJECTS_ID  = [subject_id]
                        TRAIN_SUBJECTS_ID = list(set(ALL_SUBJECTS_ID).difference(set(TEST_SUBJECTS_ID)))
                        
                        ########################## DATA PREPROCESSING #########################
                        X_train, y_train, X_test, y_test, start1, end1 = data_preprocessing(ALL_SUBJECTS_ID, TRAIN_SUBJECTS_ID,
                                                                                            TEST_SUBJECTS_ID, All_users, All_data,
                                                                                            All_labels, X_train, classifier_name,
                                                                                            STFT_intervals, POS_NUM, args.test_split)
                        #######################################################################
                        
                        ############### LOG DATASET INFO AND NETWORK PARAMETERS ###############
                        nb_classes = log_dataset_network_info(logger, ALL_SUBJECTS_ID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,
                                                              X_train, X_test, y_train, y_test, cal_attitude_angle,
                                                              ACT_LABELS, ActID, label2act, BATCH_SIZE, EPOCH, LR)
                        #######################################################################
                        
                        ##### SPLIT TRAINSET TO TRAIN AND VAL DATASETS #####
                        # Initilize the logging variables
                        if subject_id == min(ALL_SUBJECTS_ID):
                            models, scores, log_training_duration = initialize_saving_variables(X_train, X_test, nb_classes)
                        
                        X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, y_train, 
                                                                    test_size=0.1,
                                                                    random_state=args.seed,
                                                                    stratify=y_train)
                        ######################## Training and Testing Process ############################
                        # Create classifier
                        classifier, classifier_func, classifier_parameter = create_cuda_classifier(dataset_name, classifier_name,
                                                                                                   INPUT_CHANNEL, POS_NUM, X_tr.shape[-1],
                                                                                                   X_tr.shape[0], X_val.shape[0],
                                                                                                   X_test.shape[0], nb_classes, STFT_intervals,
                                                                                                   BATCH_SIZE, args.INFERENCE_DEVICE, args.test_split)
                        
                        ###### TRAIN FOR EACH SUBJECT (if have already finished training, print 'Already_done') ######
                        per_training_duration, log_training_duration, output_directory_models = training_process(
                                                                                                   logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                                                                                   y_test, nb_classes, classifier_parameter, classifier,
                                                                                                   classifier_func, MODEL_DIR, args.PATTERN, EPOCH, BATCH_SIZE, LR,
                                                                                                   log_training_duration, args.test_split)
                        ########## TEST FOR EACH SUBJECT, record inference time ###########
                        pred_train, pred_valid, pred_test, scores, time_duration = classifier_func.predict_tr_val_test(dataset_name, classifier, nb_classes, ACT_LABELS,
                                                                                                                       X_tr, X_val, X_test,
                                                                                                                       Y_tr, Y_val, y_test,
                                                                                                                       scores, per_training_duration,
                                                                                                                       subject_id, output_directory_models,
                                                                                                                       args.test_split)
                        
                        time2 = time2 + time_duration
                        ##################################################################################
                        
                    ################ LOG TEST RESULTS, LOG INFERENCE TIME #################
                    log_final_test_results(logger, log_training_duration, scores, label2act, nb_classes, ALL_SUBJECTS_ID,
                                           start1, end1, X_tr, y_train, y_test, time2, dataset_name, classifier_name, MODELS_COMP_LOG_DIR,
                                           args.CLASSIFIERS, classifier, classifier_parameter, fileHandler)
                    #######################################################################
                
                ######################################## K-FOLD CROSS VALIDATION ###########################################
                else: # MobiAct Dataset, ten folds
                    
                    K_folds_subjects = get_K_folds_subjects(ALL_SUBJECTS_ID, K_FOLDS)
                    # set logging settings
                    EXEC_TIME, LOG_DIR, MODEL_DIR, logger, fileHandler = logging_settings(classifier_name, CUR_DIR, dataset_name)
                    if args.PATTERN == 'TEST' and getattr(args, "auto_download_models", True):
                        ensure_tsf_model_weights(
                            dataset_name,
                            classifier_name,
                            MODEL_DIR,
                            repo_id=getattr(args, "hf_model_repo", "crocodilegogogo/TSF-Models"),
                            revision=getattr(args, "hf_revision", None),
                            cache_dir=getattr(args, "hf_cache_dir", None),
                            token=getattr(args, "hf_token", None),
                            force=getattr(args, "force_hf_download", False),
                        )
                    time2 = 0
                    
                    for (fold_id, subject_id) in enumerate(K_folds_subjects):
                        
                        # get train and test subjects
                        TEST_SUBJECTS_ID  = subject_id
                        TRAIN_SUBJECTS_ID = list(set(ALL_SUBJECTS_ID).difference(set(TEST_SUBJECTS_ID)))
                        
                        ########################## DATA PREPROCESSING #########################
                        X_train, y_train, X_test, y_test, start1, end1 = data_preprocessing(ALL_SUBJECTS_ID, TRAIN_SUBJECTS_ID,
                                                                                            TEST_SUBJECTS_ID, All_users, All_data,
                                                                                            All_labels, X_train, classifier_name,
                                                                                            STFT_intervals, POS_NUM, args.test_split)
                        #######################################################################
                        
                        ############### LOG DATASET INFO AND NETWORK PARAMETERS ###############
                        nb_classes = log_dataset_network_info_fold(logger, K_folds_subjects, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,
                                                                   X_train, X_test, y_train, y_test, cal_attitude_angle,
                                                                   ACT_LABELS, ActID, label2act, BATCH_SIZE, EPOCH, LR)
                        #######################################################################
                        
                        ##### SPLIT TRAINSET TO TRAIN AND VAL DATASETS #####
                        # Initilize the logging variables
                        if fold_id == 0:
                            models, scores, log_training_duration = initialize_saving_variables(X_train, X_test, nb_classes)
                        
                        X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, y_train, 
                                                                    test_size=0.1,
                                                                    random_state=args.seed,
                                                                    stratify=y_train)
                        ######################## Training and Testing Process ############################
                        # Create classifier
                        classifier, classifier_func, classifier_parameter = create_cuda_classifier(dataset_name, classifier_name,
                                                                                                   INPUT_CHANNEL, POS_NUM, X_tr.shape[-1],
                                                                                                   X_tr.shape[0], X_val.shape[0],
                                                                                                   X_test.shape[0], nb_classes, STFT_intervals,
                                                                                                   BATCH_SIZE, args.INFERENCE_DEVICE, args.test_split)
                        ###### TRAIN FOR EACH FOLD (if have already finished training, print 'Already_done') ######
                        per_training_duration, log_training_duration, output_directory_models = training_process_fold(
                                                                                                   logger, fold_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                                                                                   y_test, nb_classes, classifier_parameter, classifier,
                                                                                                   classifier_func, MODEL_DIR, args.PATTERN, EPOCH, BATCH_SIZE, LR,
                                                                                                   log_training_duration, args.test_split)
                        ########## TEST FOR EACH FOLD, record inference time ###########
                        pred_train, pred_valid, pred_test, scores, time_duration = classifier_func.predict_tr_val_test(dataset_name, classifier, nb_classes, ACT_LABELS,
                                                                                                   X_tr, X_val, X_test,
                                                                                                   Y_tr, Y_val, y_test,
                                                                                                   scores, per_training_duration,
                                                                                                   fold_id, output_directory_models,
                                                                                                   args.test_split)
                        time2 = time2 + time_duration
                        ##################################################################################
                        
                    ################ LOG TEST RESULTS, LOG INFERENCE TIME #################
                    log_final_test_results_fold(logger, log_training_duration, scores, label2act, nb_classes, K_folds_subjects,
                                                start1, end1, X_tr, y_train, y_test, time2, dataset_name, classifier_name, MODELS_COMP_LOG_DIR,
                                                args.CLASSIFIERS, classifier, classifier_parameter, fileHandler)
                    #######################################################################
                    
def main(args):
    configure_runtime(args)
    Main = Train_Test(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)