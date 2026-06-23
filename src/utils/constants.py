import os
import numpy as np
import yaml
import argparse

HF_DATASET_REPO = "crocodilegogogo/TSF-Datasets"
HF_MODEL_REPO = "crocodilegogogo/TSF-Models"

# Backwards-compatible default. Runtime code should prefer args.INFERENCE_DEVICE.
INFERENCE_DEVICE = "TEST_CUDA"

def parse_args():
    """Parse command-line arguments for TSF training/testing."""
    parser = argparse.ArgumentParser(description="TSF for HAR")

    parser.add_argument(
        "--PATTERN",
        type=str,
        default="TRAIN",
        choices=["TRAIN", "TEST"],
        help="Execution mode: TRAIN or TEST.",
    )
    parser.add_argument(
        "--DATASETS",
        nargs="+",
        default=["DSADS"],
        help="Dataset names, e.g. HAPT, Motion_Sense, SHL_2018, HHAR, MobiAct, Opportunity, Pamap2, DSADS, RealWorld, SHO.",
    )
    parser.add_argument(
        "--CLASSIFIERS",
        nargs="+",
        default=["TSF_torch"],
        help="Classifier names, e.g. Deep_Conv_LSTM_torch, Deep_ConvLSTM_Attn_torch, DeepSense_torch, AttnSense_torch, \
        GlobalFusion_torch, Transformer_Encoder_torch, Attend_And_Discriminate_torch, DynamicWHAR_torch, Deep_Conv_Transformer_torch, \
        IF_ConvTransformer_torch, Attn_Boost_Single_torch, ConvBoost_Single_torch, TSF_torch.",
    )
    parser.add_argument(
        "--test_split",
        type=int,
        default=50,
        help="Number of pieces used to split the test set during inference.",
    )
    parser.add_argument(
        "--INFERENCE_DEVICE",
        type=str,
        default="TEST_CUDA",
        choices=["TEST_CUDA", "TEST_CPU"],
        help="TEST_CUDA uses CUDA when available; TEST_CPU forces CPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6,
        help="Random seed used by Python, NumPy, and PyTorch.",
    )
    parser.add_argument(
        "--auto-download-data",
        dest="auto_download_data",
        action="store_true",
        default=True,
        help="Automatically download missing datasets from Hugging Face before loading data. Enabled by default.",
    )
    parser.add_argument(
        "--no-auto-download-data",
        dest="auto_download_data",
        action="store_false",
        help="Disable automatic dataset downloads.",
    )
    parser.add_argument(
        "--auto-download-models",
        dest="auto_download_models",
        action="store_true",
        default=True,
        help="Automatically download missing TSF model weights from Hugging Face in TEST mode. Enabled by default.",
    )
    parser.add_argument(
        "--no-auto-download-models",
        dest="auto_download_models",
        action="store_false",
        help="Disable automatic model weight downloads.",
    )
    parser.add_argument(
        "--hf-dataset-repo",
        type=str,
        default=HF_DATASET_REPO,
        help="Hugging Face dataset repo ID used by the automatic data pipeline.",
    )
    parser.add_argument(
        "--hf-model-repo",
        type=str,
        default=HF_MODEL_REPO,
        help="Hugging Face model repo ID used by the automatic model pipeline.",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision, branch, tag, or commit SHA.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory. If omitted, dataset cache is stored under datasets/.hf_cache and model cache under src/saved_models/.hf_cache, not the system user cache.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face access token. If omitted, huggingface_hub uses the cached login or HF_TOKEN env var.",
    )
    parser.add_argument(
        "--force-hf-download",
        action="store_true",
        help="Force rematerializing datasets/model weights from Hugging Face even if local markers exist.",
    )

    return parser.parse_args()

def get_HAPT_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = os.path.join(filepath, 'datasets', 'UCI HAPT', 'HAPT_Dataset')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    ACT_LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
                  "SITTING", "STANDING", "LAYING",
                  "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
                  "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    ActID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    TRAIN_SUBJECTS_ID = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    TEST_SUBJECTS_ID  = [2, 4, 9, 10, 12, 13, 18, 20, 24]
    ALL_SUBJECTS_ID   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                         25, 26, 27, 28, 29, 30]
    WINDOW_SIZE       = 128 # default: 128
    OVERLAP           = 64  # default: 64
    if separate_gravity_flag == True:
        INPUT_CHANNEL     = 9
    else:
        INPUT_CHANNEL     = 6
    cal_attitude_angle = True
    STFT_intervals    = 16
    POS_NUM           = 1
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID,\
           TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM,\
           cal_attitude_angle, STFT_intervals

def get_Motion_Sense_dataset_param(CUR_DIR, dataset_name):
    
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = os.path.join(filepath, 'datasets', 'Motion-Sense')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
    ActID = [0, 1, 2, 3, 4, 5]
    TRAIN_SUBJECTS_ID = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 17, 18, 21, 22, 23, 24]
    TEST_SUBJECTS_ID = [1, 7, 14, 16, 19, 20]
    TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
    }
    WINDOW_SIZE   = 128 # 128
    OVERLAP       = 10  # 10
    INPUT_CHANNEL = 9
    to_NED_flag   = True # for correcting the data using orientation
    cal_attitude_angle = False
    STFT_intervals     = 16
    POS_NUM            = 1
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
           TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, TRIAL_CODES,\
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM, to_NED_flag,\
           cal_attitude_angle, STFT_intervals

def get_SHL2018_dataset_param(CUR_DIR, dataset_name):
    
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = os.path.join(filepath, 'datasets', 'SHL2018')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    ACT_LABELS = ['Still', 'Walk', 'Run', 'Bike',
                  'Car', 'Bus', 'Train', 'Subway']
    ActID = [1, 2, 3, 4, 5, 6, 7, 8]
    TRAIN_SUBJECTS_ID = [1]
    TEST_SUBJECTS_ID = [1]
    SPLIT_NUM = 12        # Split 6000 length data to SPLIT_NUM pieces
    DATASET_SIZE = -1     # default:-1
    save_npy_flag = False
    to_NED_flag   = True # for correcting the data using quarternion
    STD_ALL = ['gra', 'mag', 'gyr', 'lacc', 'acc', 'label', 'ori'] # for saving npy
    STD     = ['gra', 'gyr', 'lacc', 'ori']
    INPUT_CHANNEL   = 9
    cal_attitude_angle = True
    STFT_intervals     = 50
    POS_NUM            = 1
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, \
           TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, SPLIT_NUM, \
           DATASET_SIZE, STD_ALL, STD, INPUT_CHANNEL, POS_NUM, \
           save_npy_flag, to_NED_flag, cal_attitude_angle, STFT_intervals

def get_HHAR_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'HHAR', 'Per_subject_npy')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    ACT_LABELS          = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
    ActID               = [0, 1, 2, 3, 4, 5]
    SUBJECTS            = ["a","b","c","d","e","f","g","h","i"]
    TRAIN_SUBJECTS_ID   = [1, 2, 3, 4, 6, 7, 8]
    TEST_SUBJECTS_ID    = [0, 5]
    WINDOW_SIZE         = 200 # default: 200
    OVERLAP             = 100 # default: 100
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 9
    else:
        INPUT_CHANNEL       = 6
    cal_attitude_angle  = False
    STFT_intervals      = 20
    POS_NUM             = 1
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, SUBJECTS, \
           TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, \
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle, \
           STFT_intervals

def get_MobiAct_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'MobiAct', 'Per_subject_no_NED_npy')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    ACT_LABELS          = ['STD','WAL','JOG','JUM','STU','STN','SCH','SIT','CHU','CSI','CSO']
    ActID               = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    SUBJECTS            = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 16,
                           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33,
                           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                           51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    TRAIN_SUBJECTS_ID   = [ 2,  3,  4,  6,  7,  8,  9, 10, 11, 12, 16,
                           18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 33,
                           34, 35, 36, 37, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                           51, 52, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67]
    TEST_SUBJECTS_ID    = [ 5, 22, 1, 32, 38, 40, 50, 53, 58, 65]
    WINDOW_SIZE         = 200            # default: 200
    OVERLAP             = WINDOW_SIZE//2 # default: 100
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 9
    else:
        INPUT_CHANNEL       = 6
    to_NED_flag         = False # for correcting the data using orientation
    cal_attitude_angle  = False
    STFT_intervals      = 20
    POS_NUM           = 1
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, SUBJECTS, \
           TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, \
           INPUT_CHANNEL, POS_NUM, to_NED_flag,\
           cal_attitude_angle, STFT_intervals
    
def get_Opportunity_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'Opportunity')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    SUBJECTS                = [1,2,3,4]
    TRIALS                  = [1,2,3,4,5]
    SELEC_LABEL             = 'MID_LABEL_COL' # 'LOCO_LABEL_COL', 'MID_LABEL_COL', 'HI_LABEL_COL'
    ACT_LABELS              = ['null', 'Open_Door_1', 'Open_Door_2', 'Close_Door_1', 'Close_Door_2', 'Open_Fridge',
                               'Close_Fridge', 'Open_Dishwasher', 'Close_Dishwasher', 'Open Drawer1','Close Drawer1',
                               'Open_Drawer2','Close_Drawer2', 'Open_Drawer3', 'Close_Drawer3', 'Clean_Table',
                               'Drink_Cup', 'Toggle_Switch']
    ACT_ID                  = (np.arange(18)).tolist()
    TRAIN_SUBJECTS_ID       = [1]
    TRAIN_SUBJECTS_TRIAL_ID = [1,2,3,4,5]
    WINDOW_SIZE             = 48 # default: 48
    OVERLAP                 = 24 # default: 24
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 63
    else:
        INPUT_CHANNEL       = 42
    to_NED_flag             = True
    cal_attitude_angle      = False
    STFT_intervals          = 6
    POS_NUM                 = 7
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS, TRIALS, SELEC_LABEL,\
           ACT_LABELS, ACT_ID, TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID,\
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM, \
           to_NED_flag, cal_attitude_angle, STFT_intervals

def get_Pamap2_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'Pamap2')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    SUBJECTS                = [1,2,3,4,5,6,7,8]
    TRAIN_SUBJECTS_ID       = [1]
    ACT_LABELS              = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling',
                               'Nordic_walking', 'ascending_stairs', 'descending_stairs','vacuum_cleaning',
                               'ironing','rope_jumping']
    Act_ID                  = [1,2,3,4,5,6,7,8,9,10,11,12]
    WINDOW_SIZE             = 48 # default: 48
    OVERLAP                 = 24 # default: 24
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 27
    else:
        INPUT_CHANNEL       = 18
    cal_attitude_angle      = False
    STFT_intervals          = 6
    POS_NUM                 = 3
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
           TRAIN_SUBJECTS_ID, ACT_LABELS, Act_ID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle,\
           STFT_intervals

def get_RealWorld_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'RealWorld')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    SUBJECTS                = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    TRAIN_SUBJECTS_ID       = [1]
    ACT_LABELS              = ['climbingdown','climbingup','jumping','lying',
                               'running','sitting','standing','walking']
    Act_ID                  = [1,2,3,4,5,6,7,8]
    WINDOW_SIZE             = 48 # default: 48
    OVERLAP                 = 24 # default: 24
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 63
    else:
        INPUT_CHANNEL       = 42
    POS_NUM                 = 7
    STFT_intervals          = 6
    cal_attitude_angle      = False
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
           TRAIN_SUBJECTS_ID, ACT_LABELS, Act_ID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals

def get_DSADS_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'DSADS')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    SUBJECTS                = [1,2,3,4,5,6,7,8]
    TRAIN_SUBJECTS_ID       = [1]
    ACT_LABELS              = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9','A10',
                               'A11','A12','A13','A14','A15','A16','A17','A18','A19']
    ACT_ID                  = (np.arange(19)+1).tolist()
    WINDOW_SIZE             = 125 # default: 125
    OVERLAP                 = 0   # default: 0
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 45
    else:
        INPUT_CHANNEL       = 30
    cal_attitude_angle      = False
    STFT_intervals          = 25
    POS_NUM                 = 5
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
           TRAIN_SUBJECTS_ID, ACT_LABELS, ACT_ID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle,\
           STFT_intervals

def get_SHO_dataset_param(CUR_DIR, dataset_name):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'SHO')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    SUBJECTS                = [1,2,3,4,5,6,7,8,9,10]
    TRAIN_SUBJECTS_ID       = [1]
    ACT_LABELS              = ['walking',  'standing', 'jogging', 'sitting', 'biking',
                               'upstairs', 'downstairs']
    Act_ID                  = [1,2,3,4,5,6,7]
    WINDOW_SIZE             = 48 # default: 48
    OVERLAP                 = 24 # default: 24
    INPUT_CHANNEL           = 45
    POS_NUM                 = 5
    STFT_intervals          = 6
    cal_attitude_angle      = False
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
           TRAIN_SUBJECTS_ID, ACT_LABELS, Act_ID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals

def create_classifier(dataset_name, classifier_name, input_channel, POS_NUM,
                      data_length, train_size, val_size, test_size, nb_classes, STFT_intervals,
                      BATCH_SIZE, INFERENCE_DEVICE, test_split):
    
    hparam_file     = open(os.path.join(os.path.dirname(__file__), 'hyperparams.yaml'), mode='r', encoding='utf-8')
    hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)
    conv_chnnl      = hyperparameters[classifier_name]['conv_chnnl'][dataset_name]
    context_chnnl   = hyperparameters[classifier_name]['context_chnnl'][dataset_name]
    dropoutrate     = hyperparameters['Dropout_rate'][dataset_name]
    
############################## comparison methods #############################
    
    if classifier_name=='Deep_Conv_LSTM_torch': 
        from classifiers.comparison_methods import Deep_Conv_LSTM_torch 
        # __init__(self, input_2Dfeature_channel, input_channel, kernel_size,
        #      feature_channel, hidden_size, drop_rate, num_class)
        return Deep_Conv_LSTM_torch.Deep_Conv_LSTM(1, input_channel, 5, conv_chnnl, context_chnnl, 0.2, nb_classes), Deep_Conv_LSTM_torch

    if classifier_name=='Deep_ConvLSTM_Attn_torch': 
        from classifiers.comparison_methods import Deep_ConvLSTM_Attn_torch
        # __init__(self, input_2Dfeature_channel, input_channel, kernel_size,
        #      feature_channel, hidden_size, drop_rate, num_class)
        return Deep_ConvLSTM_Attn_torch.Deep_ConvLSTM_Attn(1, input_channel, 5, conv_chnnl, context_chnnl, 0.2, nb_classes), Deep_ConvLSTM_Attn_torch

    if classifier_name=='DeepSense_torch':
        from classifiers.comparison_methods import DeepSense_torch
        # __init__(self, input_2Dfeature_channel, input_channel, POS_NUM, kernel_size,
        #      feature_channel, merge_kernel_size1, merge_kernel_size2, merge_kernel_size3,
        #      hidden_size, drop_rate, drop_rate_gru, num_class, datasetname)
        return DeepSense_torch.DeepSense(1, input_channel, POS_NUM, 3, conv_chnnl, 4, 3, 2, context_chnnl, 0, 0.2, nb_classes, dataset_name), DeepSense_torch

    if classifier_name=='AttnSense_torch':
        from classifiers.comparison_methods import AttnSense_torch
        # __init__(self, input_2Dfeature_channel, input_channel, POS_NUM, kernel_size,
        #      feature_channel, merge_kernel_size1, merge_kernel_size2, merge_kernel_size3,
        #      hidden_size, drop_rate, drop_rate_gru, num_class, datasetname)
        return AttnSense_torch.AttnSense(1, input_channel, POS_NUM, 3, conv_chnnl, 4, 3, 2, context_chnnl, 0, 0.2, nb_classes, dataset_name), AttnSense_torch

    if classifier_name=='Transformer_Encoder_torch':
        from classifiers.comparison_methods import Transformer_Encoder_torch
        # __init__(self, input_channel, kernel_size, feature_channel_2D, feature_channel,
        #      multiheads, drop_rate, data_length, num_class)
        return Transformer_Encoder_torch.Transformer_Encoder(input_channel, 1, conv_chnnl, context_chnnl, 1, 0.2, data_length, nb_classes), Transformer_Encoder_torch

    if classifier_name=='GlobalFusion_torch': 
        from classifiers.comparison_methods import GlobalFusion_torch
        # __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
        #      kernel_size, kernel_size_grav, scale_num, feature_channel_out,
        #      multiheads, drop_rate, dataset_name, spe_interv, sin_pos_chnnl, num_class)
        return GlobalFusion_torch.GlobalFusion(1, input_channel, conv_chnnl, 3, 3, 2, context_chnnl, 1, 0.2, dataset_name, \
                                               data_length, input_channel//POS_NUM, nb_classes), GlobalFusion_torch

    if classifier_name=='Attend_And_Discriminate_torch': 
        from classifiers.comparison_methods import Attend_And_Discriminate_torch
        # __init__(
        #     self, input_dim, filter_num, filter_size, hidden_dim, enc_num_layers, enc_is_bidirectional,
        #     dropout, dropout_rnn, dropout_cls, activation, sa_div, num_class, train_mode)
        return Attend_And_Discriminate_torch.Attend_And_Discriminate(input_channel, conv_chnnl, 5, context_chnnl, 2, False,\
                                                                     0.2, 0.25, 0.2, "ReLU", 1, nb_classes,\
                                                                     "True"), Attend_And_Discriminate_torch

    if classifier_name=='DynamicWHAR_torch': 
        from classifiers.comparison_methods import DynamicWHAR_torch 
        # __init__(self, node_num, node_dim, window_size, channel_dim, time_reduce_size, hid_dim, class_num)
        return DynamicWHAR_torch.DynamicWHAR(POS_NUM, node_dim=input_channel//POS_NUM, window_size=data_length,
                                             time_reduce_size=conv_chnnl, hid_dim=context_chnnl, class_num=nb_classes,
                                             INFERENCE_DEVICE=INFERENCE_DEVICE), DynamicWHAR_torch

    if classifier_name=='Deep_Conv_Transformer_torch': 
        from classifiers.comparison_methods import Deep_Conv_Transformer_torch
        # __init__(self, input_2Dfeature_channel, input_channel, kernel_size, feature_channel,
        #      feature_channel_out, multiheads, drop_rate, data_length, num_class)
        return Deep_Conv_Transformer_torch.Deep_Conv_Transformer(1, input_channel, 7, conv_chnnl, context_chnnl, 8, 0.2, data_length, nb_classes), Deep_Conv_Transformer_torch

    if classifier_name == 'IF_ConvTransformer_torch' and dataset_name in ['HAPT','Motion_Sense','SHL_2018','HHAR','MobiAct']:
        from classifiers.comparison_methods import IF_ConvTransformer_Phone_torch
        # __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
        #      kernel_size, kernel_size_grav, scale_num, feature_channel_out,
        #      multiheads, drop_rate, dataset_name, data_length, num_class)
        return IF_ConvTransformer_Phone_torch.IF_ConvTransformer_Phone(1, input_channel, conv_chnnl, 5, 3, 2, context_chnnl, 1, 0.2, \
                                                                       dataset_name, data_length, nb_classes), IF_ConvTransformer_Phone_torch

    if classifier_name == 'IF_ConvTransformer_torch' and dataset_name in ['Opportunity','Pamap2','DSADS','RealWorld','SHO']:
        from classifiers.comparison_methods import IF_ConvTransformer_WearableDevice_torch
        # __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
        #      kernel_size, kernel_size_grav, scale_num, feature_channel_out,
        #      multiheads, drop_rate, dataset_name, data_length, num_class)
        return IF_ConvTransformer_WearableDevice_torch.IF_ConvTransformer_WearableDevice(1, input_channel, conv_chnnl, 5, 3, 2, \
                                                                                         context_chnnl, 1, 0.2, dataset_name, data_length, nb_classes), IF_ConvTransformer_WearableDevice_torch

    if classifier_name=='Attn_Boost_Single_torch': 
        from classifiers.comparison_methods import Attn_Boost_Single_torch 
        # __init__(self, input_dim, n_classes, FILTER_SIZE, NUM_FILTERS, hidden_size, num_layers=2, is_bidirectional=False, dropout=0.2,
        #          attention_dropout=0.2)
        return Attn_Boost_Single_torch.Attn_Boost_Single(input_channel, nb_classes, 5, conv_chnnl, context_chnnl), Attn_Boost_Single_torch

    if classifier_name=='ConvBoost_Single_torch': 
        from classifiers.comparison_methods import ConvBoost_Single_torch 
        # __init__(self, input_channel, num_classes, cnn_channel, n_hidden, drop_rate)
        return ConvBoost_Single_torch.ConvBoost_Single(input_channel, nb_classes, conv_chnnl, context_chnnl, 0.2), ConvBoost_Single_torch

    if classifier_name=='TSF_torch': 
        from classifiers.comparison_methods import TSF_torch
        # __init__(self, input_2Dfeature_channel, input_channel, feature_channel,
        #      kernel_size, kernel_size_grav, scale_num, feature_channel_out,
        #      multiheads, drop_rate, dataset_name, POS_NUM, data_length, train_size,
        #      val_size, test_size, num_class)
        return TSF_torch.TSF(1, input_channel, conv_chnnl, 11, 5, 1, context_chnnl, 1, dropoutrate, dataset_name, POS_NUM, data_length, 
                             train_size, val_size, test_size, nb_classes, BATCH_SIZE, INFERENCE_DEVICE, test_split), TSF_torch