"The implementation of article 'AttnSense: multi-level attention mechanism for multimodal human activity recognition' (AttnSense)"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
import time
from utils.utils import *
import os

class Individial_Pos_Convs(nn.Module):
    def __init__(self, input_2Dfeature_channel, feature_channel, kernel_size, drop_rate):
        super(Individial_Pos_Convs, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_2Dfeature_channel, feature_channel, (1,2*3*3), (1,2*3), (0,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,3), 1, (0,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        self.dropout2 = nn.Dropout(drop_rate)
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,0)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = x.unsqueeze(dim=4)
        return x

class AttnSense(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, POS_NUM, kernel_size,
                 feature_channel, merge_kernel_size1, merge_kernel_size2, merge_kernel_size3,
                 hidden_size, drop_rate, drop_rate_gru, num_class, datasetname):
        
        super(AttnSense, self).__init__()
        
        self.datasetname = datasetname
        self.input_channel = input_channel
        self.POS_NUM       = POS_NUM
        
        if datasetname in ['DSADS','SHO']:
            kernel_size = 1
        
        self.Acc_Pos_Convs     = []
        self.Mag_Pos_Convs     = []
        self.Grav_Pos_Convs    = []
        self.Gyro_Pos_Convs    = []
        
        for i in range(POS_NUM):
            
            if input_channel//POS_NUM == 12:
                self.mag_convs = Individial_Pos_Convs(input_2Dfeature_channel, feature_channel, kernel_size, drop_rate)
                setattr(self, 'Mag_Pos_Convs%i' % i, self.mag_convs)
                self.Mag_Pos_Convs.append(self.mag_convs)
            
            if input_channel//POS_NUM == 9 or input_channel//POS_NUM == 12:
                self.acc_convs = Individial_Pos_Convs(input_2Dfeature_channel, feature_channel, kernel_size, drop_rate)
                setattr(self, 'Acc_Pos_Convs%i' % i, self.acc_convs)
                self.Acc_Pos_Convs.append(self.acc_convs)
                
            self.grav_convs = Individial_Pos_Convs(input_2Dfeature_channel, feature_channel, kernel_size, drop_rate)
            setattr(self, 'Grav_Pos_Convs%i' % i, self.grav_convs)
            self.Grav_Pos_Convs.append(self.grav_convs)
            
            self.gyro_convs = Individial_Pos_Convs(input_2Dfeature_channel, feature_channel, kernel_size, drop_rate)
            setattr(self, 'Gyro_Pos_Convs%i' % i, self.gyro_convs)
            self.Gyro_Pos_Convs.append(self.gyro_convs)
        
        self.merge_dropout = nn.Dropout(drop_rate)
        
        if self.datasetname in ['SHL_2018','HHAR','MobiAct','SHO']:
            attn_mul = 4
        elif self.datasetname in ['DSADS']:
            attn_mul = 1
        else:
            attn_mul = 2
        
        self.sensor_attention = nn.Sequential(
            nn.Linear(attn_mul*feature_channel,1),
            nn.Tanh()
            )
        
        if self.datasetname in ['SHL_2018','HHAR','MobiAct','SHO']:
            mul = 4
        elif self.datasetname in ['DSADS']:
            mul = 1
        else:
            mul = 2
        self.gru = nn.GRU(
            input_size=mul*feature_channel,
            hidden_size=hidden_size,
            num_layers = 2,
            batch_first = True
            )
        
        self.gru_dropout = nn.Dropout(drop_rate_gru)
        
        self.time_attention1 = nn.Sequential(
            nn.Linear(hidden_size,64),
            nn.Tanh()
            )
        
        self.time_attention2 = nn.Sequential(
            nn.Linear(64,1),
            nn.Tanh()
            )
        
        self.linear = nn.Linear(hidden_size, num_class)
        
    def forward(self, x, test_flag=False):
        
        # flops
        if len(x.shape) == 3:
            x           = x.unsqueeze(0)
        # flops
        batch_size = x.shape[0]
        
        for i in range(self.POS_NUM):
            x_pos = x[:,i,:,:].unsqueeze(1)
            if self.input_channel//self.POS_NUM == 12:
                inputs = torch.split(x_pos, x_pos.shape[3]//4, dim=3)
                grav_inputs = inputs[0]
                mag_inputs  = inputs[1]
                gyro_inputs = inputs[2]
                acc_inputs  = inputs[3]
                x_mag       = self.Mag_Pos_Convs[i](mag_inputs)
                x_acc       = self.Acc_Pos_Convs[i](acc_inputs)
            if self.input_channel//self.POS_NUM == 9:
                inputs = torch.split(x_pos, x_pos.shape[3]//3, dim=3)
                grav_inputs = inputs[0]
                gyro_inputs = inputs[1]
                acc_inputs  = inputs[2]
                x_acc       = self.Acc_Pos_Convs[i](acc_inputs)
            if self.input_channel//self.POS_NUM == 6:
                inputs = torch.split(x_pos, x_pos.shape[3]//2, dim=3)
                grav_inputs = inputs[0]
                gyro_inputs = inputs[1]
            x_grav          = self.Grav_Pos_Convs[i](grav_inputs)
            x_gyro          = self.Gyro_Pos_Convs[i](gyro_inputs)
            
            if i == 0:
                if self.input_channel//self.POS_NUM == 12:
                    x_all_sensor = torch.cat([x_acc, x_grav, x_gyro, x_mag],4)
                elif self.input_channel//self.POS_NUM == 9:
                    x_all_sensor = torch.cat([x_acc, x_grav, x_gyro],4)
                elif self.input_channel//self.POS_NUM == 6:
                    x_all_sensor = torch.cat([x_grav, x_gyro],4)
            else:
                if self.input_channel//self.POS_NUM == 12:
                    x_all_sensor = torch.cat([x_all_sensor, x_acc, x_grav, x_gyro, x_mag],4)
                elif self.input_channel//self.POS_NUM == 9:
                    x_all_sensor = torch.cat([x_all_sensor, x_acc, x_grav, x_gyro],4)
                elif self.input_channel//self.POS_NUM == 6:
                    x_all_sensor = torch.cat([x_all_sensor, x_grav, x_gyro],4)
        
        # cal sensor attention
        x_all_sensor = self.merge_dropout(x_all_sensor)
        # [batch, input_time_step, sensor_mode, input_channel, feature_dim]
        x_all_sensor = x_all_sensor.permute(0,2,4,3,1)
        # [batch, input_time_step, sensor_mode, input_channel*feature_dim]
        x_all_sensor = x_all_sensor.reshape([x_all_sensor.shape[0], x_all_sensor.shape[1], x_all_sensor.shape[2], -1])
        # sensor_attention: [batch, input_time_step, sensor_mode, 1]
        sensor_attention = self.sensor_attention(x_all_sensor)
        sensor_attention = F.softmax(sensor_attention.squeeze(-1), dim=2)
        
        # weighting sensors
        x_all_sensor = x_all_sensor * (sensor_attention.unsqueeze(-1))
        # x: [batch, input_time_step, input_channel*feature_dim]
        x = torch.sum(x_all_sensor, dim=2).squeeze(-1)
        
        # GRU calculation
        data_length = x.shape[1]
        x, hidden = self.gru(x, None)
        x = self.gru_dropout(x)
        
        # cal time attention
        time_attention1 = self.time_attention1(x)
        time_attention2 = self.time_attention2(time_attention1).squeeze(-1)
        time_attention2 = F.softmax(time_attention2, dim=1)
        
        # weighting time steps
        x = x * (time_attention2.unsqueeze(-1))
        x = torch.sum(x, dim=1).squeeze(-1)
        
        output = self.linear(x)
        
        return output, x
    
def train_op(network, EPOCH, BATCH_SIZE, LR,
              train_x, train_y, val_x, val_y, X_test, y_test,
              output_directory_models, log_training_duration, test_split):
     # prepare training_data
     if train_x.shape[0] % BATCH_SIZE == 1:
         drop_last_flag = True
     else:
         drop_last_flag = False
     torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
     train_loader = Data.DataLoader(dataset = torch_dataset,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True,
                                     drop_last = drop_last_flag
                                    )
    
     # init lr&train&test loss&acc log
     lr_results = []
    
     loss_train_results = []
     accuracy_train_results = []
    
     loss_validation_results = []
     accuracy_validation_results = []
     macro_f1_val_results        = []
    
     loss_test_results = []
     accuracy_test_results = []
     macro_f1_test_results       = []
    
     # prepare optimizer&scheduler&loss_function
     parameters = network.parameters()
     optimizer  = torch.optim.Adam(parameters,lr = LR)
     scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                            patience=5,
                                                            min_lr=LR/10, verbose=True)
     # loss_function = nn.CrossEntropyLoss(reduction='sum')
     loss_function = LabelSmoothingCrossEntropy()
    
     # save init model
     output_directory_init = os.path.join(output_directory_models, 'init_model.pkl')
     torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
     training_duration_logs = []
     start_time = time.time()
     for epoch in range (EPOCH):
        
         for step, (x,y) in enumerate(train_loader):
            
             batch_x = x.to(get_runtime_device())
             batch_y = y.to(get_runtime_device())
             output_bc = network(batch_x)[0]
            
             # cal the sum of pre loss per batch 
             loss = loss_function(output_bc, batch_y)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
        
         # test per epoch
         network.eval()
         # loss_train:loss of training set; accuracy_train:pre acc of training set
         loss_train, accuracy_train, _ = get_test_loss_acc(network, loss_function, train_x, train_y, test_split)        
         loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, loss_function, val_x, val_y, test_split) 
         loss_test, accuracy_test, macro_f1_test = get_test_loss_acc(network, loss_function, X_test, y_test, test_split)
         network.train()  
        
         # update lr
         scheduler.step(accuracy_validation)
         lr = optimizer.param_groups[0]['lr']
        
         # log lr&train&validation loss&acc per epoch
         lr_results.append(lr)
         loss_train_results.append(loss_train)    
         accuracy_train_results.append(accuracy_train)
        
         loss_validation_results.append(loss_validation)    
         accuracy_validation_results.append(accuracy_validation)
         macro_f1_val_results.append(macro_f1_val)
        
         loss_test_results.append(loss_test)    
         accuracy_test_results.append(accuracy_test)
         macro_f1_test_results.append(macro_f1_test)
        
         # print training process
         if (epoch+1) % 1 == 0:
             print('Epoch:', (epoch+1), '|lr:', lr,
                   '| train_loss:', loss_train, 
                   '| train_acc:', accuracy_train, 
                   '| validation_loss:', loss_validation, 
                   '| validation_acc:', accuracy_validation)
        
         save_models(network, output_directory_models, 
                     loss_train, loss_train_results, 
                     accuracy_validation, accuracy_validation_results,
                     start_time, training_duration_logs)
    
     # log training time 
     per_training_duration = time.time() - start_time
     log_training_duration.append(per_training_duration)
    
     # save last_model
     output_directory_last = os.path.join(output_directory_models, 'last_model.pkl')
     torch.save(network.state_dict(), output_directory_last)
    
     # log history
     history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                           loss_validation_results, accuracy_validation_results,
                           loss_test_results, accuracy_test_results,
                           output_directory_models)
    
     plot_learning_history(EPOCH, history, output_directory_models)
    
     return(history, per_training_duration, log_training_duration)