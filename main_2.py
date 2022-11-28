import numpy as np
from torch import nn
import pandas as pd
import warnings

from data_processing.sliding_window import apply_sliding_window
from data_processing.preprocess_data import load_dataset

warnings.filterwarnings('ignore')

def main():
    gettinDataForTraining()




class DeepConvLSTM(nn.Module):
    def __init__(self, config):
        super(DeepConvLSTM, self).__init__()
        # parameters
        self.window_size = config['window_size']
        self.drop_prob = config['drop_prob']
        self.nb_channels = config['nb_channels']
        self.nb_classes = config['nb_classes']
        self.seed = config['seed']
        self.nb_filters = config['nb_filters']
        self.filter_width = config['filter_width']
        self.nb_units_lstm = config['nb_units_lstm']
        self.nb_layers_lstm = config['nb_layers_lstm']

        # define activation function
        self.relu = nn.ReLU(inplace=True)

        # define conv layers
        self.conv1 = nn.Conv2d(1, self.nb_filters, (self.filter_width, 1))
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1))
        self.conv3 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1))
        self.conv4 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1))
        
        # define lstm layers
        self.lstm = nn.LSTM(input_size=self.nb_filters * self.nb_channels, hidden_size=self.nb_units_lstm, num_layers=self.nb_layers_lstm)

        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        
        # define classifier
        self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)

    def forward(self, x):
        # reshape data for convolutions
        x = x.view(-1, 1, self.window_size, self.nb_channels)
        
        # apply convolution and the activation function
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        # sets the final sequence length 
        final_seq_len = x.shape[2]
        
        # permute dimensions and reshape for LSTM
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, final_seq_len, self.nb_filters * self.nb_channels)

        # apply LSTM (note: it has two outputs!)
        x, _ = self.lstm(x)
            
        # reshape data for classifier
        x = x.view(-1, self.nb_units_lstm)
        
        # apply dropout and feed data through classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        # reshape data and return predicted label of last sample within final sequence (determines label of window)
        out = x.view(-1, final_seq_len, self.nb_classes)
        return out[:, -1, :]


def gettinDataForTraining():
    # data loading (we are using a predefined method called load_dataset, which is part of the DL-ARC feature stack)
    X, y, num_classes, class_names, sampling_rate, has_null = load_dataset('rwhar_3sbjs', include_null=True)
    # since the method returns features and labels separatley, we need to concat them
    data = np.concatenate((X, y[:, None]), axis=1)

    # define the train data to be all data belonging to the first two subjects
    train_data = data[data[:, 0] <= 1]
    # define the validation data to be all data belonging to the third subject
    valid_data = data[data[:, 0] == 2]

    # settings for the sliding window (change them if you want to!)
    sw_length = 50
    sw_unit = 'units'
    sw_overlap = 50

    # apply a sliding window on top of both the train and validation data; you can use our predefined method
    # you can import it via from preprocessing.sliding_window import apply_sliding_window
    X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1], sliding_window_size=sw_length, unit=sw_unit, sampling_rate=50, sliding_window_overlap=sw_overlap)
    X_valid, y_valid = apply_sliding_window(valid_data[:, :-1], valid_data[:, -1], sliding_window_size=sw_length, unit=sw_unit, sampling_rate=50, sliding_window_overlap=sw_overlap)

    print("\nShape of the train and validation datasets after splitting and windowing: ")
    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)

    # (optional) omit the first feature column (subject_identifier) from the train and validation dataset
    X_train, X_valid = X_train[:, :, 1:], X_valid[:, :, 1:]

    print("\nShape of the train and validation feature dataset after splitting and windowing: ")
    print(X_train.shape, X_valid.shape)

    # convert the features of the train and validation to float32 and labels to uint8 for GPU compatibility 
    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.uint8)
    X_valid, y_valid = X_valid.astype(np.float32), y_valid.astype(np.uint8)

if __name__ == "__main__":
    main()