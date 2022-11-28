import numpy as np
from torch import nn
import pandas as pd
import warnings



def main():
    pass   




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



if __name__ == "__main__":
    main()