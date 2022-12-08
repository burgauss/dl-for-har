import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import pandas as pd
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from misc.torchutils import seed_torch
import time
from sklearn.model_selection import train_test_split
from data_processing.sliding_window import apply_sliding_window
from data_processing.preprocess_data import load_dataset

warnings.filterwarnings('ignore')

pathAll = "C:/Users/juan.burgos/Desktop/JuanBurgos/04 Thesis/12_DataCollection/TrainSets/Combination/WeightLog_ALL.csv"

def main():
    dataLoader = Dataloader(pathAll)
    dataset, waveIndexBegin, waveIndexEnding= dataLoader.processData()
    print(dataset["Bottle"].unique())

    train_test_split(dataset, 0.2, )


class Dataloader():
    """Following class has as responsabilities to get the data coming from the dataset.
    moreover to follow a series of preprocessing like normalization"""
    def __init__(self, path):
        #self.path = path
        self.dataset = pd.read_csv(path, delimiter=";", header = None)
        #self.dataset = self.dataset.drop(columns=[2,3,4,5,6,7,8], axis=1)
        
        #print(self.dataset.head(10))
    def processData(self):
        """ Method process the data according to the definition set in the WeightOfflineCalculation
        Args: None
        Returns: a pd.df with the processed data
        """
        processedDataset = self.dataset.copy()
        processedDataset[10] = processedDataset[0].diff()
        waveIndexBegin = processedDataset.index[processedDataset[10] == -1].to_numpy()  
        waveIndexEnding = processedDataset.index[processedDataset[10] == 2].to_numpy()
        # print("Windows start at" + str(waveIndexBegin)) # To print the actual index
        # print("Windows end at" + str(waveIndexEnding))
        print("Len of of index for start of Wave " + str(len(waveIndexBegin)))
        print("Len of of index for end of Wave " + str(len(waveIndexEnding)))
        ######################################################################
        # Droping the datasets
        processedDataset = processedDataset.drop(columns=[2,3,4,5,6,7,8], index=1)
        
        #Asssingning names to columns
        processedDataset.columns = ['Satus', 'Data', 'Bottle', 'Diff']
        
        # General information
        processedDataset.info()
        print(processedDataset.describe())

        return processedDataset, waveIndexBegin, waveIndexEnding

def getWindowedData(dataset, tStepLeftShift=0, tStepRightShift=0, expectedWaves=None):
    """Function determines the size of the windows for the dataset
    param dataset: pd.Dataframe
        the dataset
    param tSstepLeftShift: int
        the number of data points that to the left of LS1ON that will be taken into account
    param tStepRightShift: int
        the number of data points that to the right of LS1ON that will be taken into account
    param expectedWaves: int or list
        the number of waves per sequence, i.e., Bottle ColaHalb has 51 waves
    return: np.array with the X_train, y_train, X_test, y_test"""

    num_classes = dataset["Bottle"].unique()
    npDataSet = np.array(dataset)
    batchedDataSet = []

    for i in range(expectedWaves):
        batchedDataSet.append()
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

    return X_train, y_train, X_valid, y_valid, num_classes, class_names, sampling_rate, has_null

def test1DLForHAR():
    #Program based on the implementation from Marius Bock, in the training iypb
    X_train, y_train, X_valid, y_valid, num_classes, class_names, sampling_rate, has_null = gettinDataForTraining()
    
    config = {
    'nb_filters': 64,
    'filter_width': 11,
    'nb_units_lstm': 128,
    'nb_layers_lstm': 1,
    'drop_prob': 0.5,
    'seed': 1,
    'epochs': 20,
    'batch_size': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-6,
    'gpu_name': 'cuda:0',
    'print_counts': False
    }

    # in order to get reproducible results, we need to seed torch and other random parts of our implementation
    seed_torch(config['seed'])


    # define the missing parameters within the config file. 
    # window_size = size of the sliding window in units
    # nb_channels = number of feature channels
    # nb_classes = number of classes that can be predicted
    config['window_size'] = X_train.shape[1]
    config['nb_channels'] = X_train.shape[2]
    config['nb_classes'] = len(class_names)

    # initialize your DeepConvLSTM object 
    network = DeepConvLSTM(config)

    # sends network to the GPU and sets it to training mode
    network.to(config['gpu_name'])
    network.train()


    # initialize the optimizer and loss
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # initializes the train and validation dataset in Torch format
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
        
    # define the train- and valloader; use from torch.utils.data import DataLoader
    trainloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)    
    valloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    # define your training loop; iterates over the number of epochs
    for e in range(config['epochs']):
        # helper objects needed for proper documentation
        train_losses = []
        train_preds = []
        train_gt = []
        start_time = time.time()
        batch_num = 1

        # iterate over the trainloader object (it'll return batches which you can use)
        for i, (x, y) in enumerate(trainloader):
            # sends batch x and y to the GPU
            inputs, targets = x.to(config['gpu_name']), y.to(config['gpu_name'])
            optimizer.zero_grad()

            # send inputs through network to get predictions
            train_output = network(inputs)

            # calculates loss
            loss = criterion(train_output, targets.long())

            # backprogate your computed loss through the network
            # use the .backward() and .step() function on your loss and optimizer
            loss.backward()
            optimizer.step()

            # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax()
            train_output = torch.nn.functional.softmax(train_output, dim=1)

            # appends the computed batch loss to list
            train_losses.append(loss.item())

            # creates predictions and true labels; appends them to the final lists
            y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
            train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))

            # prints out every 100 batches information about the current loss and time per batch
            if batch_num % 100 == 0 and batch_num > 0:
                cur_loss = np.mean(train_losses)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | train loss {:5.2f}'.format(e, batch_num, elapsed * 1000 / config['batch_size'], cur_loss))
                start_time = time.time()
                batch_num += 1
            
            #print(i)

        # helper objects
        val_preds = []
        val_gt = []
        val_losses = []

        # sets network to eval mode and 
        network.eval()
        with torch.no_grad():
            # iterate over the valloader object (it'll return batches which you can use)
            for i, (x, y) in enumerate(valloader):
                # sends batch x and y to the GPU
                inputs, targets = x.to(config['gpu_name']), y.to(config['gpu_name'])

                # send inputs through network to get predictions
                val_output = network(inputs)

                # calculates loss by passing criterion both predictions and true labels 
                val_loss = criterion(val_output, targets.long())

                # calculate actual predictions (i.e. softmax probabilites); use torch.nn.functional.softmax() on dim=1
                val_output = torch.nn.functional.softmax(val_output, dim=1)

                # appends validation loss to list
                val_losses.append(val_loss.item())

                # creates predictions and true labels; appends them to the final lists
                y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
                y_true = targets.cpu().numpy().flatten()
                val_preds = np.concatenate((np.array(val_preds, int), np.array(y_preds, int)))
                val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))

            # print epoch evaluation results for train and validation dataset
            print("\nEPOCH: {}/{}".format(e + 1, config['epochs']),
                    "\nTrain Loss: {:.4f}".format(np.mean(train_losses)),
                    "Train Acc: {:.4f}".format(jaccard_score(train_gt, train_preds, average='macro')),
                    "Train Prec: {:.4f}".format(precision_score(train_gt, train_preds, average='macro')),
                    "Train Rcll: {:.4f}".format(recall_score(train_gt, train_preds, average='macro')),
                    "Train F1: {:.4f}".format(f1_score(train_gt, train_preds, average='macro')),
                    "\nVal Loss: {:.4f}".format(np.mean(val_losses)),
                    "Val Acc: {:.4f}".format(jaccard_score(val_gt, val_preds, average='macro')),
                    "Val Prec: {:.4f}".format(precision_score(val_gt, val_preds, average='macro')),
                    "Val Rcll: {:.4f}".format(recall_score(val_gt, val_preds, average='macro')),
                    "Val F1: {:.4f}".format(f1_score(val_gt, val_preds, average='macro')))

            # if chosen, print the value counts of the predicted labels for train and validation dataset
            if config['print_counts']:
                print('Predicted Train Labels: ')
                print(np.vstack((np.nonzero(np.bincount(train_preds))[0], np.bincount(train_preds)[np.nonzero(np.bincount(train_preds))[0]])).T)
                print('Predicted Val Labels: ')
                print(np.vstack((np.nonzero(np.bincount(val_preds))[0], np.bincount(val_preds)[np.nonzero(np.bincount(val_preds))[0]])).T)


        # set network to train mode again
        network.train()


if __name__ == "__main__":
    main()