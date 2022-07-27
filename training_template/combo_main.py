# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import os
import math
import argparse
 
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, labels):
        'Initialization'
        self.data = data
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.data[index]
        y = self.labels[index]
        return X, y, index

class LSTMPredictor(nn.Module):

    def __init__(self, look_back):
        super(LSTMPredictor, self).__init__()

        # Nerual Layers
        self.layer_a = nn.Linear(look_back, 32)
        self.relu    = nn.ReLU()
        self.output  = nn.Linear(32,1)

    def predict(self, input):
        with torch.no_grad():
            return self.forward(input)

    def forward(self, input):

        input = input.view(input.shape[0], input.shape[2], -1)
        logits = self.output(self.relu(self.layer_a(input)))
        logits = logits.view(input.shape[0], logits.shape[2], -1)

        return logits


def preprocess(data):
    """
    # (OPTIONAL)
    # Save all the columns to variables
    """
    prices = np.array(
            [ [GSPC, AMC] for Date, TSLA, GSPC, AMC, RIVN, LCID, XPEV, LI, PTRA, F, GM, TWTR, TGT, SVNDY, EMR, GRMN, DHR, NUE, NSANY, TM, HMC in data ]
        ).astype(np.float64)
    return prices


def train_test_split(data, percentage=0.8):
    train_size  = int(len(data) * percentage)
    train, test = data[:train_size], data[train_size:]
    return train, test


def transform_dataset(dataset, look_back=5):
    # N days as training sample
    dataX = [dataset[i:(i + look_back)]
            for i in range(len(dataset)-look_back-1)]
    # 1 day as groundtruth
    dataY = [dataset[i + look_back]
            for i in range(len(dataset)-look_back-1)]
    return torch.tensor(np.array(dataX), dtype=torch.float32), torch.tensor(np.array(dataY), dtype=torch.float32)


def trainer(net, criterion, optimizer, trainloader, devloader, epoch_n=100, path="./checkpoint/save.pt"):
    for epoch in range(epoch_n): # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, data_index = data
            inputs.requires_grad = True
            labels.requires_grad = True

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            train_loss += loss.item()*inputs.shape[0]
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        ######################    
        # validate the model #
        ######################
        net.eval()
        for i, data in enumerate(devloader, 0):
            # move tensors to GPU if CUDA is available
            inputs, labels, data_index = data
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = net(inputs)
            # calculate the batch loss
            loss = criterion(outputs, labels)
            # update average validation loss 
            valid_loss += loss.item()*inputs.shape[0]
        
        # calculate average losses
        train_loss = train_loss/len(trainloader.dataset)
        valid_loss = valid_loss/len(devloader.dataset)
    
        # print training/validation statistics 
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
        
    print('Finished Training')

    ## Save model
    saveModel(net, path)


def tester(net, criterion, testloader):
    loss = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels, data_index = data
            outputs = net(inputs)
            loss += criterion(outputs, labels.unsqueeze(1))
    return loss.item()


def readData(f):
    return np.genfromtxt(f, delimiter='\t', dtype=str)[1:].tolist()


def saveModel(net, path):
    torch.save(net.state_dict(), path)

if __name__ == '__main__':

    # ## Parser initializing
    # parser = argparse.ArgumentParser(description='Train prediction model')
    # parser.add_argument('--ngpu', default=1, type=int, required=False)
    # args   = parser.parse_args()


    ## Device
    # device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    device = torch.device("cpu")


    ## Data
    data = readData("./data/data.csv")
    print('Num of samples:', len(data))


    ## Preprocess
    prices = preprocess(data)
    # Divide trainset and test set
    train, test = train_test_split(prices, 0.8)
    # Set the N(look_back)=5
    look_back = 5
    trainX, trainY = transform_dataset(train, look_back)
    testX, testY   = transform_dataset(test, look_back)
    # Get dataset
    trainset = Dataset(trainX, trainY)
    testset  = Dataset(testX, testY)
    # Get dataloader
    batch_size = 10
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)


    ## Model
    net = LSTMPredictor(look_back)


    ## Loss function
    criterion = nn.MSELoss()


    ## Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001)


    ## Training
    checkpoint = "./checkpoint/save.pt"
    if not os.path.isfile(checkpoint):
        trainer(net, criterion, optimizer, trainloader, testloader, epoch_n=100, path=checkpoint)
    else:
        net.load_state_dict(torch.load(checkpoint))

    ## Test the model
    test = tester(net, criterion, testloader)
    # Show the difference between predict and groundtruth (loss)
    print('Test Result: ', test)

    ## Predict
    test_data = torch.tensor(
            [[[126, 124],[126, 124],[126, 124],[126, 124],[126, 124]]],
            dtype=torch.float32
        )
    print(test_data.shape)
    predict = net.predict(test_data)
    print('Predict Result', predict)