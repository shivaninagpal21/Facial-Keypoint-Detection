## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## The last layer output should have 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
       
        # 1 input image channel (grayscale), 32 output channels/feature maps, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        I.xavier_normal_(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        # dropout layer to avoid overfitting
        self.conv3_drop = nn.Dropout(p=0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        # Another dropout layer to avoid overfitting
        self.conv4_drop = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(256*12*12, 136)
        #self.fc1_drop = nn.Dropout(p=0.4)
        
        #self.fc2 = nn.Linear(50, 136)
        
        

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image
        #x = self.conv1(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_drop(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.conv4_drop(x)
        
        x = x.view(x.size(0), -1)
        
        # 1 linear(Fully connected) layer 
        #x = F.relu(self.fc1(x))
        #x = self.fc1_drop(x)
        x = self.fc1(x)

        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
