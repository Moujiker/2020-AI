import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import os
from skimage import io

from layers import unetConv2
from init_weights import init_weights
from unet3 import UNet_3Plus,UNet_3Plus_DeepSup
from myDataLoader import myDataSet

#can only set to 1, otherwise the GPU memory may not be enough
BATCH_SIZE=1

#load train data
data_transform = transforms.Compose([transforms.ToTensor()]) #just ToTensor(), no other transforms
root_directory = "dataset"
#the class myDataSet is in myDataLoader.py, defined by myself
trainDataset = myDataSet(root_directory, type="train", transform=data_transform)
trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

#change the device according to your server
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
net = UNet_3Plus_DeepSup()
net.to(device)

#loss function and optimizer
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)

#start train
for epoch in range(0, 40):

    running_loss = 0.0
    for i, tdata in enumerate(trainLoader):
        inputs, labels = tdata
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, torch.squeeze(labels).long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i*BATCH_SIZE + 1, running_loss))
        running_loss = 0.0

    #save every epoch
    PATH = 'model/model_epoch%d' % epoch
    torch.save(net.state_dict(), PATH)







