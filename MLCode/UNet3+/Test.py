import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import os
from torch import optim 
import matplotlib.pyplot as plt

from layers import unetConv2
from init_weights import init_weights
from unet3 import UNet_3Plus
from myDataLoader import myDataSet

with torch.no_grad():    #reduce memory use
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  #can change according to your server
    net = UNet_3Plus()
    net.to(device)

    #load data
    root_directory = "dataset"
    data_transform = transforms.Compose([transforms.ToTensor()])
    testDataset = myDataSet(root_directory, type="test", transform=data_transform)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=0)

    correct_array = []
    for epoch in range(0, 40):
        PATH = 'model/model_epoch%d' % epoch
        net.load_state_dict(torch.load(PATH))

        avg_correct = 0
        for i, tdata in enumerate(testLoader):
            inputs, labels = tdata
            inputs = inputs.to(device)
            outputs = net(inputs)
            
            #move ouputs back to cpu, and set the value to 0 or 1 in order to calculate accuracy rate
            outputs = outputs.cpu()
            outputs = outputs.detach().numpy()
            outputs = outputs[0][0]
            for i in range(len(outputs)):
                for j in range(len(outputs[i])):
                    if outputs[i][j]>=0.5:
                        outputs[i][j]=1
                    else:
                        outputs[i][j]=0
                        
            labels = labels.numpy()
            labels = labels[0][0]

            #compare per pixel and count correct number
            correct = 0
            for i in range(len(outputs)):
                for j in range(len(outputs[i])):
                    if outputs[i][j]==labels[i][j]:
                        correct = correct+1
            avg_correct += correct/len(outputs)/len(outputs[0])
        #avg_correct is the average of 5 test images
        avg_correct /= len(testLoader)
        print("%d: %f" % (epoch+1, avg_correct))
        correct_array.append(avg_correct)
    
    #at last print all data, the data can be used
    print(correct_array)

