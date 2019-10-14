import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import argparse

import model_functions

parser = argparse.ArgumentParser(description='Image classifier trainer file')

parser.add_argument('data_dir', nargs='*', action = 'store', default = './flowers', help = 'Data directory, default = "./flowers"')
parser.add_argument('--architecture', dest = 'architecture', type = str, action = 'store', default = 'vgg16', help = '3 options: vgg16, densenet161, alexnet')
parser.add_argument('--hidden_layer', dest = 'hidden_layer', type = int, action = 'store', default = 4096, help = 'Type your hidden_layer number, default = 4096')
parser.add_argument('--learning_rate', dest = 'learning_rate', type = int, action = 'store', default = 0.001, help = 'Type your learning rate, default = 0.001')
parser.add_argument('--gpu', dest = 'gpu', type = str, action = 'store', default = 'cuda', help = 'Select your processor unit, default = "cuda" ')
parser.add_argument('--epochs', dest = 'epochs', type = int, action = 'store', default = 12, help = 'Select number of epochs, default = 12')
parser.add_argument('--save_dir', dest = 'save_dir', type = str, action = 'store', default = 'checkpoint.pth', help = 'Select save path, default "checkpoint.pth"')

parse = parser.parse_args()

architecture = parse.architecture
data_dir = parse.data_dir
hidden_layer = parse.hidden_layer
alpha = parse.learning_rate
processor_unit = parse.gpu
epochs = parse.epochs
chkt_path = parse.save_dir
###################################################################
dataloaders = model_functions.load_data(data_dir)

architecture, hidden_layer = model_functions.model_build(model, classifier, criterion, optimizer)

model_functions.validation_pass(model, validation_loader, criterion)

model_functions.train_model(model, criterion, optimizer, alpha, epochs, processor_unit)

model_functions.test_accuracy(model, test_loader)

model_functions.save_model(chkt_path, architecture)

print("Model training completed...")
