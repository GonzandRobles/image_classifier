import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import json
from workspace_utils import active_session
from PIL import Image

import argparse
import model_functions

parser = argparse.ArgumentParser(description='Image classifier prediction file')

parser.add_argument('img', nargs = '*', type = str, default = 'aind-project/flowers/test/1/image_06743.jps', help = 'Type an image path')
parser.add_argument('checkpoint', nargs='*', action="store", type = str, default="checkpoint.pth")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--top_k', default = 5, dest="top_k", action="store", type = int)
parser.add_argument('--category_names', dest= "category_names", action="store", default='cat_to_name.json')

parse = parser.parse_args()
outputs = parse.top_k
processor = parse.gpu
input_img = parse.img
chkt_path = parse.checkpoint

model_load = model_functions.load_checkpoint(chkt_path)

model_functions.load_checkpoint(filepath)

# cpu
device = torch.device("cpu")
# gpu
if parse.gpu:
 device = torch.device("cuda:0")


with open(category_names) as json_file:
    cat_to_name = json.load(json_file)

top_probabilities, top_classes = model_functions.predict(img_path, model, outputs)

print(top_probabilities)
print(top_classes)
