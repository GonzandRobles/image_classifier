#Imports here

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
from workspace_utils import active_session
from PIL import Image
import argparse

####################################

def load_data(data_dir = "/.flowers"):

    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    # Datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size =32,shuffle = True)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = 24, shuffle = True)

    return train_loader, validation_loader, test_loader

####################################

architectures = {'vgg16':25088,
                 'densenet161': 2208,
                 'alexnet':9216
                 }

def model_build(architecture = 'vgg16', dropout = 0.5, hidden_layer=4096, lr = 0.001):

    if architecture == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif architecture == 'densenet161':
        model = models.densenet161(pretrained = True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print('Please select one of these 3 options:\nvgg16 \ndensenet161 \nalexnet ')

    model = models.__dict__[architecture](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(architectures[architecture], hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

    criterion = NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr)

    model.to(device)

    return model, optimizer, criterion, lr

####################################

def validation_pass(model, validation_loader, criterion):

    val_loss = 0
    accuracy = 0

    for images, labels in iter(validation_loader):

        images, labels = images.to(device), labels.to(device)

        logps = model.forward(images)
        val_loss += criterion(logps, labels).item()

        probabilities = torch.exp(logps)

        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return val_loss, accuracy

####################################

def train_model(model, optimizer, criterion, lr, epochs = 12, device = 'cuda'):
    with active_session():
        steps = 0
        print_every = 40
        model.to(device)

        for epoch in range(epochs):
            model.train()
            running_loss = 0

            for inputs, labels in train_loader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:

                    model.eval()

                    with torch.no_grad():
                        validation_loss, accuracy = validation_pass(model, validation_loader, criterion)

                    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_loader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))

                    running_loss = 0
                    model.train()
                    return model

####################################

def test_accuracy(model, test_loader):

    model.eval()
    model.to(device)

    with torch.no_grad():
        accuracy = 0

        for images, labels in iter(test_loader):
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            probabilities = torch.exp(output)
            equality = (labels.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

    print('Accuracy on the test images: %d %%' % (100*(accuracy / len(test_loader))))


####################################

def save_model(checkpoint_path = 'checkpoint.pth ', architecture = 'vgg16'):

    model.class_to_idx = training_dataset.class_to_idx
    model.cpu
    checkpoint = {'architecture': architecture,
            'hidden_layer': 4096,
            'model_state_dict': model.state_dict(),
            'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, checkpoint_path)

####################################

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    architecture = checkpoint['architecture']
    hidden_layer = checkpoint['hidden_layer']
    model,_,_,_ = model_build(architecture, 0.5, hidden_layer, 0.001)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(chkpt['model_state_dict'])
    return model

####################################

def process_image(image):
    img_pil = Image.open(image)

    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = adjustments(img_pil)

    return img_tensor

####################################

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()

    image = process_image(image_path)
    image = image.unsqueeze(0)
    output = model.forward(Variable(image, volatile=True))
    top_prob, top_labels = torch.topk(output, topk)
    top_prob = top_prob.exp()
    top_prob_array = top_prob.data.numpy()[0]

    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}

    top_labels_data = top_labels.data.numpy()
    top_labels_list = top_labels_data[0].tolist()

    top_classes = [inv_class_to_idx[x] for x in top_labels_list]

    return top_prob_array, top_classes
