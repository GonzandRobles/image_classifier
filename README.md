# Flower Classifier App

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Installation
___

## Environment requirements
* Create a conda environment with python=3.6 and the following packages.
  * conda create --name myenv python=3.6 numpy cython jupyter nb_conda_kernels scipy matplotlib
  * source activate myenv
* Go to [PyTorch](https://pytorch.org/) website and follow directions to install PyTorch for your system.

## Clone requirements
* If not installed please install [Git LFS](https://git-lfs.github.com/).
* Clone repository
  * git clone [here](https://github.com/GonzandRobles/image_classifier.git)

## Usage
___

### Two ways to use the app

* Predict flower types using checkpoint.pth included in the repository.
  * Doesn't require downloading the Dataset.
  * Doesn't require using a GPU to train.
  * Test photos included in repository under test-photos directory.
  
* Train classifier yourself.
  * GPU is highly recommended to train in a reasonable time.
  * Downloading training data is required, see "Download data" section.
  
### Predict
* Run predict.py script
  * python predict.py -h
    * Usage help
  * python predict.py image_path checkpoint_path
    * Basic usage gives numeric label of flower in image.
  * python predict.py image_path checkpoint_path --category_names cat_to_name.json
    * Gives name of flower in image.
  * python predict.py image_path checkpoint_path --top_k 5
    * List top 5 most likely numeric labels
  * Use --gpu if you are using a GPU

### Train

**Download data**
* Download data into main repository from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
  * Download "Dataset images" and "The image labels"
  * Un-tar 102flowers.tar which should create a jpg directory.
  * At this point you should have "jpg" directory with 8189 jpgs in it and a imagelabels.mat file.
  
**Make Training, Validation and Testing sets**
* Run setup.py script.
  * python setup.py -h
    * Usage help
* python setup.py jpg imagelabels.mat
  * Creates flowers directory with train, valid and test subdirectories.
    * Each subdirectory has subdirectories labeled 1-102 with jpg files corresponding to 102 flower labels.
      * Note that cat_to_name.json has mappings of numeric labels to actual names of flowers.

**Train and create checkpoint**
* Run train.py script
  * python train.py -h
    * Usage help
  * python train.py flowers
    * Basic usage trains classifier on top of "vgg19" architecture and prints out training loss and validation accuracy.
  * Use --save_dir SAVE_DIR to save a checkpoint after training.
  * Explore usage help (-h) to see how to adjust other parameters, including:
    * --arch to change architecture
    * --learning_rate to adjust learning rate
    * --hidden_units to pick number of hidden units in Classifier
    * --dropout to adjust dropout
    * --epochs to pick number of epochs
    * --gpu if you are training on a GPU
