# **PyTorch: Training your first Convolutional Neural Network (CNN)**

I’ll then show you the KMNIST dataset (a drop-in replacement for the MNIST digits dataset) that contains Hiragana characters. Later in this tutorial, you’ll learn how to train a CNN to recognize each of the Hiragana characters in the KMNIST dataset.

We’ll then implement three Python scripts with PyTorch, including our CNN architecture, training script, and a final script used to make predictions on input images.

By the end of this tutorial, you’ll be comfortable with the steps required to train a CNN with PyTorch.

## ****Configuring your development environment****

```python
pip install torch torchvision opencv-contrib-python scikit-learn
```

### ****The KMNIST dataset****

![img01.png](/Users/malone/study/python/ml/torch/toy-proj/Convolutional_Neural_Network/imgs/img01.png)

The KMNIST dataset is a drop-in replacement for the standard MNIST dataset. The KMNIST dataset contains examples of handwritten Hiragana characters

The dataset we are using today is the **[Kuzushiji-MNIST dataset](https://github.com/rois-codh/kmnist)**, or KMNIST, for short. This dataset is meant to be a drop-in replacement for the standard MNIST digits recognition dataset.

The KMNIST dataset consists of 70,000 images and their corresponding labels (60,000 for training and 10,000 for testing).

There are a total of 10 classes (meaning 10 Hiragana characters) in the KMNIST dataset, each equally distributed and represented. **Our goal is to train a CNN that can accurately classify each of these 10 characters.**

And lucky for us, the KMNIST dataset is built into PyTorch, making it super easy for us to work with!

# ****Project structure****

```python
$ tree . --dirsfirst
.
├── output
│   ├── model.pth
│   └── plot.png
├── neural_lib
│   ├── __init__.py
│   └── lenet.py
├── predict.py
└── train.py
2 directories, 6 files
```

We have three Python scripts:

1. `lenet.py`: Our PyTorch implementation of the famous LeNet architecture
2. `train.py`: Trains LeNet on the KMNIST dataset using PyTorch, then serializes the trained model to disk (i.e., `model.pth`)
3. `predict.py`: Loads our trained model from disk, makes predictions on testing images, and displays the results on our screen

The `output`directory will be populated with `plot.png` (a plot of our training/validation loss and accuracy) and `model.pth`(our trained model file) once we run train.py.

With our project directory structure reviewed, we can move on to implementing our CNN with PyTorch.

## ****Implementing a Convolutional Neural Network (CNN) with PyTorch****

![img02.png](/Users/malone/study/python/ml/torch/toy-proj/Convolutional_Neural_Network/imgs/img02.png)

The Convolutional Neural Network (CNN) we are implementing here with PyTorch is the seminal **[LeNet architecture](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)**, first proposed by one of the grandfathers of deep learning, Yann LeCunn.

By today’s standards, LeNet is a *very shallow* neural network, consisting of the following layers:

(CONV => RELU => POOL) * 2 => FC => RELU => FC => SOFTMAX

As you’ll see, we’ll be able to implement LeNet with PyTorch in only 60 lines of code (including comments).