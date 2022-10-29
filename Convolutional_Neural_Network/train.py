import matplotlib
from neural_lib.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import KMNIST
from torchvision import transforms
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
from typing import Generator

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-m", "--model", type=str, required=True, help="path to output trained model"
)
ap.add_argument(
    "-p", "--plot", type=str, required=True, help="path to output loss/accuracy plot"
)
args = vars(ap.parse_args())

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 30

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# set the device we will be using to train the model
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the KMNIST dataset
print("[INFO] loading the KMNIST dataset ...")
trainData = KMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

testData = KMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(
    trainData,
    [numTrainSamples, numValSamples],
    generator=torch.Generator().manual_seed(42),
)

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(
    trainData,
    shuffle=True,
    batch_size=BATCH_SIZE,
)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# initialize the LeNet model
print("[INFO] initializing the LeNet model...")
model = LeNet(numChannels=1, classes=len(trainData.dataset.classes)).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()


def train(model: LeNet, trainDataLoader: Generator, epochs: int = EPOCHS):
    model.train()
    for epoch in range(epochs):
        # initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0

        # initialize the number of correct predictions in the training
        # and validation step
        train_correct = 0
        val_correct = 0

        for (x, y) in trainDataLoader:
            (x, y) = (x.to(device), y.to(device))

            pred = model(x)
            loss = lossFn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train_loss += loss
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        with torch.no_grad():
            model.eval()

            for (x, y) in valDataLoader:
                (x, y) = (x.to(device), y.to(device))

                pred = model(x)
                total_val_loss += lossFn(pred, y)
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # calcuate the average training and valiation loss
    avg_training_loss = total_train_loss / trainSteps
    avg_val_loss = total_val_loss / valSteps

    # calculate the training and validation accuracy
    train_correct = train_correct / len(trainDataLoader.dataset)
    val_correct = val_correct / len(valDataLoader.dataset)
    # update our training history
    H["train_loss"].append(avg_training_loss.detach().numpy())
    H["train_acc"].append(train_correct)
    H["val_loss"].append(avg_val_loss.detach().numpy())
    H["val_acc"].append(val_correct)


# finish measuring how long training took
endTime = time.time()


def test(model, testDataLoader):
    print(
        "[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime
        )
    )

    print("[INFO] evaluating network...")

    # turn off autograd for testing evaluation
    with torch.no_grad():
        model.eval()
        preds = []

        # loop over the test set
        for (x, y) in testDataLoader:
            # send the input to the device
            x = x.to(device)

            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).detach().numpy())

    # generate a classification report
    print(
        classification_report(
            testData.targets.detach().numpy(),
            np.array(preds),
            target_names=testData.classes,
        )
    )


if __name__ == "__main__":
    train(model, trainDataLoader, EPOCHS)
    test(model, testDataLoader)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

    # serialize the model to disk
    torch.save(model, args["model"])
