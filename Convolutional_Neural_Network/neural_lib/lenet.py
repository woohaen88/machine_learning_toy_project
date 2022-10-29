from torch import nn, flatten
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, LogSoftmax


class LeNet(nn.Module):
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()

        self.conv1 = Conv2d(
            in_channels=numChannels,
            out_channels=20,
            kernel_size=(5, 5),
        )
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv2d(
            in_channels=20,
            out_channels=50,
            kernel_size=(5, 5),
        )
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = Linear(
            in_features=800,
            out_features=500,
        )
        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        # flatten the tensor without first dimension
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)

        # return the output predictions
        return output
