from MLP.neural_lib import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch
from typing_extensions import Final

# from torch import Tensor


def next_batch(inputs, targets, batch_size):
    for i in range(0, inputs.shape[0], batch_size):  # range(start, stop, step)
        yield inputs[i : i + batch_size], targets[i : i + batch_size]


BATCH_SIZE: Final = 64
EPOCHS: Final = 10
LR: Final = 1e-2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

# generate a 3-class classification problem with 1000 data points,
# where each data point is a 4D feature vector
print("[INFO] preparing data...")
(x_data, y_data) = make_blobs(
    n_samples=1000,
    n_features=4,
    centers=3,
    cluster_std=2.5,
    random_state=95,
)

(train_x, test_x, train_y, test_y) = train_test_split(
    x_data,
    y_data,
    test_size=0.15,
    random_state=95,
)

train_x = torch.from_numpy(train_x).float()
test_x = torch.from_numpy(test_x).float()
train_y = torch.from_numpy(train_y).float()
test_y = torch.from_numpy(test_y).float()

# initialize our model and display its architecture
mlp = mlp.get_training_model().to(DEVICE)
print(mlp)
# initialize optimizer and loss function
opt = SGD(mlp.parameters(), lr=LR)
lossFunc = nn.CrossEntropyLoss()

# create a template to summarize current training progress
testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"

for epoch in range(0, EPOCHS):
    print(f"[INFO] epoch: {epoch+1}...")
    train_loss: float = 0
    train_acc: float = 0
    samples: float = 0
    mlp.train()

    for (batch_x, batch_y) in next_batch(train_x, train_y, BATCH_SIZE):
        (batch_x, batch_y) = (batch_x.to(DEVICE), batch_y.to(DEVICE))
        predictions = mlp(batch_x)
        loss = lossFunc(predictions, batch_y.long())

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item() * batch_y.size(0)
        train_acc += (predictions.max(1)[1] == batch_y).sum().item()
        samples += batch_y.size(0)

    trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
    print(
        trainTemplate.format(
            epoch + 1,
            (train_loss / samples),
            (train_acc / samples),
        )
    )

    # initialize tracker variables for testing, then set our model to
    # evaluation mode
    test_loss: float = 0
    test_acc: float = 0
    samples: float = 0
    mlp.eval()

    # initialize a no-gradient context
    with torch.no_grad():
        # loop over the current batch of test data
        for (batch_x, batch_y) in next_batch(test_x, test_y, BATCH_SIZE):
            # flash the data to the current device
            (batch_x, batch_y) = (batch_x.to(DEVICE), batch_y.to(DEVICE))

            # run data through our model and calculate loss
            predictions = mlp(batch_x)
            loss = lossFunc(predictions, batch_y.long())

            # update test loss, accuracy, and the number of samples visited
            test_loss += loss.item() * batch_y.size(0)
            test_acc += (predictions.max(1)[1] == batch_y).sum().item()
            samples += batch_y.size(0)

        # display template progress on the current test batch
        print(
            testTemplate.format(
                epoch + 1,
                (test_loss / samples),
                (test_acc / samples),
            )
        )
        print("")
