from collections import OrderedDict
import torch.nn as nn


# OrderedDict: A dictionary object that remembers the order in which objects were added
# — we use this ordered dictionary to provide human-readable names to each layer in the network
# nn: PyTorch’s neural network implementations


def get_training_model(in_features: int = 4, hidden_dim: int = 8, nb_classes: int = 3):
    # construct a shallow, sequential neural network
    mlp_model = nn.Sequential(
        OrderedDict(
            [
                ("hidden_layer_1", nn.Linear(in_features, hidden_dim)),
                ("activation_1", nn.ReLU()),
                ("output_layer", nn.Linear(hidden_dim, nb_classes)),
            ]
        )
    )

    return mlp_model
