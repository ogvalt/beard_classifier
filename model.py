import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def get_model(num_classes=2):
    model = resnet50(pretrained=True)

    for name, child in model.named_children():
        if name in ['layer3', 'layer4', 'avgpool', 'fc']:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False

    nb_filters = model.fc.in_features
    model.fc = nn.Sequential(
        Flatten(),
        nn.BatchNorm1d(nb_filters),
        nn.Dropout(),
        nn.Linear(nb_filters, 200),
        nn.ReLU(),
        nn.BatchNorm1d(200),
        nn.Dropout(),
        nn.Linear(200, num_classes)
    )
    return model
