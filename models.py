import torch.nn as nn
from utils import *
import torchvision.transforms as tt

#Defining models classes
class Model(ImageClassificationBase):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, xb):
        return self.model(xb)

# Define of models

def Shufflenet(n_classes):
    from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
    model = shufflenet_v2_x0_5()

    for param in model.parameters():
        param.requires_grad_ = False

    model.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes)
    )

    transform = tt.Compose([tt.ToTensor(), ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms(antialias = True)])
    
    return transform, model

def Resnet18(n_classes):
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18()

    for param in model.parameters():
        param.requires_grad_ = False

    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes)
    )

    transform = tt.Compose([tt.ToTensor(), ResNet18_Weights.DEFAULT.transforms(antialias = True)])
    
    return transform, model

def Resnet34(n_classes):
    from torchvision.models import resnet34, ResNet34_Weights
    model = resnet34()

    for param in model.parameters():
        param.requires_grad_ = False

    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes)
    )

    transform = tt.Compose([tt.ToTensor(), ResNet34_Weights.DEFAULT.transforms(antialias = True)])
    
    return transform, model

def Resnet50(n_classes):
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50()

    for param in model.parameters():
        param.requires_grad_ = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes)
    )

    transform = tt.Compose([tt.ToTensor(), ResNet50_Weights.DEFAULT.transforms(antialias = True)])
    
    return transform, model

