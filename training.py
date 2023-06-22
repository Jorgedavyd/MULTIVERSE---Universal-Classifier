import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from utils import *
import os
# Define the transformations to be applied to the data

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
stats = (mean, std)
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])
#Getting the path
path = os.path.join(os.getcwd(), 'data')

# Getting the dataset
dataset = ImageFolder(root=path, transform=transform)

n_classes = len(dataset.classes)

# crossvalidation, and train set.

val_size = round(0.1*len(dataset))

train_ds, val_ds = random_split(dataset , [len(dataset) - val_size, val_size])

batch_size = 32  # Change based in dataset size

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)

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

class Model(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.model = model

    def forward(self, xb):
        out = self.model(xb)
        return out

device = get_default_device()
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
model = to_device(Model(), device)


epochs = 15
max_lr = 0.0001
grad_clip = 0.007
weight_decay = 1e-5
opt_func = torch.optim.Adam

history= []

torch.cuda.empty_cache()
history += fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)

torch.save(model.state_dict(), 'model.pt')