import torch
import torch.nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from utils import *
from models import *


#Define path and number of classes

path, n_classes = get_class_path()

#Define the model architecture and the transformations to the dataset. You have to choose the same model in every step

#transform, model = Shufflenet(n_classes)
#transform, model = Resnet34(n_classes)
#transform, model = Resnet18(n_classes)
transform, model = Resnet50(n_classes)


# Getting the dataset
dataset = ImageFolder(root=path, transform=transform)

# Crossvalidation and training set.

val_size = round(0.1*len(dataset)) #Using the 10% of the dataset

train_ds, val_ds = random_split(dataset , [len(dataset) - val_size, val_size])

#Getting the data loaders
batch_size = 32  # Change based in dataset size

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

#Setting the model into the main training device
device = get_default_device()
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
model = to_device(Model(model), device)

#Hyperparameter tuning
epochs = 15
max_lr = 0.0001
grad_clip = 0.007
weight_decay = 1e-5
opt_func = torch.optim.Adam

#Training

history = []

torch.cuda.empty_cache()

history += fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)

#Showing results

plot_accuracies(history)

plot_losses(history)

#Saving the model

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model.pt') # Save