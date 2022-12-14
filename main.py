'''Train Noisy CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision import models

import argparse
import time
import copy
from matplotlib import pyplot as plt

from models.dla import *
from dataset.datasets import *


parser = argparse.ArgumentParser(
    description='PyTorch CIFAR100 Training with Noisy Labels')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=100,
                    type=float, help='number of epochs')
parser.add_argument('--momentum', default=0.9,
                    help='momentum for optimizer')
parser.add_argument('--w_dec', default=5e-4,
                    help='weight decay for optimizer')
parser.add_argument('--model', default='DLA', type=str,
                    choices=['ResNet18', 'ResNet50', 'ResNet101', 'DLA'])
parser.add_argument('--csv_file', help='Path to csv file name')
args = parser.parse_args()
args.num_epochs = int(args.num_epochs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create CIFAR100 dataset loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = C100Dataset(args.csv_file,
                       transform_train, 'train')
testset = C100Dataset(args.csv_file,
                      transform_test, 'val')

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                temp_dataloader = trainloader
                data_size = trainloader.__len__()
            else:
                model.eval()   # Set model to evaluate mode
                temp_dataloader = testloader
                data_size = testloader.__len__()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(temp_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects / total

            if phase == 'train':
                train_loss_arr.append(epoch_loss)
                train_acc_arr.append(epoch_acc)
            else:
                val_loss_arr.append(epoch_loss)
                val_acc_arr.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr


if args.model == 'DLA':
    model_ft = DLA(num_classes=100)
elif args.model in ['ResNet18', 'ResNet50', 'ResNet101']:
    if args.model == 'ResNet18':
        model_ft = models.resnet18(pretrained=True)
    elif args.model == 'ResNet50':
        model_ft = models.resnet50(pretrained=True)
    elif args.model == 'ResNet101':
        model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    num_classes = 100
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

model_ft, train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr = train_model(
    model_ft, criterion, optimizer, scheduler, num_epochs=args.num_epochs)

# Plot loss and accuracy graphs
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
epoch_range = [i for i in range(1, args.num_epochs + 1)]

axs[0].plot(epoch_range, train_loss_arr, c='blue', label='Training loss')
axs[0].plot(epoch_range, val_loss_arr, c='red', label='Validation loss')
axs[0].legend()
axs[0].set_title('Loss curves for Training and Validation')

axs[1].plot(epoch_range, train_acc_arr, c='blue', label='Training Acc')
axs[1].plot(epoch_range, val_acc_arr, c='red', label='Validation Acc')
axs[1].legend()
axs[1].set_title('Accuracy curves for Training and Validation')

# plt.show()
plt.savefig(
    f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "Acc_loss-Train_Val.png"))}')

# Save the model for evaluation
if args.model == 'DLA':
    torch.save(model_ft.state_dict(), os.path.abspath(os.path.join(
        os.path.dirname(__file__), "dla-clean-labels-100-epoch")))
elif args.model in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101']:
    model_scripted = torch.jit.script(model_ft)
    model_scripted.save(
        f'{os.path.abspath(os.path.join(os.path.dirname(__file__), args.model+"-clean-labels-"+str(args.num_epochs)+"-epoch.pt"))}')
