import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision import models

from models.dla import *
from dataset.datasets import *


def main():
    # argument parsing
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR100 Training with Noisy Labels')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--w_dec', default=5e-4,
                        help='weight decay for optimizer')
    parser.add_argument('--csv_file', help='Path to csv file name')
    parser.add_argument('--thresh', '-r', default=0.5,
                        help='Threshold to filter out noisy data from csv file after clustering')
    parser.add_argument('--model', default='DLA', type=str,
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'DLA'])
    parser.add_argument('--load_path', type=str,
                        default='saved_models/dla/dla-clean-labels-100epoch')
    args = parser.parse_args()

    torch.cuda.set_device(device)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    testset = C100TestDataset(args.csv_file,
                              transform_test)
    testloader = DataLoader(testset, batch_size=8, shuffle=False)

    # Load model
    if args.model == 'DLA':
        model_ft = DLA(num_classes=100)
        model_ft.load_state_dict(torch.load(args.load_path))
    elif args.model in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101']:
        model_ft = torch.jit.load(args.load_path)

    model_ft = model_ft.to(device)
    model_ft.eval()

    criterion = nn.CrossEntropyLoss()
    test(model_ft, testloader, criterion)


def test(net, test_dataloader, criterion):
    net.eval()
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            acc += sum(outputs.argmax(dim=1) == targets)
    acc = acc/test_dataloader.dataset.__len__()
    print('Accuracy :' + '%0.4f' % acc)
    return acc


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
