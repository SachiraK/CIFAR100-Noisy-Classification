'''Train Noisy CIFAR100 with PyTorch - Create new dataset.'''
import torch

import argparse
import csv

from models.dla import *
from utils import get_csv_data
from views import clean_created_data, initial_data
from dataset.datasets import *


parser = argparse.ArgumentParser(
    description='PyTorch CIFAR100 Training with Noisy Labels')
parser.add_argument('--csv_file', help='Path to csv file name')
parser.add_argument('--thresh', '-r', default=0.5,
                    help='Threshold to filter out noisy data from csv file after clustering')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read initial csv and get data
images_class, images_data, file_names, label_dict = get_csv_data.get_chunks(
    args.csv_file)

# Write class name and value to csv
with open('Classes.csv', 'w') as file:
    writer = csv.writer(file)
    for cls in list(label_dict.keys()):
        writer.writerow([cls, label_dict[cls]])

# Plot distribution of initial and clean data
all_labels = initial_data.plot_graph(images_data, threshold=args.thresh)
clean_files, clean_data = clean_created_data.get_clean_data(
    all_labels, file_names, images_data, args.thresh)

# Write clean data to a new CSV file
with open('NewData.csv', 'w') as file:
    writer = csv.writer(file)
    for cls in list(clean_files.keys()):
        for image in clean_files[cls]:
            writer.writerow([image, cls])
