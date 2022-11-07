import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import csv
import os


class C100Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, d_type='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            d_type (string): Specifies whether it is training or validating
        """
        self.all_images = []
        self.all_labels = []
        self.label_dict = {}

        self.unique_labels = []
        with open(csv_file, newline='') as csvfile:
            imagedata = csv.reader(csvfile, delimiter=',')
            for row in imagedata:
                img_name = row[0]
                category = img_name.split('/')[2]
                if category == 'train':
                    label = row[1]
                    self.all_images.append(img_name)
                    if label not in self.unique_labels:
                        self.label_dict[label] = len(self.unique_labels)
                        self.unique_labels.append(label)
                    self.all_labels.append(self.label_dict[label])

        # Shuffle data
        combined = list(zip(self.all_images, self.all_labels))
        random.shuffle(combined)
        self.all_images, self.all_labels = zip(*combined)

        # Divide to Train and Val sets
        ratio = 0.9
        limit = int(len(self.all_images) * ratio)

        if d_type == 'train':
            self.all_images = self.all_images[:limit]
            self.all_labels = self.all_labels[:limit]
        else:
            self.all_images = self.all_images[limit:]
            self.all_labels = self.all_labels[limit:]

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.all_images[idx])
        image = Image.open(img_name)
        # image = image.numpy().transpose((2, 0, 1))
        label = self.all_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class C100TestDataset(C100Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_images = []
        self.all_labels = []
        self.label_dict = {}

        # Read pre-defined classes from csv file
        with open('Classes.csv', newline='') as classfile:
            classdata = csv.reader(classfile, delimiter=',')
            for row in classdata:
                self.label_dict[row[0]] = int(row[1])

        self.unique_labels = []
        with open(csv_file, newline='') as csvfile:
            imagedata = csv.reader(csvfile, delimiter=',')
            for row in imagedata:
                img_name = row[0]
                label = row[1]

                self.all_images.append(img_name)
                self.all_labels.append(self.label_dict[label])

        self.root_dir = root_dir
        self.transform = transform
