from PIL import Image
import csv
import numpy as np
from torchvision import transforms
import os


def train_transform(image):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(image).numpy()


def get_chunks(csvfilename):
    unique_labels = []
    label_dict = {}
    images_class = {}
    images_data = {}
    file_names = {}

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset', csvfilename)), newline='') as csvfile:
        imagedata = csv.reader(csvfile, delimiter=',')
        for row in imagedata:
            img_name = row[0]
            category = img_name.split('/')[2]
            if category == 'train':
                label = row[1]
                if label not in unique_labels:
                    label_dict[label] = len(unique_labels)
                    images_class[label] = []
                    unique_labels.append(label)
                images_class[label].append(img_name)

    for cls in list(images_class.keys()):
        print(f'Class name: {cls} length: {len(images_class[cls])}')
        temp_arr = []
        file_arr = []
        for img in images_class[cls]:
            file_arr.append(img)
            image = np.array(Image.open(os.path.abspath(os.path.join(
                os.path.dirname(__file__), '../dataset', img))))
            temp_arr.append(train_transform(image))
        images_data[cls] = np.array(temp_arr).reshape(len(temp_arr), -1)
        file_names[cls] = file_arr
        print(f'Shape of each class: {images_data[cls].shape} \n')

    return images_class, images_data, file_names, label_dict
