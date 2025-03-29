import glob
import pandas as pd
import numpy as np
import os
import PIL as pil
import concurrent.futures
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import cv2

# ----------------------------------------------
# Load data
# ----------------------------------------------
W, H = 224, 224
def choose_class1(x):
    if x[1] == 'B' or x[1] == 'e':
        return x[:2]
    else:
        return x[0]

def choose_class2(x):
    if x[1] == 'B':
        return x[2:3]
    elif x[1] == 'e':
        return 'b'+x[2:3]
    elif x[0] == 'E':
        return x[1:2]+'s'
    else:
        return x[1:2]

def choose_class3(x):
    if x[1] == 'B' or x[1] == 'e':
        if x[3:4] not in '1234+?':
            return '0'
        else:
            return x[3:4]
    else:
        if x[2:3] not in '1234+?':
            return '0'
        else:
            return x[2:3]

def choose_class4(x):
    if x[1] == 'B' or x[1] == 'e':
        if x[4:5] not in 'tml':
            return '0'
        else:
            return x[4:5]+'s'
    else:
        if x[3:4] not in 'tml':
            return '0'
        else:
            return x[3:4]+'s'

def choose_class5(x):
    if x[1] == 'B' or x[1] == 'e':
        return x[5:6]
    else:
        if x[4:5] == '0':
            if x[2:3] in 'rldiomu':
                return x[2:3]
            else:
                return x[5:6]
        else:
            return x[4:5]
        

# ----
def visualize_data(loader, num):
    dataiter = iter(loader)
    images, labels = next(dataiter)

    print("Batch size:", images.size(), labels.size())

    for i in range(num):
        img = images[i].permute(1, 2, 0).numpy()
        img = img[..., ::-1] 

        label = labels[i].item()
        plt.figure(figsize=(3,3))
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

def img_process(entry):
    img = cv2.imread(entry)
    if img is None:
        print(f"Warning: Failed to load image: {entry}")
    img = cv2.resize(img, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_data = img.astype('float32') / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))

    label_layer = np.zeros((1, H, W), dtype='float32')

    return np.vstack([img_data, label_layer]), int((entry.split("/")[-1]).split('.')[0])


class galaxy_img_dataset(Dataset):
    def __init__(self, file_list, data, label_mapping=None, class_mapping=None):
        self.file_list = file_list
        self.data = data
        self.label_mapping = label_mapping
        if class_mapping:
            self.class_mapping = class_mapping
        self.asset_id_to_r = self.data.set_index('asset_id')[self.label_mapping].to_dict()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img, asset_id = img_process(self.file_list[idx])
        label = self.asset_id_to_r.get(asset_id, None)
        
        if label is not None and label in self.class_mapping:
            label = self.class_mapping[label]
        else:
            label = 0  # or some default value or raise an error
        
        return torch.tensor(img), torch.tensor(label)