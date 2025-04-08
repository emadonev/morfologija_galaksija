import glob
import pandas as pd
import numpy as np
import os
import PIL as pil
import concurrent.futures
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import torch
import matplotlib.pyplot as plt
import cv2
import random
from collections import Counter

# ==================

W, H = 224, 224

# ==================

# processing of data for data loaders

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

def img_process_bench(entry):
    img = cv2.imread(entry)
    if img is None:
        print(f"Warning: Failed to load image: {entry}")
    img = cv2.resize(img, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_data = img.astype('float32') / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))

    return img_data, int((entry.split("/")[-1]).split('.')[0])


class galaxy_img_dataset(Dataset):
    def __init__(self, file_list, hard_labels, aux_layer=None, soft_labels_dict=None):
        self.file_list = file_list
        self.hard_labels = hard_labels
        self.aux_layer = aux_layer
        self.soft_labels_dict = soft_labels_dict

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img, asset_id = img_process(self.file_list[idx])
        #if self.soft_labels_dict:
            #label = self.soft_labels_dict.get(asset_id, None)
            #if label is None:
                #return None, None
            #label = torch.tensor(label, dtype=torch.float32)
        
        label = self.hard_labels.get(asset_id, None)
        label = torch.tensor(label, dtype=torch.long)
            
        if self.aux_layer is not None:
            
            aux = np.full((1, H, W), self.aux_layer[idx], dtype='float32')
            img[3] = np.add(img[3], aux, dtype=np.float32)
        else:
            img = img
        
        return torch.tensor(img), label

class galaxy_img_dataset_bench(Dataset):
    def __init__(self, file_list, hard_labels, coarse_set=None):
        self.file_list = file_list
        self.hard_labels = hard_labels
        self.coarse_set = coarse_set

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img, asset_id = img_process_bench(self.file_list[idx])
        
        label = self.hard_labels.get(asset_id, None)
        if label is None:
            return None, None
        label = torch.tensor(label, dtype=torch.long)

        if self.coarse_set is not None:
            coarse_label = self.coarse_set[idx]
            if coarse_label is None:
                return None, None
            coarse_label = coarse_label.clone().detach().to(dtype=torch.long)
            return torch.tensor(img, dtype=torch.float32), label, coarse_label

        return torch.tensor(img, dtype=torch.float32), label

# =================

# original split of data

def data_setup(file_list, labels_dict, n):
    runs = {}

    for f in file_list:
        asset_id = f[1]
        label_val = labels_dict.get(asset_id, None) # get the label value
        runs[f[0]] = label_val # connect the filename and the label value

    print(Counter(list(runs.values())))

    images_orig = [x for x in runs]
    labels_orig = [runs[x] for x in runs]
    
    pairs = [(images_orig[x],labels_orig[x]) for x in range(len(images_orig))]

    print(pairs[:4])

    label0 = [x for x in pairs if x[1]==0]
    label1 = [x for x in pairs if x[1]==1]
    label2 = [x for x in pairs if x[1]==2]
    label3 = [x for x in pairs if x[1]==3]
    label4 = [x for x in pairs if x[1]==4]
    label5 = [x for x in pairs if x[1]==5]
    label6 = [x for x in pairs if x[1]==6]

    print(len(label0), len(label1), len(label2), len(label3), len(label4), len(label5), len(label6))

    label0_selection = random.sample(label0, n-500)
    label1_selection = random.sample(label1, n-500)
    label2_selection = random.sample(label2, n-500)
    label3_selection = random.sample(label3, n)
    label4_selection = random.sample(label4, n)
    label5_selection = random.sample(label5, n)
    label6_selection = random.sample(label6, n)

    pairs_rand = label0_selection + label1_selection + label2_selection + label3_selection + label4_selection + label5_selection + label6_selection

    images_orig = [x[0] for x in pairs_rand]
    labels_orig = [x[1] for x in pairs_rand]

    return images_orig, labels_orig

def split_data(x, y):
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.7, random_state=42, 
    stratify=y, 
    shuffle=True)

    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=0.34, random_state=42, 
    stratify=y_rem, 
    shuffle=True)

    print(len(x_train), len(x_valid), len(x_test))

    print(x_train[:5], y_train[:5])

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def create_data_loaders(xt, xv, xte, hard_labels, soft_labels_dict, bs, aux_train=None, aux_valid=None, aux_test=None):

    def get_dataset(x, aux):
        return galaxy_img_dataset(x, hard_labels=hard_labels,aux_layer=aux,soft_labels_dict=None)
    
    train_ds = get_dataset(xt, aux_train)
    valid_ds = get_dataset(xv, aux_valid)
    test_ds  = get_dataset(xte, aux_test)

    y_train, y_valid, y_test = [x[1] for x in train_ds], [x[1] for x in valid_ds], [x[1] for x in test_ds]

    print(train_ds[0][0])
    print(y_train[0])

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=32, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=32, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=True, num_workers=32, pin_memory=True)

    return train_dl, valid_dl, test_dl, y_train, y_valid, y_test

def create_data_loaders_bench(x_train, x_valid, x_test, hard_labels, bs, coarse_train=None, coarse_valid=None):

    def get_dataset(x, coarse_set=None):
        return galaxy_img_dataset_bench(x, hard_labels=hard_labels, coarse_set=coarse_set)
    
    train_ds = get_dataset(x_train, coarse_train)
    valid_ds = get_dataset(x_valid, coarse_valid)
    test_ds  = get_dataset(x_test)

    y_train, y_valid, y_test = [x[1] for x in train_ds], [x[1] for x in valid_ds], [x[1] for x in test_ds]

    print(train_ds[0][0])
    print(y_train[0])

    print(valid_ds[0][0])
    print(y_valid[0])

    print(y_train[:5])
    print(type(y_train))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=32, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=32, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=True, num_workers=32, pin_memory=True)

    return train_dl, valid_dl, test_dl, y_train, y_valid, y_test