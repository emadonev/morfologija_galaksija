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
        if self.soft_labels_dict:
            label = self.soft_labels_dict.get(asset_id, None)
            if label is None:
                return None, None
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = self.hard_labels.get(asset_id, None)
            if label is None:
                return None, None
            label = torch.tensor(label, dtype=torch.long)
            
        if self.aux_layer is not None:
            
            aux = np.full((1, H, W), self.aux_layer[idx], dtype='float32')
            img[3] = np.add(img[3], aux, dtype=np.float32)
        else:
            img = img
        
        return torch.tensor(img), label

class galaxy_img_dataset_bench(Dataset):
    def __init__(self, file_list, hard_labels, soft_labels_dict=None):
        self.file_list = file_list
        self.hard_labels = hard_labels
        self.soft_labels_dict = soft_labels_dict

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img, asset_id = img_process(self.file_list[idx])
        if self.soft_labels_dict:
            soft_label = self.soft_labels_dict.get(asset_id, None)
            if soft_label is None:
                return None, None
            label = torch.tensor(soft_label, dtype=torch.float32)
        else:
            label = self.hard_labels.get(asset_id, None)
            if label is None:
                return None, None
            label = torch.tensor(label, dtype=torch.long)

        return torch.tensor(img, dtype=torch.float32), label

# =================

# original split of data

def data_setup(file_list, labels_dict, n):
    runs = {f: () for f in file_list}

    for f in file_list:
        asset_id = int(f.split('/')[-1].split('.')[0]) # select the asset_id
        label_val = labels_dict.get(asset_id, None) # get the label value
        runs[f] = runs.get(f, ()) + (label_val,) # connect the filename and the label value

    run = [(f, runs[f][0]) for f in file_list]

    images_orig = [x[0] for x in run]
    labels_orig = [x[1] for x in run]
    
    pairs = [(images_orig[x],labels_orig[x]) for x in range(len(images_orig))]

    label0 = [x for x in pairs if x[1]==0]
    label1 = [x for x in pairs if x[1]==1]
    label2 = [x for x in pairs if x[1]==2]
    label3 = [x for x in pairs if x[1]==3]

    label0_selection = random.sample(label0, n)
    label1_selection = random.sample(label1, n)
    label2_selection = random.sample(label2, n)
    label3_selection = random.sample(label3, n)

    pairs_rand = label0_selection + label1_selection + label2_selection + label3_selection

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

    return x_train, x_valid, x_test

def create_data_loaders(x_train, x_valid, x_test, hard_labels, bs, aux_train=None, aux_valid=None, aux_test=None, soft_labels_dict=None):

    def get_dataset(x, aux, split):
        if split == "train":
            return galaxy_img_dataset(x, hard_labels=hard_labels,aux_layer=aux,soft_labels_dict=soft_labels_dict)
        else:
            return galaxy_img_dataset(x, hard_labels=hard_labels,aux_layer=aux,soft_labels_dict=None)
    
    train = get_dataset(x_train, aux_train, "train")
    valid = get_dataset(x_valid, aux_valid, "valid")
    test  = get_dataset(x_test, aux_test,  "test")

    x_train = torch.stack([x[0] for x in train])
    y_train = torch.stack([x[1] for x in train])

    print(x_train[0],x_train[1])

    x_valid = torch.stack([x[0] for x in valid])    
    y_valid = torch.stack([x[1] for x in valid])


    x_test = torch.stack([x[0] for x in test])     
    y_test = torch.stack([x[1] for x in test])


    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    test_ds = TensorDataset(x_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=16, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=16, pin_memory=True)

    return train_dl, valid_dl, test_dl, y_train, y_valid, y_test

def create_data_loaders_bench(x_train, x_valid, x_test, hard_labels, bs, soft_labels_dict=None):


    train = galaxy_img_dataset_bench(x_train, hard_labels, soft_labels_dict=soft_labels_dict)
    valid = galaxy_img_dataset_bench(x_valid, hard_labels)
    test = galaxy_img_dataset_bench(x_test, hard_labels)

    x_train = torch.stack([x[0] for x in train])
    y_train = torch.tensor([x[1] for x in train]) 

    print(x_train[0],y_train[0])
    print(x_train[1],y_train[1])

    x_valid = torch.stack([x[0] for x in valid])    
    y_valid = torch.tensor([x[1] for x in valid])

    x_test = torch.stack([x[0] for x in test])     
    y_test = torch.tensor([x[1] for x in test])

    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    test_ds = TensorDataset(x_test, y_test)

    train_sampler = DistributedSampler(train_ds) if torch.cuda.device_count() > 1 else None
    valid_sampler = DistributedSampler(valid_ds) if torch.cuda.device_count() > 1 else None
    test_sampler = DistributedSampler(test_ds) if torch.cuda.device_count() > 1 else None

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=(train_sampler is None),
                          num_workers=16, pin_memory=True, sampler=train_sampler)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=16, pin_memory=True, sampler=valid_sampler)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False,
                          num_workers=16, pin_memory=True, sampler=test_sampler)

    return train_dl, valid_dl, test_dl
