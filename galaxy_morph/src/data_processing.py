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
import random

# ============


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

# ==================

# creating the file list

def create_file_list(imgs_path, label_diagram):
    file_list = glob.glob(os.path.join(imgs_path, '*.jpg'))
    file_list = sorted(file_list)

    file_list = [f for f in file_list if int(f.split('/')[-1].split('.')[0]) in label_diagram['asset_id'].values]

    return file_list 

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
    def __init__(self, file_list, data, label_mapping=None, class_mapping=None, aux_layer=None):
        self.file_list = file_list
        self.data = data
        self.label_mapping = label_mapping
        self.aux_layer = aux_layer
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
        
        if self.aux_layer is not None:
            
            aux = np.full((1, H, W), self.aux_layer[idx], dtype='float32')
            img[3] = np.add(img[3], aux, dtype=np.float32)
        else:
            img = img
        
        return torch.tensor(img), torch.tensor(label)

class galaxy_img_dataset_bench(Dataset):
    def __init__(self, file_list, label_mapping, class_mapping=None):
        self.file_list = file_list
        self.label_mapping = label_mapping
        if class_mapping:
            self.class_mapping = class_mapping

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img, asset_id = img_process_bench(self.file_list[idx])
        label = self.label_mapping.get(asset_id, None)
        
        if label is not None and label in self.class_mapping:
            label = self.class_mapping[label]
        else:
            label = 0  # or some default value or raise an error

        return torch.tensor(img), torch.tensor(label)

# =================

# original split of data

def data_setup(file_list, label_diagram, n):
    runs = {f: () for f in file_list}

    for i in label_diagram.columns:
        if i == 'asset_id':
            continue
        # Create a mapping from asset_id to the current label value
        label_map = label_diagram.set_index('asset_id')[i].to_dict()
        # For each file, append the label value to the tuple already stored (or create a new tuple)
        for f in file_list:
            asset_id = int(f.split('/')[-1].split('.')[0]) # select the asset_id
            label_val = label_map.get(asset_id, None) # get the label value
            runs[f] = runs.get(f, ()) + (label_val,) # connect the filename and the label value

    run = [(f, runs[f][0]) for f in file_list]

    images_orig = [x[0] for x in run]
    labels_orig = [x[1] for x in run]
    
    pairs = [(images_orig[x],labels_orig[x]) for x in range(len(images_orig))]

    label0 = [x for x in pairs if x[1]=='E']
    label1 = [x for x in pairs if x[1]=='S']
    label2 = [x for x in pairs if x[1]=='SB']
    label3 = [x for x in pairs if x[1]=='Se']

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

def create_data_loaders(x_train, x_valid, x_test, label_diagram, run_id, bs, aux_train=None, aux_valid=None, aux_test=None):

    class_mapping = {x : i for i, x in enumerate(sorted(label_diagram[f'r{run_id+1}'].unique()))}

    if aux_train is not None and aux_valid is not None and aux_test is not None:
        train = galaxy_img_dataset(x_train, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping, aux_layer=aux_train)
        valid = galaxy_img_dataset(x_valid, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping, aux_layer=aux_valid)
        test = galaxy_img_dataset(x_test, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping, aux_layer=aux_test)
    else:
        train = galaxy_img_dataset(x_train, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping)
        valid = galaxy_img_dataset(x_valid, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping)
        test = galaxy_img_dataset(x_test, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping)

    x_train = torch.stack([x[0] for x in train])
    y_train = torch.tensor([x[1] for x in train]) 

    print(x_train[0],x_train[1])

    x_valid = torch.stack([x[0] for x in valid])    
    y_valid = torch.tensor([x[1] for x in valid])

    x_test = torch.stack([x[0] for x in test])     
    y_test = torch.tensor([x[1] for x in test])

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    return train_dl, valid_dl, test_dl, y_train, y_valid, y_test, class_mapping

def create_data_loaders_bench(x_train, x_valid, x_test, labels_bench, label_mapping, bs):

    class_mapping = {x : i for i, x in enumerate(sorted(set(labels_bench)))}

    train = galaxy_img_dataset_bench(x_train, label_mapping, class_mapping=class_mapping)
    valid = galaxy_img_dataset_bench(x_valid, label_mapping, class_mapping=class_mapping)
    test = galaxy_img_dataset_bench(x_test, label_mapping, class_mapping=class_mapping)

    x_train = torch.stack([x[0] for x in train])
    y_train = torch.tensor([x[1] for x in train]) 

    print(x_train[0],y_train[0])
    print(x_train[1],y_train[1])

    x_valid = torch.stack([x[0] for x in valid])    
    y_valid = torch.tensor([x[1] for x in valid])

    x_test = torch.stack([x[0] for x in test])     
    y_test = torch.tensor([x[1] for x in test])

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    return train_dl, valid_dl, test_dl, class_mapping
