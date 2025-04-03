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

def create_file_list(imgs_path, label_diagram):
    file_list = glob.glob(os.path.join(imgs_path, '*.jpg'))
    file_list = sorted(file_list)

    file_list = [f for f in file_list if int(f.split('/')[-1].split('.')[0]) in label_diagram['asset_id'].values]

    return file_list        

# ----
def visualize_data(loader, num):
    dataiter = iter(loader)
    images, labels = next(dataiter)

    print("Batch size:", images.size(), labels.size())

    for i in range(num):
        img = images[i].permute(1, 2, 0).numpy()
        img = img[:3]

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

def img_process_bench(entry):
    img = cv2.imread(entry)
    if img is None:
        print(f"Warning: Failed to load image: {entry}")
    img = cv2.resize(img, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_data = img.astype('float32') / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))    

    return img_data

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
            img[3].fill(self.aux_layer[idx])
        else:
            img = img
        
        return torch.tensor(img), torch.tensor(label)

class galaxy_img_dataset_bench(Dataset):
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
        img, asset_id = img_process_bench(self.file_list[idx])
        label = self.asset_id_to_r.get(asset_id, None)
        
        if label is not None and label in self.class_mapping:
            label = self.class_mapping[label]
        else:
            label = 0  # or some default value or raise an error
        
        return torch.tensor(img), torch.tensor(label)

def data_setup_run(img_to_label, file_list, label_diagram, run_id, n, bs, aux=None):
    run = [(f, img_to_label[f][run_id]) for f in file_list]
    images_orig = [x[0] for x in run]
    labels_orig = [x[1] for x in run]
    print(len(images_orig), len(labels_orig))
    print(labels_orig[:2])

    img_sub, im, labels_sub, l = train_test_split(images_orig, labels_orig, train_size=n, random_state=42, stratify=labels_orig, shuffle=True)
    print(len(img_sub), len(labels_sub))
    print(img_sub[:5], labels_sub[:5])

    class_mapping = {x : i for i, x in enumerate(sorted(label_diagram[f'r{run_id+1}'].unique()))}

    if aux is not None:
        gmorph_d = galaxy_img_dataset(img_sub, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping, aux_layer=aux)
    else:
        gmorph_d = galaxy_img_dataset(img_sub, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping)
    
    images = [x[0] for x in gmorph_d]
    labels = [x[1] for x in gmorph_d]

    x = np.array(images)
    y = np.array(labels)

    print(x[:1], y[:1])

    # ---------
    print("preparing the data loaders")

    x_train, x_rem, y_train, y_rem = train_test_split(x,y, train_size=0.7, random_state=42, 
    #stratify=y, 
    shuffle=True)

    x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.34, random_state=42, 
    #stratify=y_rem, 
    shuffle=True)

    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test))

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    return class_mapping, labels_orig, train_dl, valid_dl, test_dl


def data_setup_run_benchmark(img_to_label, file_list, label_diagram, run_id, n, bs, aux=None):
    run = [(f, img_to_label[f][run_id]) for f in file_list]
    images_orig = [x[0] for x in run]
    labels_orig = [x[1] for x in run]
    print(len(images_orig), len(labels_orig))
    print(labels_orig[:2])

    img_sub, im, labels_sub, l = train_test_split(images_orig, labels_orig, train_size=n, random_state=42, stratify=labels_orig, shuffle=True)
    print(len(img_sub), len(labels_sub))
    print(img_sub[:5], labels_sub[:5])

    class_mapping = {x : i for i, x in enumerate(sorted(label_diagram[f'r{run_id+1}'].unique()))}

    gmorph_d = galaxy_img_dataset_bench(img_sub, label_diagram, label_mapping=f'r{run_id+1}', class_mapping=class_mapping)
    
    images = [x[0] for x in gmorph_d]
    labels = [x[1] for x in gmorph_d]

    x = np.array(images)
    y = np.array(labels)

    print(x[:1], y[:1])

    # ---------
    print("preparing the data loaders")

    x_train, x_rem, y_train, y_rem = train_test_split(x,y, train_size=0.7, random_state=42, stratify=y, shuffle=True)

    x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.34, random_state=42, stratify=y_rem, shuffle=True)

    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test))

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)

    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=16,pin_memory=True)

    return class_mapping, labels_orig, train_dl, valid_dl, test_dl