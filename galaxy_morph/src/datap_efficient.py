import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
from sklearn.model_selection import train_test_split

# NVIDIA DALI imports
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# Global image dimensions
W, H = 224, 224

# =========
def create_dali_file_list(file_list, hard_labels, output_filename):
    with open(output_filename, 'w') as f:
        for filepath in file_list:
            # Extract asset_id from the filename
            asset_id = int(os.path.splitext(os.path.basename(filepath))[0])
            label = hard_labels.get(asset_id, None)
            if label is None:
                continue  # skip if no label
            f.write(f"{os.path.abspath(filepath)} {label}\n")
    return output_filename

# =======

@pipeline_def
def get_dali_pipeline(file_list, random_shuffle=True):
    # DALI expects a text file where each line is "<image path> <label>"
    images, labels = fn.readers.file(file_list=file_list, random_shuffle=random_shuffle, name="Reader")
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_x=W, resize_y=H)
    # Convert to float32 and normalize to [0,1]
    images = fn.cast(images, dtype=types.FLOAT)
    images = fn.normalize(images, mean=0.0, stddev=255.0)
    # Transpose from NHWC to NCHW format
    images = fn.transpose(images, perm=[2, 0, 1])
    return images, labels

# =======

def create_dali_iterators(x_train, x_valid, x_test, hard_labels, bs, dali_tmp_dir="dali_filelists"):
    # Create a temporary directory to store file lists if it does not exist.
    os.makedirs(dali_tmp_dir, exist_ok=True)
    train_list_file = os.path.join(dali_tmp_dir, "train_list.txt")
    valid_list_file = os.path.join(dali_tmp_dir, "valid_list.txt")
    test_list_file  = os.path.join(dali_tmp_dir, "test_list.txt")
    
    # Write file lists with labels
    create_dali_file_list(x_train, hard_labels, train_list_file)
    create_dali_file_list(x_valid, hard_labels, valid_list_file)
    create_dali_file_list(x_test, hard_labels, test_list_file)
    
    # Create pipelines with proper configuration
    train_pipeline = get_dali_pipeline(
        file_list=train_list_file, 
        random_shuffle=True,
        batch_size=bs,
        num_threads=4,
        device_id=0,
        exec_async=True,
        exec_pipelined=True,
        prefetch_queue_depth=2
    )
    valid_pipeline = get_dali_pipeline(
        file_list=valid_list_file, 
        random_shuffle=False,
        batch_size=bs,
        num_threads=4,
        device_id=0,
        exec_async=True,
        exec_pipelined=True,
        prefetch_queue_depth=2
    )
    test_pipeline = get_dali_pipeline(
        file_list=test_list_file, 
        random_shuffle=False,
        batch_size=bs,
        num_threads=4,
        device_id=0,
        exec_async=True,
        exec_pipelined=True,
        prefetch_queue_depth=2
    )
    
    # Build the pipelines (this allocates resources on GPU)
    train_pipeline.build()
    valid_pipeline.build()
    test_pipeline.build()
    
    # Create DALI generic iterators. Note: the output keys will be 'data' and 'label'
    train_iter = DALIGenericIterator(
        pipelines=[train_pipeline],
        output_map=['data', 'label'],
        reader_name="Reader",
        auto_reset=True
    )
    valid_iter = DALIGenericIterator(
        pipelines=[valid_pipeline],
        output_map=['data', 'label'],
        reader_name="Reader",
        auto_reset=True
    )
    test_iter = DALIGenericIterator(
        pipelines=[test_pipeline],
        output_map=['data', 'label'],
        reader_name="Reader",
        auto_reset=True
    )
    
    return train_iter, valid_iter, test_iter

# =========

def img_process_bench(entry):
    # Read in color explicitly.
    img = cv2.imread(entry, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Failed to load image: {entry}")
    # Resize image first (this may help if the original image is large)
    img = cv2.resize(img, (W, H))
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize the image to float32 and scale to [0,1]
    img_data = img.astype(np.float32) / 255.0
    # Transpose from H x W x C to C x H x W
    img_data = np.transpose(img_data, (2, 0, 1))
    # Extract asset_id using os.path (more robust and possibly faster)
    asset_id = int(os.path.splitext(os.path.basename(entry))[0])
    return img_data, asset_id

class galaxy_img_dataset_bench(Dataset):
    def __init__(self, file_list, hard_labels, coarse_set=None):
        """
        file_list: list of file paths
        hard_labels: dictionary mapping asset_id to label
        coarse_set: list/dictionary of coarse labels indexed by position
        """
        self.file_list = file_list
        self.hard_labels = hard_labels
        self.coarse_set = coarse_set

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Process image and extract asset_id
        img, asset_id = img_process_bench(self.file_list[idx])
        
        # Retrieve the hard label based on asset_id
        label = self.hard_labels.get(asset_id, None)
        if label is None:
            return None, None 
        label = torch.tensor(label, dtype=torch.long)
        
        # If coarse labels are provided, retrieve and convert them
        if self.coarse_set is not None:
            coarse_label = self.coarse_set[idx]
            if coarse_label is None:
                return None, None
            # If coarse_label is not already a tensor, convert it.
            if not isinstance(coarse_label, torch.Tensor):
                coarse_label = torch.tensor(coarse_label, dtype=torch.long)
            else:
                coarse_label = coarse_label.to(dtype=torch.long)
            return torch.from_numpy(img), label, coarse_label

        return torch.from_numpy(img), label

def create_data_loaders_bench(x_train, x_valid, x_test, hard_labels, bs, coarse_train=None, coarse_valid=None):
    def get_dataset(x, coarse_set=None):
        return galaxy_img_dataset_bench(x, hard_labels=hard_labels, coarse_set=coarse_set)
    
    train_ds = get_dataset(x_train, coarse_train)
    valid_ds = get_dataset(x_valid, coarse_valid)
    test_ds  = get_dataset(x_test)

    # Optional: For debugging, get labels from datasets (be mindful this iterates over the dataset)
    y_train = [x[1] for x in train_ds if x[1] is not None]
    y_valid = [x[1] for x in valid_ds if x[1] is not None]
    y_test  = [x[1] for x in test_ds if x[1] is not None]

    print("Sample training image tensor:", train_ds[0][0])
    print("Sample training label:", y_train[0])
    print("Sample validation image tensor:", valid_ds[0][0])
    print("Sample validation label:", y_valid[0])
    
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=16, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=16, pin_memory=True)

    return train_dl, valid_dl, test_dl, y_train, y_valid, y_test

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