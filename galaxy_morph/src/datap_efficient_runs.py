import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
from sklearn.model_selection import train_test_split
import pickle

# NVIDIA DALI imports
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# Global image dimensions
W, H = 224, 224

# ===========
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

# ==============
def to_one_hot(tensor, num_classes):
    if isinstance(tensor, int):
        # Handle single integer label
        one_hot = torch.zeros(num_classes)
        one_hot[tensor] = 1
        return one_hot
    else:
        # Handle tensor input
        return torch.zeros(tensor.shape[0], num_classes, device=tensor.device).scatter_(1, tensor.unsqueeze(1), 1)

def create_dali_file_list(file_list, hard_labels, output_filename, test=False, previous_coarse=None, save_coarse=False):
    coarse_labels = []
    with open(output_filename, 'w') as f:
        if test and previous_coarse is not None:
            for filepath in file_list:
                # Extract asset_id from the filename
                asset_id = int(os.path.splitext(os.path.basename(filepath))[0])
                label = hard_labels.get(asset_id, None)
                if label is None:
                    continue  # skip if no label
                # Write filepath and label
                f.write(f"{os.path.abspath(filepath)} {label}\n")
                # Store coarse label for saving if requested
                if save_coarse:
                    # Use the previous coarse label directly
                    coarse_labels.append(previous_coarse)
        else:
            for filepath in file_list:
                # Extract asset_id from the filename
                asset_id = int(os.path.splitext(os.path.basename(filepath))[0])
                label = hard_labels.get(asset_id, None)
                if label is None:
                    continue  # skip if no label
                # Write filepath and label
                f.write(f"{os.path.abspath(filepath)} {label}\n")
                # Store coarse label for saving if requested
                if save_coarse:
                    # Use the current label directly
                    coarse_labels.append(label)
    
    # Save coarse labels if requested
    if save_coarse and coarse_labels:
        coarse_labels_path = output_filename.replace('.txt', '_coarse_labels.pkl')
        with open(coarse_labels_path, 'wb') as f:
            pickle.dump(coarse_labels, f)
    
    return output_filename

# =======

class GalaxyIDSource:
    def __init__(self, file_list):
        self.galaxy_ids = []
        for f in file_list:
            # Extract numeric part from filename
            filename = os.path.basename(f)
            # Remove file extension and any non-numeric characters
            numeric_part = ''.join(filter(str.isdigit, os.path.splitext(filename)[0]))
            if numeric_part:  # Only add if we found a numeric part
                self.galaxy_ids.append(int(numeric_part))
            else:
                # If no numeric part found, use a default ID (0)
                self.galaxy_ids.append(0)
        self.total_samples = len(self.galaxy_ids)
    
    def __call__(self, sample_info):
        if sample_info.idx_in_epoch >= self.total_samples:
            return np.array([0], dtype=np.int32)  # Return default ID for out-of-range indices
        return np.array([self.galaxy_ids[sample_info.idx_in_epoch]], dtype=np.int32)

@pipeline_def
def get_dali_pipeline(file_list, random_shuffle=True):
    # Read the file list with labels and coarse labels
    images, labels = fn.readers.file(
        file_list=file_list,
        random_shuffle=random_shuffle,
        name="Reader",
    )
    
    # Process images
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_x=W, resize_y=H)
    images = fn.cast(images, dtype=types.FLOAT)
    images = fn.normalize(images, mean=0.0, stddev=255.0)
    images = fn.transpose(images, perm=[2, 0, 1])
    
    # Convert labels to integers
    labels = fn.cast(labels, dtype=types.INT32)
    
    # Convert coarse labels to float32
    #coarse_labels = fn.cast(coarse_labels, dtype=types.FLOAT32)
    
    # Get galaxy IDs from external source
    galaxy_ids = fn.external_source(
        source=GalaxyIDSource(file_list),
        batch=False,
        dtype=types.INT32,
        num_outputs=1
    )
    
    # Ensure galaxy_ids is a single output
    galaxy_ids = galaxy_ids[0] if isinstance(galaxy_ids, (list, tuple)) else galaxy_ids
    
    return images, labels, galaxy_ids

# =======

def create_dali_iterators(x_train, x_valid, x_test, hard_labels, bs, dali_tmp_dir="dali_filelists", hint=False, previous=None):
    # Create a temporary directory to store file lists if it does not exist.
    os.makedirs(dali_tmp_dir, exist_ok=True)
    train_list_file = os.path.join(dali_tmp_dir, "train_list.txt")
    valid_list_file = os.path.join(dali_tmp_dir, "valid_list.txt")
    test_list_file  = os.path.join(dali_tmp_dir, "test_list.txt")
    
    # Write file lists with labels
    create_dali_file_list(x_train, hard_labels, train_list_file, save_coarse=True)
    create_dali_file_list(x_valid, hard_labels, valid_list_file, save_coarse=True)
    create_dali_file_list(x_test, hard_labels, test_list_file, test=True, previous_coarse=previous, save_coarse=True)
    
    # Load saved coarse labels
    train_coarse_path = train_list_file.replace('.txt', '_coarse_labels.pkl')
    valid_coarse_path = valid_list_file.replace('.txt', '_coarse_labels.pkl')
    test_coarse_path = test_list_file.replace('.txt', '_coarse_labels.pkl')
    
    with open(train_coarse_path, 'rb') as f:
        train_coarse_labels = pickle.load(f)
    with open(valid_coarse_path, 'rb') as f:
        valid_coarse_labels = pickle.load(f)
    with open(test_coarse_path, 'rb') as f:
        test_coarse_labels = pickle.load(f)
    
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
    
    # Build the pipelines
    train_pipeline.build()
    valid_pipeline.build()
    test_pipeline.build()
    
    # Create DALI generic iterators with galaxy IDs
    train_iter = DALIGenericIterator(
        pipelines=[train_pipeline],
        output_map=['data', 'label', 'galaxy_id'],
        reader_name="Reader",
        auto_reset=True
    )
    valid_iter = DALIGenericIterator(
        pipelines=[valid_pipeline],
        output_map=['data', 'label', 'galaxy_id'],
        reader_name="Reader",
        auto_reset=True
    )
    test_iter = DALIGenericIterator(
        pipelines=[test_pipeline],
        output_map=['data', 'label', 'galaxy_id'],
        reader_name="Reader",
        auto_reset=True
    )
    
    return train_iter, valid_iter, test_iter, train_coarse_labels, valid_coarse_labels, test_coarse_labels

# ===========
