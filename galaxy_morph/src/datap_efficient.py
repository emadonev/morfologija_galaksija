import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Global image dimensions
W, H = 224, 224

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
    
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=32, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=32, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=32, pin_memory=True)

    return train_dl, valid_dl, test_dl, y_train, y_valid, y_test

