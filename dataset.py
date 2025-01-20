import os

import cv2
import numpy as np
import torch
import torch.utils.data
import glob
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        
        mask = []
        for i in range(self.num_classes):

            # print(os.path.join(self.mask_dir, str(i),
            #             img_id + self.mask_ext))
      
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
                        
                        
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        if mask.max()<1:
            mask[mask>0] = 1.0

        return img, mask, {'img_id': img_id}

def get_train_val_test_loader_from_train(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    ## train all labeled data 
    ## fold denote the validation data in training data
    all_paths = glob.glob(f"{data_dir}/*.png")
    # fold_data = get_kfold_data(all_paths, 5)[fold]

    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)
    # random_state = random.random
    random.shuffle(all_paths)

    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:] 

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    train_ds = ImageDataset(train_datalist)
    val_ds = ImageDataset(val_datalist)
    test_ds = ImageDataset(test_datalist)

    loader = [train_ds, val_ds, test_ds]

    return loader
    

def get_train_val_test_indices(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    all_paths = glob.glob(f"{data_dir}/*.png")
    random.seed(seed)
    random.shuffle(all_paths)

    total_number = len(all_paths)
    train_number = int(total_number * train_rate)
    val_number = int(total_number * val_rate)
    test_number = total_number - train_number - val_number  # 确保总和为1

    print(f"test data is {test_number}")
    train_indices = [os.path.splitext(os.path.basename(os.path.abspath(path)))[0] for path in all_paths[:train_number]]
    val_indices = [os.path.splitext(os.path.basename(os.path.abspath(path)))[0] for path in all_paths[train_number:train_number + val_number]]
    test_indices = [os.path.splitext(os.path.basename(os.path.abspath(path)))[0] for path in all_paths[train_number + val_number:]]

    print(f"train data is {train_indices}")
    print(f"validation data is {val_indices}")
    print(f"test data is {test_indices}")



    return train_indices, val_indices, test_indices
   
