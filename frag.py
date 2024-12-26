import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import os

# ImageNet normalization
mean3 = [0.485, 0.456, 0.406]
mean4 = [0.485, 0.456, 0.406, 0] # Add 0 for the alpha channel
std3 = [0.229, 0.224, 0.225]
std4 = [0.229, 0.224, 0.225, 1] # Add 1 for the alpha channel

# A function for reversing ImageNet normalization
denormalize = T.Normalize(mean=[-m/s for m, s in zip(mean3, std3)], std=[1/s for s in std3])

def train_transform(alpha=False):
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.ToTensor(),
        # ImageNet normalization
        T.Normalize(mean=mean4, std=std4) if alpha else T.Normalize(mean=mean3, std=std3),
    ])
    
def eval_transform(alpha=False):
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNet normalization
        T.Normalize(mean=mean4, std=std4) if alpha else T.Normalize(mean=mean3, std=std3),
    ])


def load_image(path):
    # Assume 'RGBA' and make sure the background is full of 0s (relevant for CLEOPATRA)
    image = Image.open(path)
    rgb = Image.new('RGB', image.size, (0, 0, 0))
    alpha = image.split()[3]
    rgb.paste(image, mask=alpha)
    image = rgb
    # Pad non-square images
    width, height = image.size
    if width != height:
        padding = (0, (width - height) // 2, 0, (width - height) // 2) if width > height else ((height - width) // 2, 0, (height - width) // 2, 0)
        image = T.functional.pad(image, padding)
        alpha = T.functional.pad(alpha, padding)
    image.putalpha(alpha)
    return image


class StyleDataset(Dataset):
    """
    Dataset class for multi-style dataset

    Args:
    -----
    `paths` (list of str):
        list of paths to images
    `labels` (list of int):
        list of labels for each image
    `is_train` (bool):
        whether the dataset is for training (True) or testing (False)
    `n_channels` (int):
        number of channels in the images (3 for RGB, 4 for RGBA)
    """
    
    def __init__(self, paths, labels, is_train):
        self.paths = paths
        unique_labels = np.unique(labels)
        self.n_styles = len(unique_labels)
        if 0 not in unique_labels:
            labels = [l - 1 for l in labels] # 1-indexed to 0-indexed
        self.labels = labels
        self.transform = train_transform() if is_train else eval_transform()

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        rgba = load_image(self.paths[idx])
        rgba = self.transform(rgba)
        image = rgba[:3]
        alpha = (rgba[3] > 0).unsqueeze(0)
        label = torch.tensor(self.labels[idx])
        return image, alpha, label
        

class StyleDataModule(pl.LightningDataModule):
    """
    DataModule class for multi-style dataset

    Args:
    -----
    `train_paths` (list of str):
        list of paths to training images
    `train_labels` (list of int):
        list of labels for each training image
    `val_paths` (list of str):
        list of paths to validation images
    `val_labels` (list of int):
        list of labels for each validation image
    `test_paths` (list of str):
        list of paths to testing images
    `test_labels` (list of int):
        list of labels for each testing image
    """
    
    def __init__(self, train_paths, train_labels, 
                 val_paths, val_labels, 
                 test_paths, test_labels,
                 batch_size, num_workers):
        super().__init__()
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.val_paths = val_paths
        self.val_labels = val_labels
        self.test_paths = test_paths
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            StyleDataset(self.train_paths, self.train_labels, is_train=True),
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            StyleDataset(self.val_paths, self.val_labels, is_train=False),
            batch_size=self.batch_size, num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            StyleDataset(self.test_paths, self.test_labels, is_train=False),
            batch_size=self.batch_size, num_workers=self.num_workers
        )
    

def init_data_module(data_dir, batch_size=32, num_workers=4):
    train_paths = [os.path.join(data_dir, 'train', f) for f in os.listdir(os.path.join(data_dir, 'train'))]
    train_labels = [int(f.split('.')[0]) for f in os.listdir(os.path.join(data_dir, 'train'))]
    val_paths = [os.path.join(data_dir, 'valid', f) for f in os.listdir(os.path.join(data_dir, 'valid'))]
    val_labels = [int(f.split('.')[0]) for f in os.listdir(os.path.join(data_dir, 'valid'))]
    test_paths = [os.path.join(data_dir, 'test', f) for f in os.listdir(os.path.join(data_dir, 'test'))]
    test_labels = [int(f.split('.')[0]) for f in os.listdir(os.path.join(data_dir, 'test'))]
    return StyleDataModule(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        batch_size, num_workers
    )