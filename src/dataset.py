import os

import albumentations as A
from clearml import Task
import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    def __init__(self, config, transforms, ds_type):
        
        self.dishes_df = pd.read_csv(config.DISHES_PATH)
        self.dishes_df = self.dishes_df[self.dishes_df['split'] == ('train' if ds_type == 'train' else 'test')]

        self.ingredients_df = pd.read_csv(config.INGREDIENTS_PATH)
        self.all_ingredients_ids = self.ingredients_df['id'].to_numpy()

        self.transforms = transforms

    def __len__(self):
        return len(self.dishes_df)
    
    def _get_ingredients_ids(self, ingredients):
        return [int(item.split('_')[1]) for item in ingredients.split(';')]

    def __getitem__(self, idx):
        ingredients_ids = self._get_ingredients_ids(self.dishes_df.iloc[idx]['ingredients'])
        table_data = np.isin(self.all_ingredients_ids, ingredients_ids).astype(np.float32)

        img_folder = self.dishes_df.iloc[idx]['dish_id']
        image = Image.open(os.path.join('data/images', img_folder, 'rgb.png'))
        image = self.transforms(image=np.array(image))['image']

        total_mass = self.dishes_df.iloc[idx]['total_mass']

        label = self.dishes_df.iloc[idx]['total_calories']

        return {'image': image, 'table_data': table_data, 'total_mass': total_mass, 'label': label}


def get_transforms(config, ds_type):

    cfg = timm.get_pretrained_cfg(model_name=config.IMAGE_MODEL_NAME)
    task = Task.current_task()

    if ds_type == 'train':
        transforms = A.Compose([
            A.Resize(height=cfg.input_size[1], width=cfg.input_size[2]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=10),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.15, contrast_limit=0.15),
            A.HueSaturationValue(p=0.5, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.ToTensorV2()
        ], seed=42)

        alb_cfg = transforms.to_dict()
    else:
        transforms = A.Compose([
            A.Resize(height=cfg.input_size[1], width=cfg.input_size[2]),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.ToTensorV2()
        ], seed=42)

        alb_cfg = transforms.to_dict()
    
    if task is not None:
        task.connect(alb_cfg, name='albumentations config')

    return transforms


def collate_fn(batch):
    images = [item['image'] for item in batch]
    table_data = [torch.from_numpy(item['table_data']) for item in batch]
    total_masses = [item['total_mass'] for item in batch]
    labels = [item['label'] for item in batch]

    images = torch.stack(images)
    table_data = torch.stack(table_data)

    total_mass_tensor = torch.tensor(total_masses, dtype=torch.float32)
    total_mass_tensor = total_mass_tensor.unsqueeze(1)

    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return {
        'image': images, 
        'table_data': table_data, 
        'total_mass': total_mass_tensor,
        'label': labels_tensor
    }
