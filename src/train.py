import os
import random

from clearml import Task
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics

from src.dataset import collate_fn, get_transforms, MultimodalDataset
from src.model import MultimodalModel, set_requires_grad


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def train(config, device):

    task = Task.current_task()
    logger = task.get_logger()

    train_dataset = MultimodalDataset(config=config,
                                  transforms=get_transforms(config=config, ds_type='train'),
                                  ds_type='train')
    val_dataset = MultimodalDataset(config=config,
                                    transforms=get_transforms(config=config, ds_type='test'),
                                    ds_type='test')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            collate_fn=collate_fn)
    

    model = MultimodalModel(config=config).to(device)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE)

    optimizer = AdamW([
        {'params': model.image_model.parameters(),
         'lr': config.IMAGE_LR},
         {'params': model.mlp.parameters(),
         'lr': config.MLP_LR},
         {'params': model.regressor.parameters(),
         'lr': config.REGRESSOR_LR}
    ])
    criterion = nn.L1Loss(reduction='mean')

    mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_metric_val = torchmetrics.MeanAbsoluteError().to(device)

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    for epoch in range(config.EPOCHS):
        model.train()

        total_loss = 0

        for batch in train_loader:
            inputs = {
                'image': batch['image'].to(device),
                'table_data': batch['table_data'].to(device),
                'total_mass': batch['total_mass'].to(device)
            }
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(**inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            _ = mae_metric_train(preds=outputs, target=labels)
        
        train_mae = mae_metric_train.compute().cpu().numpy()
        train_loss = total_loss / len(train_loader)
        val_loss, val_mae = validate(model, criterion, val_loader, device, mae_metric_val)

        logger.report_scalar(title='Loss', series='Train', value=train_loss, iteration=epoch)
        logger.report_scalar(title='Loss', series='Val', value=val_loss, iteration=epoch)
        logger.report_scalar(title='MAE', series='Train', value=train_mae, iteration=epoch)
        logger.report_scalar(title='MAE', series='Val', value=val_mae, iteration=epoch)

        print(f'Epoch: {epoch} | val loss: {val_loss}, train loss: {train_loss} | val mae: {val_mae}, train mae: {train_mae}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_mae)
        val_metrics.append(val_mae)

        mae_metric_train.reset()
        mae_metric_val.reset()

        save_model(model.state_dict(), os.path.join(config.SAVE_PATH, f'epoch_{epoch}.pth'))

    return train_losses, val_losses, train_metrics, val_metrics


def validate(model, criterion, loader, device, metric):
    model.eval()

    with torch.no_grad():
    
        total_loss = 0
        for batch in loader:
            inputs = {
                    'image': batch['image'].to(device),
                    'table_data': batch['table_data'].to(device),
                    'total_mass': batch['total_mass'].to(device)
                }
            labels = batch['label'].to(device)

            outputs = model(**inputs)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _ = metric(preds=outputs, target=labels)
    
        val_mae = metric.compute().cpu().numpy()
        val_loss = total_loss / len(loader)
    return val_loss, val_mae

def save_model(state_dict, path):
    torch.save(state_dict, path)
