import timm
import torch
from torch import nn


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(config.MLP_IN_FEATURES, config.MLP_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config.MLP_HIDDEN_SIZE, config.MLP_OUT_FEATURES)
        )

        self.image_model = timm.create_model(
            model_name=config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        # self.image_model = timm.create_model(
        #     model_name=config.IMAGE_MODEL_NAME,
        #     pretrained=False,
        #     num_classes=0
        # )

        #---
        # from timm.models import load_checkpoint
        # load_checkpoint(
        #     model=self.image_model,
        #     checkpoint_path='C:/Users/a.berdnikov/Downloads/pytorch_model.bin',
        #     strict=False
        # )
        #---

        self.image_proj = nn.Linear(self.image_model.num_features, config.PROJ_SIZE)
        self.mlp_proj = nn.Linear(config.MLP_OUT_FEATURES, config.PROJ_SIZE)

        self.regressor = nn.Sequential(
            # +1 так как добавляю значение total_mass
            nn.Linear(config.PROJ_SIZE + 1, config.PROJ_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(config.PROJ_SIZE // 2, 1),
        )
    
    def forward(self, image, table_data, total_mass):

        image_features = self.image_model(image)
        mlp_features = self.mlp(table_data)

        image_emb = self.image_proj(image_features)
        mlp_emb = self.mlp_proj(mlp_features)
        
        fused_emb = image_emb * mlp_emb

        fused_emb = torch.cat([fused_emb, total_mass], dim=1)

        logits = self.regressor(fused_emb)
        return logits


def set_requires_grad(model, unfreeze_pattern=''):
    if len(unfreeze_pattern) == 0:
        for _, param in model.named_parameters():
            param.requires_grad = False
        return
    
    patterns = unfreeze_pattern.split('|')

    for name, param in model.named_parameters():
        if any([name.startswith(p) for p in patterns]):
            param.requires_grad = True
        else:
            param.requires_grad = False
