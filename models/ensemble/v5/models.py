import timm
import torch
from torch import nn


class BinaryBackboneHead(nn.Module):
    def __init__(self, model_name, pretrained=True, dropout=0.0, drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        return self.head(features)


def _apply_partial_finetune_efficientnet(model, blocks_to_unfreeze=30):
    for p in model.backbone.parameters():
        p.requires_grad = False

    backbone = model.backbone
    if hasattr(backbone, "blocks"):
        blocks = list(backbone.blocks)
        for block in blocks[-blocks_to_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

    for attr in ["conv_head", "bn2", "global_pool"]:
        if hasattr(backbone, attr):
            module = getattr(backbone, attr)
            if hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True


def build_binary_model(cfg):
    model_cfg = cfg["model"]
    model = BinaryBackboneHead(
        model_name=model_cfg["arch"],
        pretrained=True,
        dropout=model_cfg.get("dropout", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )

    if model_cfg.get("freeze_backbone", False):
        for p in model.backbone.parameters():
            p.requires_grad = False

    if "efficientnet" in model_cfg["arch"] and model_cfg.get("partial_finetune", False):
        _apply_partial_finetune_efficientnet(
            model,
            blocks_to_unfreeze=model_cfg.get("partial_finetune_blocks", 30),
        )

    return model


def build_optimizer(model, cfg):
    train_cfg = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    if train_cfg.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError("Only AdamW is implemented in this pipeline")
    return torch.optim.AdamW(
        params,
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )


def build_scheduler(optimizer, cfg):
    sched_cfg = cfg["train"].get("scheduler")
    if not sched_cfg:
        return None
    if sched_cfg.get("name") == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "max"),
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 1),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    raise ValueError(f"Unsupported scheduler: {sched_cfg}")
