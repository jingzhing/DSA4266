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


class SwinEfficientNetEnsemble(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        swin_cfg = cfg["swin"]
        eff_cfg = cfg["efficientnet"]
        weights = cfg["ensemble_weights"]

        self.swin_weight = weights["swin"]
        self.efficientnet_weight = weights["efficientnet"]

        self.swin = BinaryBackboneHead(
            model_name=swin_cfg["model_name"],
            pretrained=True,
            dropout=swin_cfg.get("dropout", 0.0),
            drop_path_rate=swin_cfg.get("drop_path_rate", 0.0),
        )
        self.efficientnet = BinaryBackboneHead(
            model_name=eff_cfg["model_name"],
            pretrained=True,
            dropout=eff_cfg.get("dropout", 0.0),
            drop_path_rate=0.0,
        )

        if swin_cfg.get("freeze_backbone", False):
            for p in self.swin.backbone.parameters():
                p.requires_grad = False
        if eff_cfg.get("freeze_backbone", False):
            for p in self.efficientnet.backbone.parameters():
                p.requires_grad = False

    def forward(self, inputs):
        swin_x = inputs["swin"]
        eff_x = inputs["efficientnet"]

        swin_logits = self.swin(swin_x)
        efficientnet_logits = self.efficientnet(eff_x)

        swin_probs = torch.sigmoid(swin_logits)
        efficientnet_probs = torch.sigmoid(efficientnet_logits)

        ensemble_probs = (
            self.swin_weight * swin_probs +
            self.efficientnet_weight * efficientnet_probs
        ) / (self.swin_weight + self.efficientnet_weight)
        ensemble_probs = torch.clamp(ensemble_probs, min=1e-6, max=1 - 1e-6)
        ensemble_logits = torch.logit(ensemble_probs)

        return {
            "swin_logits": swin_logits,
            "swin_probs": swin_probs,
            "efficientnet_logits": efficientnet_logits,
            "efficientnet_probs": efficientnet_probs,
            "ensemble_logits": ensemble_logits,
            "ensemble_probs": ensemble_probs,
        }


def build_ensemble_model(cfg):
    return SwinEfficientNetEnsemble(cfg)


def build_optimizer(model, cfg):
    eff_cfg = cfg["efficientnet"]
    swin_cfg = cfg["swin"]

    param_groups = [
        {
            "params": [p for p in model.efficientnet.parameters() if p.requires_grad],
            "lr": eff_cfg["lr"],
            "weight_decay": eff_cfg.get("weight_decay", 0.0),
        },
        {
            "params": [p for p in model.swin.parameters() if p.requires_grad],
            "lr": swin_cfg["lr"],
            "weight_decay": swin_cfg.get("weight_decay", 0.0),
        },
    ]
    return torch.optim.AdamW(param_groups)
