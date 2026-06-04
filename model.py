# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import OHLCVTransformer, ViT, TimeframeFusion
from config import *

class MarketPredictor(nn.Module):
    def __init__(self, use_image=False, smc_feature_dim=6):
        super().__init__()
        self.use_image = use_image
        if use_image:
            self.vision_model = ViT()
        else:
            self.tf_encoders = nn.ModuleDict({tf: OHLCVTransformer() for tf in TIMEFRAMES})
            self.fusion = TimeframeFusion(d_model=D_MODEL, num_timeframes=len(TIMEFRAMES))
        self.smc_proj = nn.Linear(smc_feature_dim, D_MODEL)
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL*2, D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, NUM_CLASSES)
        )
        self.confidence_estimator = nn.Sequential(
            nn.Linear(D_MODEL*2, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
            nn.Sigmoid()
        )

    def forward(self, ohlcv_dict=None, image=None, smc_features=None):
        if self.use_image:
            base_features = self.vision_model(image)
        else:
            tf_feats = [self.tf_encoders[tf](ohlcv_dict[tf]) for tf in TIMEFRAMES]
            base_features = self.fusion(tf_feats)
        if smc_features is not None:
            smc_proj = self.smc_proj(smc_features)
            combined = torch.cat([base_features, smc_proj], dim=-1)
        else:
            combined = torch.cat([base_features, torch.zeros_like(base_features)], dim=-1)
        logits = self.classifier(combined)
        confidence = self.confidence_estimator(combined).squeeze(-1) * 100
        return logits, confidence

    def predict_with_uncertainty(self, *args, mc_samples=MC_DROPOUT_SAMPLES, **kwargs):
        self.train()
        preds, confs = [], []
        for _ in range(mc_samples):
            logits, conf = self.forward(*args, **kwargs)
            preds.append(F.softmax(logits, dim=-1))
            confs.append(conf)
        preds = torch.stack(preds); confs = torch.stack(confs)
        return preds.mean(0), confs.mean(0), preds.std(0).mean(-1)
