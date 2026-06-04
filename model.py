# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import OHLCVTransformer, ViT, TimeframeFusion
from config import *

class MarketPredictor(nn.Module):
    """
    Combined model:
    - For OHLCV: use OHLCVTransformer per timeframe, then fuse.
    - For image: use ViT.
    - SMC features concatenated before final classifier.
    """
    def __init__(self, use_image=False, smc_feature_dim=6):
        super().__init__()
        self.use_image = use_image
        if use_image:
            self.vision_model = ViT()
        else:
            self.tf_encoders = nn.ModuleDict({
                tf: OHLCVTransformer() for tf in TIMEFRAMES
            })
            self.fusion = TimeframeFusion(d_model=D_MODEL, num_timeframes=len(TIMEFRAMES))
        
        # SMC features projection
        self.smc_proj = nn.Linear(smc_feature_dim, D_MODEL)
        
        # Classifier
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

    def forward(self, ohlcv_dict=None, image=None, smc_features=None, return_attention=False):
        """
        ohlcv_dict: dict of {timeframe: tensor (B, seq_len, 4)}
        image: (B, C, H, W)
        smc_features: (B, smc_dim)
        """
        if self.use_image:
            base_features = self.vision_model(image)   # (B, D_MODEL)
        else:
            tf_feats = []
            for tf in TIMEFRAMES:
                x = ohlcv_dict[tf]   # (B, seq_len, 4)
                feat = self.tf_encoders[tf](x)
                tf_feats.append(feat)
            base_features = self.fusion(tf_feats)   # (B, D_MODEL)
        
        # SMC features
        if smc_features is not None:
            smc_proj = self.smc_proj(smc_features)
            combined = torch.cat([base_features, smc_proj], dim=-1)
        else:
            combined = torch.cat([base_features, torch.zeros_like(base_features)], dim=-1)
        
        logits = self.classifier(combined)
        confidence = self.confidence_estimator(combined).squeeze(-1) * 100   # 0-100%
        return logits, confidence

    def predict_with_uncertainty(self, *args, mc_samples=MC_DROPOUT_SAMPLES, **kwargs):
        """Monte Carlo Dropout for uncertainty."""
        self.train()   # keep dropout active
        preds = []
        confs = []
        for _ in range(mc_samples):
            logits, conf = self.forward(*args, **kwargs)
            preds.append(F.softmax(logits, dim=-1))
            confs.append(conf)
        preds = torch.stack(preds)
        confs = torch.stack(confs)
        mean_pred = preds.mean(dim=0)
        uncertainty = preds.std(dim=0).mean(dim=-1)   # mean std over classes
        mean_conf = confs.mean(dim=0)
        return mean_pred, mean_conf, uncertainty