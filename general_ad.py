import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import math
import timm

class FeatureExtractor(nn.Module):
    def __init__(self, hf_path, layer_indices, patch_size, image_size, device):
        super(FeatureExtractor, self).__init__()
        self.layer_indices = layer_indices
        self.patch_size = patch_size
        self.num_patches = (math.ceil(image_size / patch_size))**2

        # Load pretrained model
        self.pretrained_model = timm.create_model(hf_path, pretrained=True, num_classes=0).to(device)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # Determine embedding dimension
        if hasattr(self.pretrained_model, 'embed_dim'):
            self.embed_dim = self.pretrained_model.embed_dim
        else:
            self.embed_dim = self.pretrained_model.num_features

    def forward(self, x):
        features = self.pretrained_model.forward_features(x)
        # Drop CLS token if present (common in ViT)
        if features.dim() == 3 and features.shape[1] > self.num_patches:
             features = features[:, 1:, :] 
        return features

class General_AD(pl.LightningModule):
    def __init__(self, hf_path='vit_base_patch14_dinov2.lvd142m', 
                 learning_rate=0.0001, 
                 input_size=518,
                 patch_size=14):
        super(General_AD, self).__init__()
        self.save_hyperparameters()
        
        self.feature_extractor = FeatureExtractor(
            hf_path, 
            layer_indices=[11], 
            patch_size=patch_size, 
            image_size=input_size, 
            device=self.device
        )
        
        # Discriminator (Simple MLP)
        # FIX 1: Removed the final nn.Sigmoid() layer
        self.discriminator = nn.Sequential(
            nn.Linear(self.feature_extractor.embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1) 
            # Output is now a "logit" (range -infinity to +infinity)
        )
        
        # FIX 2: Use BCEWithLogitsLoss for numerical stability in FP16
        self.criterion = nn.BCEWithLogitsLoss()

    def generate_pseudo_anomalies(self, features):
        # 1. Stronger Noise (0.1 -> 0.5 or 1.0)
        # This forces the model to learn that 'Normal' is very stable
        noise = torch.randn_like(features) * 0.5 
        noisy_features = features + noise
        
        # 2. Patch Shuffling
        B, N, D = features.shape
        perm = torch.randperm(N)
        shuffled_features = features[:, perm, :]
        
        # 3. New Strategy: Feature Scaling (Dimming/Brightening patches)
        # This simulates shadows/discoloration
        scale = torch.rand(B, N, 1).to(features.device) * 2.0 
        scaled_features = features * scale

        # Mix them up randomly
        rand = torch.rand(B, 1, 1).to(features.device)
        fake_features = torch.where(rand < 0.33, noisy_features, 
                        torch.where(rand < 0.66, shuffled_features, scaled_features))
        
        return fake_features

    def forward(self, x):
        self.feature_extractor.pretrained_model.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
        
        logits = self.discriminator(features)
        
        # FIX 3: Apply sigmoid explicitly here for Inference
        # This ensures the output is still a probability [0, 1]
        return torch.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        
        self.feature_extractor.pretrained_model.eval()
        with torch.no_grad():
            real_features = self.feature_extractor(x)
            
        fake_features = self.generate_pseudo_anomalies(real_features)
        
        # Get Logits (raw scores)
        logits_real = self.discriminator(real_features)
        logits_fake = self.discriminator(fake_features)
        
        # Calculate Loss using Logits
        loss_real = self.criterion(logits_real, torch.zeros_like(logits_real))
        loss_fake = self.criterion(logits_fake, torch.ones_like(logits_fake))
        
        loss = loss_real + loss_fake
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate)
