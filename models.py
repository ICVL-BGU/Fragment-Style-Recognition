import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy, Precision, Recall

################################################################
#                                                              #
#                         Baselines                            #
#                                                              #
################################################################

# ================ Custom CNN ================

class CNN(nn.Module):
    """
    A shallow CNN for multi-style classification.
    """
    def __init__(self, n_styles):
        super().__init__()
        self.n_styles = n_styles
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 56 * 56, n_styles)
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        # Test metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=n_styles, average='micro')
        self.precision = Precision(task='multiclass', num_classes=n_styles, average='macro')
        self.recall = Recall(task='multiclass', num_classes=n_styles, average='macro')

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.fc(x)
        return x
    

# ================  Fine-Tuning with EfficientNet ================

class EfficientNetBackbone(nn.Module):
    """
    A fine-tuned EfficientNet model for multi-style classification.
    """
    def __init__(self, n_styles, freeze_upto=4):
        super().__init__()
        self.n_styles = n_styles
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        for layer in self.backbone.features[:freeze_upto]:
            for param in layer.parameters():
                param.requires_grad = False
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, n_styles)

    def forward(self, x):
        return self.backbone(x)


# ================ Baseline Pytorch Lightning Module ================

class Baseline(pl.LightningModule):
    """
    Root class for all baseline models.
    Used for training and evaluation with PyTorch Lightning framework.
    """
    def __init__(self, model, lr=1e-4):
        super().__init__()
        n_styles = model.n_styles
        self.model = model
        self.n_styles = n_styles
        self.lr = lr
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # Test metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=n_styles, average='micro')
        self.precision = Precision(task='multiclass', num_classes=n_styles, average='macro')
        self.recall = Recall(task='multiclass', num_classes=n_styles, average='macro')
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, _, y_true = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, y_true = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, _, y_true = batch
        y_pred = self(x)
        self.accuracy(y_pred, y_true)
        self.precision(y_pred, y_true)
        self.recall(y_pred, y_true)
        self.log_dict({
            'test_acc': self.accuracy,
            'test_precision': self.precision,
            'test_recall': self.recall
        })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

################################################################
#                                                              #
#                       Proposed Model                         #
#                                                              #
################################################################

class MaskedContentLoss(nn.Module):
    """
    Masked content loss for outpainting.
    """ 
    def __init__(self):
        super().__init__()

    def forward(self, x, x_rec, mask):
        return F.mse_loss(mask * x, mask * x_rec)

class AutoStyleLoss(nn.Module):
    """
    Auto-style loss for outpainting.
    """
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights='DEFAULT').features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.activations = [1, 6, 11, 20, 29]
        self.features = []
        for act in self.activations:
            self.vgg[act].register_forward_hook(lambda _, __, output: self.features.append(output))

    def get_features(self, x):
        self.vgg(x)
        features = self.features
        self.features = []
        return features
    
    def gram(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x_t = x.transpose(1, 2)
        return torch.bmm(x, x_t) / (c * h * w)

    def forward(self, x, x_rec):
        x_grams = [self.gram(f) for f in self.get_features(x)]
        x_rec_grams = [self.gram(f) for f in self.get_features(x_rec)]
        loss = 0
        for g, g_rec in zip(x_grams, x_rec_grams):
            loss += F.mse_loss(g, g_rec)
        return loss


class StyleExtrapolator(pl.LightningModule):
    """
    Proposed model for multi-style outpainting.
    """
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        resnet = models.resnet18(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            # Input is 512x7x7
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 256x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 128x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64x56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32x112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 16x224x224
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1), # 3x224x224
            nn.Sigmoid()
        )
        # Loss function
        self.content_loss = MaskedContentLoss()
        self.style_loss = AutoStyleLoss()
    
    def forward(self, x):
        return x + self.decoder(self.encoder(x))
    
    def training_step(self, batch, batch_idx):
        x, mask, _ = batch
        x_rec = self(x)
        loss = self.content_loss(x, x_rec, mask) + self.style_loss(x, x_rec)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, _ = batch
        x_rec = self(x)
        loss = self.content_loss(x, x_rec, mask) + self.style_loss(x, x_rec)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, mask, _ = batch
        x_rec = self(x)
        loss = self.content_loss(x, x_rec, mask) + self.style_loss(x, x_rec)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class Proposed(pl.LightningModule):
    """
    Proposed model for multi-style classification.
    """
    def __init__(self, backbone, style_extrapolator, lr):
        super().__init__()
        n_styles = backbone.n_styles
        self.style_extrapolator = style_extrapolator
        for param in self.style_extrapolator.parameters():
            param.requires_grad = False
        self.backbone = backbone
        self.lr = lr
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # Test metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=n_styles, average='micro')
        self.precision = Precision(task='multiclass', num_classes=n_styles, average='macro')
        self.recall = Recall(task='multiclass', num_classes=n_styles, average='macro')

    def forward(self, x):
        x = self.style_extrapolator(x)
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, _, y_true = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y_true = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, _, y_true = batch
        y_pred = self(x)
        self.accuracy(y_pred, y_true)
        self.precision(y_pred, y_true)
        self.recall(y_pred, y_true)
        self.log_dict({
            'test_acc': self.accuracy,
            'test_precision': self.precision,
            'test_recall': self.recall
        })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)