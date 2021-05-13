#!/usr/bin/env python
# coding: utf-8

# ## Reproducible Deep Learning (PhD course, Data Science)
# ### Lecture 1: deep learning recap

# We will code a simple audio classification model (a convolutional neural network) for the ESC-50 dataset: https://github.com/karolpiczak/ESC-50. The aim is to recap some deep learning concepts, and have a working notebook to use as starting point for the next exercises.

# **Setup the machine**:
# 1. Follow the instructions from here: https://github.com/sscardapane/reprodl2021#local-set-up
# 2. Download the ESC-50 dataset inside a 'data' folder.


import torch, torchaudio
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl


from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd















class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self, path: Path = Path('data/ESC-50'), 
                 sample_rate: int = 8000,
                 folds = [1]):
        # Load CSV & initialize all torchaudio.transforms:
        # Resample --> MelSpectrogram --> AmplitudeToDB
        self.path = path
        self.csv = pd.read_csv(path / Path('meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        self.resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        
    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / 'audio' / row['filename'])
        label = row['target']
        xb = self.db(
            self.melspec(
                self.resample(wav)
            )
        )
        return xb, label
        
    def __len__(self):
        # Returns length
        return len(self.csv)







class AudioNet(pl.LightningModule):
    
    def __init__(self, n_classes = 50, base_filters = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(base_filters, base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 2)
        self.conv4 = nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(base_filters * 4, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x
    
    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = pl.metrics.functional.accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer








def train():


    # ### Step 1: Some experiments in audio loading and transformation

    # The code in this section is just for experimentation, and will be removed when porting to a script.

    # Substitute this with your actual path. This is the root folder of ESC-50, where
    # you can find the subfolders 'audio' and 'meta'.
    datapath = Path('data/ESC-50')

    # Using Path is fundamental to have reproducible code across different operating systems.
    csv = pd.read_csv(datapath / Path('meta/esc50.csv'))

    # We can use torchaudio.load to load the file. The second value is the sampling rate of the file.
    x, sr = torchaudio.load(datapath / 'audio' / csv.iloc[0, 0], normalize=True)

    # Useful transformation to resample the original file.
    torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)(x).shape


    # Another useful transformation to build a Mel spectrogram (image-like), so that
    # we can apply any CNN on top of it.
    h = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(x)

    # Convert to DB magnitude, useful for scaling.
    # Note: values could be further normalize to significantly speed-up and simplify training.
    h = torchaudio.transforms.AmplitudeToDB()(h)




    # ### Step 2: Putting together data loading and preprocessing

    train_data = ESC50Dataset()



    # ### Step 3: Build a classification model

    # We use folds 1,2,3 for training, 4 for validation, 5 for testing.
    train_data = ESC50Dataset(folds=[1,2,3])
    val_data = ESC50Dataset(folds=[4])
    test_data = ESC50Dataset(folds=[5])

    train_loader =     torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)


    pl.seed_everything(0)

    # Test that the network works on a single mini-batch
    audionet = AudioNet()


    trainer = pl.Trainer(gpus=0, max_epochs=25)
    trainer.fit(audionet, train_loader, val_loader)


    # TODO: implement the test loop.
    # trainer.test(audionet, test_loader)










if __name__ == "__main__":
    train()

