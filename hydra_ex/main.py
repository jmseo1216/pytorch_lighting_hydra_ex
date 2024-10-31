import torch.utils
from torchvision.datasets import CIFAR10
import numpy as np
import torch # PyTorch 라이브러리
import torch.nn as nn # 모델 구성을 위한 라이브러리
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import random

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer, LightningDataModule

from torchmetrics import Accuracy 

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from pytorch_lightning.callbacks import LearningRateMonitor
from hydra.utils import instantiate
import hydra



# @hydra.main(config_path='configs', config_name='config', version_base=None)
# def main(cfg):
#     print("Starting the training process...")
#     train_dataset = CIFAR10(cfg.data.data_dir, train=True, download=True, transform=T.ToTensor())
#     test_dataset = CIFAR10(cfg.data.data_dir, train=False, download=True, transform=T.ToTensor())
    
#     train_num, valid_num = int(len(train_dataset) * (1-cfg.data.valid_split)), int(len(train_dataset) * cfg.data.valid_split)
#     print("Train dataset 개수: ", train_num)
#     print("Validation dataset 개수: ", valid_num)
#     train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
    
#     train_dataloader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
#     valid_dataloader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False)
#     test_dataloader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    
#     # if cfg.model.model.model_name =='simple_cnn':
#     #     model = SimpleCNN(cfg)
#     # else:
#     #     model = ResNet(cfg)
    
#     model = SimpleCNN(cfg)
    
#     early_stopping = EarlyStopping(monitor=cfg.callback.monitor,
#                                    mode = cfg.callback.mode, patience= cfg.callback.patience)
#     lr_moniter = LearningRateMonitor(logging_interval= cfg.callback.logging_interval)
    
#     logger = CSVLogger(save_dir= "./csv_logger", name='test')
    
#     trainer = Trainer(
#         **cfg.trainer,
#         callbacks= [early_stopping, lr_moniter],
#         logger = logger
#     )
#     trainer.fit(model,train_dataloader, valid_dataloader)
#     trainer.test(model, test_dataloader)    
    

# if __name__ == "__main__":
#     main()  # 메인 함수 호출



class SimpleCNN(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.learning_rate = cfg.optimizer.lr
        self.accuracy = Accuracy(task='multiclass', num_classes=cfg.model.model.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = cfg.optimizer
        self.scheduler = cfg.scheduler
        
         
        self.num_classes = cfg.model.model.num_classes
        self.dropout_ratio = cfg.model.model.dropout_ratio
        
        self.layer = nn.Sequential(
            # 입력: (32, 3, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # (32, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # (32, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32, 64, 16, 16)
            nn.Dropout(self.dropout_ratio),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # (32, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32, 128, 8, 8)
            nn.Dropout(self.dropout_ratio),
        )
        
        self.fc_layer1 = nn.Linear(8*8*128, 64) 
        self.fc_layer2 = nn.Linear(64, self.num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size(0), -1)
        out = self.fc_layer1(out)
        pred = self.fc_layer2(out)
        
        return pred 
    
    def configure_optimizers(self):
        #optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        optimizer = instantiate(self.optimizer, self.parameters())
        scheduler = instantiate(self.scheduler, optimizer)
        
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        images, labels = batch 
        
        outputs = self(images)
        
        loss = self.criterion(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, logger=True) 
        return loss 
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        
        self.log(f"valid_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log(f"valid_acc", acc ,on_step=False, on_epoch=True, logger=True) 
    
    def test_step(self, batch, batch_idx):
        images, labels = batch 
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        
        self.log(f"test_loss", loss, on_step=False, on_epoch=True)
        self.log(f"test_acc", acc, on_step=False, on_epoch=True)
        
    def predict_step(self, batch, batch_idx):
        images, labels = batch 
        outputs = self(images)
        _, preds = torch.max(outputs, dim=1)

        return preds     
    




@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg):
    print("Starting the training process...")
    train_dataset = CIFAR10(cfg.data.data_dir, train=True, download=True, transform=T.ToTensor())
    test_dataset = CIFAR10(cfg.data.data_dir, train=False, download=True, transform=T.ToTensor())
    
    train_num, valid_num = int(len(train_dataset) * (1-cfg.data.valid_split)), int(len(train_dataset) * cfg.data.valid_split)
    print("Train dataset 개수: ", train_num)
    print("Validation dataset 개수: ", valid_num)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    
    # if cfg.model.model.model_name =='simple_cnn':
    #     model = SimpleCNN(cfg)
    # else:
    #     model = ResNet(cfg)
    
    model = SimpleCNN(cfg)
    
    early_stopping = EarlyStopping(monitor=cfg.callback.monitor,
                                   mode = cfg.callback.mode, patience= cfg.callback.patience)
    lr_moniter = LearningRateMonitor(logging_interval= cfg.callback.logging_interval)
    
    logger = CSVLogger(save_dir= "./csv_logger", name='test')
    
    trainer = Trainer(
        **cfg.trainer,
        callbacks= [early_stopping, lr_moniter],
        logger = logger
    )
    trainer.fit(model,train_dataloader, valid_dataloader)
    trainer.test(model, test_dataloader)    
    

if __name__ == "__main__":
    main()  

        