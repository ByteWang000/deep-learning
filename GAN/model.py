import torch
import torch.nn as nn
import numpy as np


# Generator Code by Myself
class Generator1(nn.Module):
    def __init__(self, latent_dim,image_size):
        super(Generator1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),

            nn.Linear(1024,np.prod(image_size,dtype=np.int32)),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        output = self.model(z)
        # np.prod(image_size,dtype=np.int32) = 784
        # output.shape = [batchsize, 784]
        # image = output.reshape(z.shape[0], *image_size)
        image = output.reshape(z.shape[0],*(1,28,28))
        return image
# Discriminator Code by Myself
class Discriminator1(nn.Module):

    def __init__(self, image_size):
        super(Discriminator1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype=np.int32), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        # shape of image: [batchsize, 1, 28, 28]
        prob = self.model(image.reshape(image.shape[0], -1))
        return prob