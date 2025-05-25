"""
model_builder.py
This module defines a set of neural network architectures and a selection of these models for the DroneSearch project.
It provides modular implementations of VGG-like and U-Net-like models.

-----------------------------
Constants:
- MODELS (list of str): 
  Available model types for selection. ["VGG8", "UnetS"]  
  ["VGGVariation", "SimpleEncDec"] models are variatons for testing purposes.

-----------------------------
Functions:
- choose_model(model_type, output_shape, device, input_shape, window_size):
  Returns the selected model architecture, initialized with the specified parameters.

-----------------------------
Classes:
- VGG8:            A compact VGG-style convolutional neural network for classification.
- UnetS:           A U-Net style encoder-decoder network for segmentation, with skip connections and dropout.

- VGGVariation:    A deeper VGG-style network with additional fully connected layers and dropout (for testing purposes).
- SimpleEncDec:    A minimal encoder-decoder network for quick prototyping (for testing purposes).

-----------------------------
Dependencies:
- torch, logging

-----------------------------
Usage:
- Import this module in training scripts or testing scripts to create an instance of a machine learning model.
"""

import torch
from torch import nn 
import logging 

MODELS = ["VGG8", "UnetS", "VGGVariation","SimpleEncDec"]

def choose_model(model_type=MODELS[0], output_shape=1, device="cuda", input_shape=1, window_size=[64,64]):
  """Returns Model from model_type.
  Args:
  model_type(str): String for which model to load
  classes(int): An integer indicating number of classes.
  device(str): Device to be used (cuda/cpu)
  """
  if(model_type==MODELS[0]):    
    model = VGG8(output_shape=output_shape,input_shape=input_shape,window_size=window_size).to(device)
  elif(model_type==MODELS[1]):
    output_shape=1
    model = UnetS(output_shape=output_shape,input_shape=input_shape).to(device)
  elif(model_type==MODELS[2]):
     model = VGGVariation(output_shape=output_shape,input_shape=input_shape,window_size=window_size).to(device)
  elif(model_type==MODELS[3]):
    output_shape=1
    model = SimpleEncDec().to(device)
  logging.info(f"[BUILDER] Model (Type: {model_type}, Classes: {output_shape}, Device: {device}) loaded.")  
  return model



class VGG8(nn.Module):
    """Creates the VGGVariation architecture based on input with at least 24*24 input.
    Args:
    input_shape(int): An integer indicating number of input channels (default 1 channel).
    output_shape(int): An integer indicating number of classes.
    """
    def __init__(self, output_shape: int,input_shape=1, window_size=[64,64]) -> None:
        FEATURE_MAP=[32,64,128]
        LINEAR_MULTIPLIER=[window_size[0]//8,window_size[1]//8]
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, out_channels=FEATURE_MAP[0], kernel_size=3, stride=1, padding=1),  
          nn.BatchNorm2d(FEATURE_MAP[0]),
          nn.ReLU(),
          nn.Conv2d(in_channels=FEATURE_MAP[0], out_channels=FEATURE_MAP[0], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[0]),                    
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(FEATURE_MAP[0], FEATURE_MAP[1], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[1]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[1], FEATURE_MAP[1], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[1]),          
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block_3 = nn.Sequential(
          nn.Conv2d(FEATURE_MAP[1], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[2], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[2], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          #nn.Linear(in_features=FEATURE_MAP[2]*LINEAR_MULTIPLIER[0]*LINEAR_MULTIPLIER[1], out_features=FEATURE_MAP[2]*LINEAR_MULTIPLIER[0]*LINEAR_MULTIPLIER[1]),#needs to be changed according to data
          #nn.ReLU(),
          #nn.Dropout(0.5),
          #nn.Linear(in_features=FEATURE_MAP[2]*LINEAR_MULTIPLIER[0]*LINEAR_MULTIPLIER[1], out_features=FEATURE_MAP[2]*LINEAR_MULTIPLIER[0]*LINEAR_MULTIPLIER[1]),
          #nn.ReLU(),
          nn.Linear(in_features=FEATURE_MAP[2]*LINEAR_MULTIPLIER[0]*LINEAR_MULTIPLIER[1], out_features=output_shape)#,#needs to be changed according to data
          #nn.ReLU(),
          #
          #nn.Linear(in_features=4096, out_features=output_shape),
          #nn.ReLU()
          #nn.Dropout(0.5),          
          #nn.Linear(in_features=2048, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        #print(x.shape)
        x = self.conv_block_1(x)
        #print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.conv_block_3(x)
       # print(x.shape)
        x = self.classifier(x)
       # print(x.shape)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion


class UnetS(nn.Module):
    """Creates the VGGVariation architecture based on 64*64 input.
    Args:
    input_shape(int): An integer indicating number of input channels (default 1 channel).
    output_shape(int): An integer indicating number of classes.
    """
    def __init__(self, output_shape: int,input_shape=1,dropout=0.1) -> None:
        FEATURE_MAP=[32,64,128,256]
        super().__init__()
        self.dropout = dropout

        # Encoder
        self.enc_block_1 = self.conv_block(input_shape,FEATURE_MAP[0])
        self.enc_block_2 = self.conv_block(FEATURE_MAP[0],FEATURE_MAP[1])
        self.enc_block_3 = self.conv_block(FEATURE_MAP[1],FEATURE_MAP[2])

        # Bottleneck
        self.bottleneck = self.conv_block(FEATURE_MAP[2],FEATURE_MAP[3])

        #Decoder
        self.upsample_3 = nn.ConvTranspose2d(FEATURE_MAP[3],FEATURE_MAP[2],kernel_size=2,stride=2)
        self.dec_block_3 = self.conv_block(FEATURE_MAP[3],FEATURE_MAP[2])
        self.upsample_2 = nn.ConvTranspose2d(FEATURE_MAP[2],FEATURE_MAP[1],kernel_size=2,stride=2)
        self.dec_block_2 = self.conv_block(FEATURE_MAP[2],FEATURE_MAP[1])
        self.upsample_1 = nn.ConvTranspose2d(FEATURE_MAP[1],FEATURE_MAP[0],kernel_size=2,stride=2)
        self.dec_block_1 = self.conv_block(FEATURE_MAP[1],FEATURE_MAP[0])
        
        #Final
        self.final_block=nn.Sequential(
          nn.Conv2d(FEATURE_MAP[0],output_shape,kernel_size=1),
          #nn.Sigmoid()
          )
    

    def conv_block(self, in_channels, out_channels):
      """Convolutional block with BatchNorm + ReLU."""
      return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

    def forward(self, x: torch.Tensor):
        #Encoder
        enc_block_1 = self.enc_block_1(x)
        enc_block_2 = self.enc_block_2(self.downsample(enc_block_1))
        enc_block_3 = self.enc_block_3(self.downsample(enc_block_2))

        #Bottleneck
        bottleneck = self.bottleneck(self.downsample(enc_block_3))

        #Decoder
        dec_block_3=self.dec_block_3(self.skip_connection(self.upsample_3(bottleneck),enc_block_3))
        dec_block_2=self.dec_block_2(self.skip_connection(self.upsample_2(dec_block_3),enc_block_2))
        dec_block_1=self.dec_block_1(self.skip_connection(self.upsample_1(dec_block_2),enc_block_1))

        #Final
        return self.final_block(dec_block_1)
        
    def downsample(self,x):
      return nn.MaxPool2d(kernel_size=2,stride=2)(x)

    def skip_connection(self, x, skip):
      # Concatenate skip connection
      #a,b,h,w=x.size()
      #skip = skip.reshape(a, b, h, w)
      return torch.cat((skip, x), dim=1)
    










class VGGVariation(nn.Module):
    """Creates the VGGVariation architecture based on input with at least 24*24 input.
    Args:
    input_shape(int): An integer indicating number of input channels (default 1 channel).
    output_shape(int): An integer indicating number of classes.
    """
    def __init__(self, output_shape: int,input_shape=1, window_size=[64,64]) -> None:
        FEATURE_MAP=[32,64,128]
        LINEAR_MULTIPLIER=[window_size[0]//8,window_size[1]//8]
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, out_channels=FEATURE_MAP[0], kernel_size=3, stride=1, padding=1),  
          nn.BatchNorm2d(FEATURE_MAP[0]),
          nn.ReLU(),
          nn.Conv2d(in_channels=FEATURE_MAP[0], out_channels=FEATURE_MAP[0], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[0]),                    
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(FEATURE_MAP[0], FEATURE_MAP[1], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[1]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[1], FEATURE_MAP[1], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[1]),          
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block_3 = nn.Sequential(
          nn.Conv2d(FEATURE_MAP[1], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[2], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[2], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=FEATURE_MAP[2]*LINEAR_MULTIPLIER[0]*LINEAR_MULTIPLIER[1], out_features=4096),#needs to be changed according to data
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(in_features=4096, out_features=2048),
          nn.ReLU(),
          nn.Dropout(0.5),          
          nn.Linear(in_features=2048, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        #print(x.shape)
        x = self.conv_block_1(x)
        #print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.conv_block_3(x)
       # print(x.shape)
        x = self.classifier(x)
       # print(x.shape)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion




#TEST ---------------------------------------------------------------TEST




# Model: A simple UNet-like structure
class SimpleEncDec(nn.Module):
    def __init__(self, input_shape=1) -> None:
        super(SimpleEncDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            #nn.Sigmoid(),  # Output probabilities
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x