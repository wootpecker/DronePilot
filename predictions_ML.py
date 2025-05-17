"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
#import ..model_dataloader as model_dataloader
from logs import logger
import predictions
import parameters
import model_builder
import random
import pandas as pd
import math
import logging
from pathlib import Path
import os
import numpy as np 

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPES = ["VGG", "EncoderDecoder", "VGGVariation"]

HYPER_PARAMETERS = {
              "SAVE_DATASET": False,
               "TRANSFORM": False,
               "MODEL_TYPES": ["VGG", "EncoderDecoder", "VGGVariation"],
               "LOGS_SAVE": False,
               "AMOUNT_SAMPLES": 16,
               "WINDOW_SIZE": [64,64]
  }



TRAINING_PARAMETERS = {
              "NUM_EPOCHS": 50,
               "BATCH_SIZE": 128,
               "LEARNING_RATE": 0.001,
               "LOAD_SEED": 16923,
               "TRAIN_SEED": 42
  }

LOGS_SAVE = False
WINDOW_SIZE=HYPER_PARAMETERS['WINDOW_SIZE']
NAMES=["F_1","F","F_1.5s","F_60cm","F_fastGas","F_Gas","F_less10cm","F_NoGas"]
EXAMPLE=NAMES[0]
TRANFORM_TO_CENTER=True

MODEL_TO_TEST=(HYPER_PARAMETERS['MODEL_TYPES'][1],HYPER_PARAMETERS['MODEL_TYPES'][2])
#MODEL_TO_TEST=(HYPER_PARAMETERS['MODEL_TYPES'][0],HYPER_PARAMETERS['MODEL_TYPES'][1],HYPER_PARAMETERS['MODEL_TYPES'][2])


def main():
    logger.logging_config(logs_save=HYPER_PARAMETERS['LOGS_SAVE'], filename="crazyflie_predictions")
    model_type=MODEL_TO_TEST[0]
    df = predictions.load_csv(parameters.PARAMETERS[1],EXAMPLE)
    imageL,imageR = transform_data(df)
    model = model_builder.choose_model(model_type=model_type,output_shape=1024,device=device,window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
    y_pred,y_pred_percent=do_predictions(imageL, model = model, model_type=model_type)
    plot_predictions(imageL,y_pred,y_pred_percent)
    y_pred,y_pred_percent=do_predictions(imageR, model = model, model_type=model_type)
    plot_predictions(imageR,y_pred,y_pred_percent)

def transform_data(df, window_size=HYPER_PARAMETERS['WINDOW_SIZE']):
    logging.info("Plottting Gas Distribution.")    
    df_plot=predictions.transform_column(df=df)
    X=df_plot['X']
    Y=df_plot['Y']
    Gas1L = df_plot['Gas1L']
    Gas1R = df_plot['Gas1R']
    Gas1R = (Gas1R - Gas1R.min()) / (Gas1R.max() - Gas1R.min())
    Gas1L = (Gas1L - Gas1L.min()) / (Gas1L.max() - Gas1L.min())
    imageGasL = np.zeros((window_size[0], window_size[1]))
    imageGasR = np.zeros((window_size[0], window_size[1]))

    for i in range(len(Gas1L)):
        imageGasL[X[i], Y[i]] = Gas1L[i]
        imageGasR[X[i], Y[i]] = Gas1R[i]

    zeros=torch.zeros(size=Gas1R.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)#)
    img1= ax1.imshow(imageGasL, cmap="turbo", origin="lower", vmin=0, vmax=Gas1R.max())
    ax1.set_title('Gas Distribution Left')
    img2=ax2.imshow(imageGasR, cmap="turbo", origin="lower", vmin=0, vmax=Gas1R.max())
    ax2.set_title('Gas Distribution Right')
    cbar = fig.colorbar(img1, ax=ax2, orientation='vertical')
    cbar.set_label('Intensity')
    #cbar = plt.colorbar(img1, ax=ax1)
    #plt.colorbar(img2, ax=ax2)
    plt.show()
    return imageGasL, imageGasR



def do_predictions(image, model,model_type= "VGG"):
    model,_ = load_model(model=model, model_type=model_type, device=device)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(f"{name}: {param.data}")
    model.eval()
    image=torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        y_pred = model(image)
        if model_type==MODEL_TYPES[1]:
            y_pred_percent= torch.sigmoid(y_pred)
        else:
            y_pred_percent= torch.softmax(y_pred, dim=1)            
    return y_pred,y_pred_percent
 


def load_model(model: torch.nn.Module, model_type: str, device="cuda"):
  """Saves a PyTorch model to a target directory.
  Args:
  model: A target PyTorch model to save.
  target_dir: A directory for saving the model to.
  model_name: A filename for the saved model. FileEnding pth will be added
  Example usage:
  save_model(model=model_0, target_dir="model", model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(f"model")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  target_dir_path = Path(f"model/{model_type}")
  target_dir_path.mkdir(parents=True, exist_ok=True)
  files=os.listdir(target_dir_path)
  
  if len(files)==0:
    return model,0
  start=len(files)
  save_format=".pth"
  model_name = model_type + "_" + device + f"_{start:03d}" + save_format
  model_load_path = target_dir_path / model_name
  model.load_state_dict(torch.load(f=model_load_path,weights_only=True))
  model = model.to(device)
  # Save the model state_dict()
  logging.info(f"[LOAD] Loading model from: {model_load_path}")
  return model,start

def plot_predictions(X, y_pred, y_pred_percent):
    y_pred=y_pred.to("cpu").reshape(HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1])
    y_pred_percent=y_pred_percent.to("cpu").reshape(HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].set_title('Flight Path', fontsize=14)
    X_img=X[0:25,0:25]
    y_pred_img=y_pred[0:25,0:25]
    y_pred_percent_img=y_pred_percent[0:25,0:25]
    img1 = axes[0].imshow(X_img, cmap="viridis", origin="lower",alpha=1)
    #img1_overlay = axes[0].imshow(y_pred_percent_img, cmap="turbo", origin="lower",alpha=0.5)
    #plt.colorbar(img1, ax=ax1, label="X Intensity")
    #plt.colorbar(img1_overlay, ax=ax1, label="Prediction Intensity")
    img2= axes[1].imshow(y_pred_img, cmap="turbo", origin="lower")
    axes[1].set_title('Model Prediction', fontsize=14)
    img3= axes[2].imshow(y_pred_percent_img, cmap="turbo", origin="lower",alpha=1)
    img3_overlay = axes[2].imshow(X_img, cmap="viridis", origin="lower",alpha=0.2)
    color = "red"
    y, x = divmod(torch.argmax(y_pred_percent).item(), HYPER_PARAMETERS['WINDOW_SIZE'][1])
    axes[2].set_title(f'Predictions, Max={y_pred_percent.max():.2f} at ({x},{y})', color=color, fontsize=14)
    axes[0].set_ylabel(f"y (dm)", fontsize=14)
    for ax in axes:
        #ax.axis('off')
        ax.label_outer() 
        ax.set_ylabel(f"y (dm)", fontsize=14)
        ax.set_xlabel(f"x (dm)", fontsize=14)
        ax.label_outer() 
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.76])  # adjust as needed
    fig.colorbar(img1, cax=cbar_ax).set_label('Intensity')

    #cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), orientation='vertical')
    #cbar.set_label('Intensity')
    #plt.colorbar(img3, ax=axes[2], label="Prediction Intensity")
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()





if __name__ == "__main__":
    main()