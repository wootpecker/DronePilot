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
import transform_measurements
import parameters
import model_builder
import random
import pandas as pd
import math
import logging
from pathlib import Path
import os
import numpy as np 
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import flightpaths
import utils

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

HYPER_PARAMETERS = {
              "SAVE_DATASET": False,
               "TRANSFORM": False,
               "MODEL_TYPES": ["VGG8", "UnetS"],
               "LOGS_SAVE": False,
               "AMOUNT_SAMPLES": 16,
               "WINDOW_SIZE": [64,64]
  }



LOGS_SAVE = False
WINDOW_SIZE=HYPER_PARAMETERS['WINDOW_SIZE']
NAMES=["all","A","A_invers","B","B_invers"]
EXAMPLE=NAMES[3]
TRANFORM_TO_CENTER=True
SHOW_PLOT=False



MODEL_TO_TEST=[HYPER_PARAMETERS['MODEL_TYPES'][1],HYPER_PARAMETERS['MODEL_TYPES'][1]]
#MODEL_TO_TEST=(HYPER_PARAMETERS['MODEL_TYPES'][0],HYPER_PARAMETERS['MODEL_TYPES'][1],HYPER_PARAMETERS['MODEL_TYPES'][2])

BENCH=[]
PRED=[]

def main():
    logger.logging_config(logs_save=HYPER_PARAMETERS['LOGS_SAVE'], filename="crazyflie_predictions")
    dfs = utils.load_csv(parameters.PARAMETERS[1],EXAMPLE)
    model_type=MODEL_TO_TEST[0]
    model = model_builder.choose_model(model_type=model_type,output_shape=HYPER_PARAMETERS['WINDOW_SIZE'][0]*HYPER_PARAMETERS['WINDOW_SIZE'][1],device=device,window_size=HYPER_PARAMETERS['WINDOW_SIZE'])
    model,_ = load_model(model=model, model_type=model_type, device=device)
    for x,df in enumerate(dfs):
        imageL,imageR = transform_data(df)
        

        y_pred,y_pred_percent=do_predictions(imageL, model = model, model_type=model_type)
        plot_predictions(imageL,y_pred,y_pred_percent)
        y_pred,y_pred_percent=do_predictions(imageR, model = model, model_type=model_type)
        plot_predictions(imageR,y_pred,y_pred_percent)

    print("Benchmarks: ",BENCH)
    print("Predictions: ",PRED)

def transform_data(df, window_size=HYPER_PARAMETERS['WINDOW_SIZE'],plot=False):
    logging.info("Plottting Gas Distribution.")    
    df_plot=transform_measurements.transform_column(df=df)
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

    if plot:
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
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(f"{name}: {param.data}")
    model.eval()
    image=torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        y_pred = model(image)
        if model_type==HYPER_PARAMETERS["MODEL_TYPES"][1]:
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
  elif len(files)==1:
    model_load_path = target_dir_path / files[0]
    start=1
  else:
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
    source_location=[9, 15]
    wind_arrow=[12, 15]
    y_pred=y_pred.to("cpu").reshape(HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1])
    y_pred_percent=y_pred_percent.to("cpu").reshape(HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1])
    # Compute global min and max for consistent color scaling
    vmin = X.min()
    vmax = X.max()

    bench_y, bench_x = divmod(np.argmax(X).item(), HYPER_PARAMETERS['WINDOW_SIZE'][1])
    flight_path = flightpaths.flightpath_to_coordinates("Snake",[20,20],4,1)
    flight_path=[[int(x * 10),int(y*10)] for x,y in flight_path]
    flight_path=np.array(flight_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].set_title('Flight Path', fontsize=14)
    X_img=X[0:21,0:21]
    y_pred_img=y_pred[0:21,0:21]
    y_pred_percent_img=y_pred_percent[0:21,0:21]
    img1 = axes[0].imshow(X_img, cmap="viridis", origin="lower",alpha=1, vmin=vmin, vmax=vmax)
    source_location_1=axes[0].plot(source_location[0], source_location[1], marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    img1_flightpath = axes[0].scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10)

    wind_direction_1_right=axes[0].annotate('', xy=(wind_arrow[0], wind_arrow[1]), xytext=(source_location[0], source_location[1]), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)

    img2= axes[1].imshow(y_pred_percent_img, cmap="viridis", origin="lower", alpha=1, vmin=vmin, vmax=vmax)
    y, x = divmod(torch.argmax(y_pred_percent).item(), HYPER_PARAMETERS['WINDOW_SIZE'][1])
    axes[1].plot(x, y, marker='*', color='deepskyblue', markersize=15, linestyle='None', label='Model Prediction')
    
    axes[1].set_title('Model Prediction', fontsize=14)
    #img3= axes[2].imshow(y_pred_percent_img, cmap="turbo", origin="lower",alpha=1)

    
    X=torch.zeros((HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1]))


    #img3_overlay = axes[2].imshow(X_img, cmap="viridis", origin="lower",alpha=0.9, vmin=vmin, vmax=vmax)
    img2= axes[2].imshow(X_img, cmap="viridis", origin="lower",alpha=0.9)
    img2_flightpath = axes[2].scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10, label='Intended Flight Path')

 

    axes[2].set_title(f'Max={y_pred_percent.max():.2f} at ({x},{y})', fontsize=14)
    axes[2].plot(x, y, marker='*', color='deepskyblue', markersize=15, linestyle='None', label='Model Prediction')
    source_location_1=axes[2].plot(source_location[0], source_location[1], marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    #source_location_1=axes[2].plot(9, 15, marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    #wind_direction_1_right=axes[2].annotate('', xy=(11, 15), xytext=(9, 15), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)
    wind_direction_1_right=axes[2].annotate('', xy=(wind_arrow[0], wind_arrow[1]), xytext=(source_location[0], source_location[1]), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)

        #legend
    def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
        return p
    handles, labels = axes[2].get_legend_handles_labels()

    arrow = plt.arrow(0, 0, 0, 0, color='deepskyblue', label='Wind Direction')
    handles.append(arrow)
    labels.append('Wind Direction')
    axes[2].legend(loc='lower right',handles=handles, fontsize=11,  labels=labels,handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),})

    
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


    global BENCH, PRED
    PRED.append([x,y])
    BENCH.append([bench_x,bench_y])

    plt.show()

def plot_predictions3(X, y_pred, y_pred_percent):
    y_pred = y_pred.to("cpu").reshape(HYPER_PARAMETERS['WINDOW_SIZE'][0], HYPER_PARAMETERS['WINDOW_SIZE'][1])
    y_pred_percent = y_pred_percent.to("cpu").reshape(HYPER_PARAMETERS['WINDOW_SIZE'][0], HYPER_PARAMETERS['WINDOW_SIZE'][1])
    # Compute global min and max for consistent color scaling
    vmin = X.min()
    vmax = X.max()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5),constrained_layout=True)
    axes[0].set_title('Flight Path', fontsize=14)
    X_img = X[0:20, 0:20]
    y_pred_img = y_pred[0:20, 0:20]
    y_pred_percent_img = y_pred_percent[0:20, 0:20]
    img1 = axes[0].imshow(X_img, cmap="viridis", origin="lower", alpha=1, vmin=vmin, vmax=vmax)
    #img2 = axes[1].imshow(y_pred_percent_img, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    #axes[1].set_title('Model Prediction', fontsize=14)
    img2_overlay = axes[1].imshow(X_img, cmap="viridis", origin="lower", alpha=1, vmin=vmin, vmax=vmax)
    color = "red"
    y, x = divmod(torch.argmax(y_pred_percent).item(), HYPER_PARAMETERS['WINDOW_SIZE'][1])
    axes[1].set_title(f'Predictions, Max={y_pred_percent.max():.2f} at ({x},{y})', color=color, fontsize=14)
    axes[1].plot(x, y, marker='*', color='red', markersize=15, linestyle='None', label='Model Prediction')
    axes[1].plot(4, 15, marker='>', color='deepskyblue', markersize=10, linestyle='None', markeredgewidth=2, markerfacecolor='none', label='Gas Source Location')
    axes[1].legend(loc='upper right')

    axes[0].set_ylabel(f"y (dm)", fontsize=14)
    for ax in axes:
        ax.label_outer()
        ax.set_ylabel(f"y (dm)", fontsize=14)
        ax.set_xlabel(f"x (dm)", fontsize=14)
        ax.label_outer()
    #cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.76])  # adjust as needed
    #fig.colorbar(img1, cax=cbar_ax).set_label('Intensity')

    #plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()



if __name__ == "__main__":
    main()