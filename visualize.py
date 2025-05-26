import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import logs.logger as logger
import logging
import parameter_input_console
import numpy as np
import utils

NAMES=["first","F_1","F","F_1.5s","F_60cm","F_fastGas","F_Gas","F_less10cm","F_NoGas"]
EXAMPLE=NAMES[0]
LOGS_SAVE = False
WINDOW_SIZE=[250,250]

def main():
    test_all()


def test_all():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_evaluate")    
    df = utils.load_csv(None,EXAMPLE)
    df = df[0]
    plot_flightpath(df)
    plot_gdm(df)    


def plot_gdm(df, window_size=WINDOW_SIZE):
    logging.info("Plottting Gas Distribution.")    
    df_plot=transform_column(df=df)
    X=df_plot['X']
    Y=df_plot['Y']
    Time=df_plot['Time']
    Gas1L = df_plot['Gas1L']
    Gas1R = df_plot['Gas1R']
    Gas1L=Gas1L-Gas1L.min()+1
    Gas1R=Gas1R-Gas1R.min()+1
    Gas1L=Gas1L.max()-Gas1L
    Gas1R=Gas1R.max()-Gas1R
    imageGasL=torch.FloatTensor(size=window_size)
    imageGasR=torch.FloatTensor(size=window_size)

    for i in range(len(X)):

            imageGasL[X.iloc[i], Y.iloc[i]]=Gas1L.iloc[i]       
            imageGasR[X.iloc[i], Y.iloc[i]]=Gas1R.iloc[i]

    imageGasL=imageGasL.unsqueeze(-1)
    imageGasR=imageGasR.unsqueeze(-1)   
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imageGasL, cmap="turbo", origin="lower", vmin=0, vmax=Gas1R.max())
    ax1.set_title('Gas Distribution Left')
    ax2.imshow(imageGasR, cmap="turbo", origin="lower", vmin=0, vmax=Gas1R.max())
    ax2.set_title('Gas Distribution Right')
    plt.colorbar(ax2.imshow(imageGasL, cmap="turbo", origin="lower", vmin=0, vmax=Gas1R.max()), ax=ax2, label='Gas Concentration')
    plt.show()



def plot_flightpath(df, window_size=WINDOW_SIZE):
    logging.info("Plottting Flightpath.")    
    df_plot=transform_column(df=df)
    X=df_plot['X']
    Y=df_plot['Y']
   
    Time=df_plot['Time']
    image=torch.Tensor(size=window_size)
    for i in range(len(X)):
        image[X.iloc[i], Y.iloc[i]]=Time.iloc[i]
    image.unsqueeze(-1)     
    plt.imshow(image, cmap='turbo', origin="lower")#'turbo', origin="lower")
    plt.title("Flightpath")
    plt.xlabel("X in cm")
    plt.ylabel("Y in cm")
    plt.colorbar(label='Time (ms)')
    
    plt.show()

def transform_column(df):
    X=transform_m_cm(df_column=df['X'])
    Y=transform_m_cm(df_column=df['Y'])
    Z=df['Z']*100
    Time = df['Time']
    Gas1L = df['Gas1L']
    Gas1R = df['Gas1R']

    old_x=-1
    old_y=-1

    X_copy=X.copy()
    Y_copy=Y.copy()
    for i, x in enumerate(X):
        y = Y[i]
        if x==old_x and y == old_y:# or Z[i]<20:
            X_copy.drop(i, inplace=True)
            Y_copy.drop(i, inplace=True)
            Time.drop(i, inplace=True)
            Gas1L.drop(i, inplace=True)
            Gas1R.drop(i, inplace=True)

        else:        
            old_x,old_y=x,y         
    result = pd.DataFrame({'Time':Time,'X': X_copy, 'Y': Y_copy, 'Gas1L': Gas1L, 'Gas1R': Gas1R})
    return result

def transform_m_cm(df_column):
    if df_column.max() < 10:
        df_column = df_column * 100
        if df_column.min() < 0:
            df_column = df_column+abs(df_column.min())
    df_column = df_column.astype(int)
    return df_column


if __name__ == '__main__':
    main()