"""
transform_measurements.py

This module provides functions for transforming and visualizing measurement data collected by a nano-drone.
It includes utilities to convert measurement coordinates and gas concentrations for ML integration.

-----------------------------
Testing Parameters:
- LOGS_SAVE (bool): Whether to save logs.
- WINDOW_SIZE (list): Size of the grid for plotting.
- NAMES (list): List of dataset names.
- NAME_TO_USE (int): Index of the dataset to use for testing.

-----------------------------
Functions:
- main():
    Entry point for the script. Loads data and visualizes gas distribution.

- plot_gdm(df, window_size):
    Plots the gas distribution map for the provided DataFrame.

- transform_column(df):
    Transforms DataFrame columns for plotting and analysis.

- transform_m_cm(df_column):
    Converts meters to centimeters and casts to int.

- transform_m_dm(df_column):
    Converts meters to decimeters and casts to int.

- transform_gas_concentration(df_column):
    Normalizes gas concentration values.

-----------------------------
Dependencies:
- pandas, matplotlib, torch, numpy, logging
- Custom modules: logs.logger, utils

-----------------------------
Usage:
Run this script to visualize gas concentrations discretized for ML model input or import and call utilities in other scripts:
    python transform_measurements.py

"""

import pandas as pd
import matplotlib.pyplot as plt
import torch
import logs.logger as logger
import logging
import numpy as np
import utils

TESTING_PARAMETERS = {
               "LOGS_SAVE": False,
               "WINDOW_SIZE": [64,64],
               "NAMES": ["all", "A", "A_invers", "B", "B_invers","first","F_1"], # F_1 for testing purpose
               "NAME_TO_USE": 3,                                # Index of the dataset to use for testing, 0: all, 1: A, 2: A_invers, 3: B, 4: B_invers 5: first, 6: F_1
  }



def main():
    logger.logging_config(logs_save=TESTING_PARAMETERS["LOGS_SAVE"], filename="crazyflie_evaluate")    
    df = utils.load_csv(None,TESTING_PARAMETERS["NAMES"][TESTING_PARAMETERS["NAME_TO_USE"]])
    plot_gdm(df=df, window_size=TESTING_PARAMETERS["WINDOW_SIZE"])


    

def plot_gdm(df, window_size=TESTING_PARAMETERS["WINDOW_SIZE"]):
    df= df[0]
    logging.info("Plottting Gas Distribution.")    
    df_plot=transform_column(df=df)
    X=df_plot['X']
    Y=df_plot['Y']
    Gas1L = df_plot['Gas1L']
    Gas1R = df_plot['Gas1R']
    imageGasL=[X,Y,Gas1L]
    imageGasR=[X,Y,Gas1R]
    imageGasL = np.zeros((window_size[0], window_size[1]))
    imageGasR = np.zeros((window_size[0], window_size[1]))

    for i in range(len(Gas1L)):
        imageGasL[X[i], Y[i]] = Gas1L[i]
        imageGasR[X[i], Y[i]] = Gas1R[i]

    zeros=torch.zeros(size=Gas1R.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    img1= ax1.imshow(imageGasL, cmap="turbo", origin="lower", vmin=0, vmax=Gas1R.max())
    ax1.set_title('Gas Distribution Left')
    img2=ax2.imshow(imageGasR, cmap="turbo", origin="lower", vmin=0, vmax=Gas1R.max())
    ax2.set_title('Gas Distribution Right')
    im1 = ax1.imshow(imageGasL, cmap="turbo", origin="lower", vmin=0, vmax=Gas1R.max())
    plt.colorbar(img2, ax=ax2, label='Gas Concentration')
    #plt.colorbar(img2, ax=ax2)
    plt.show()
    return imageGasL, imageGasR

 
    

def transform_column(df):
    X=transform_m_dm(df_column=df['X'])
    Y=transform_m_dm(df_column=df['Y'])
    Time = df['Time']
    Gas1L=transform_gas_concentration(df_column=df['Gas1L'])
    Gas1R=transform_gas_concentration(df_column=df['Gas1R'])
    result = pd.DataFrame({'Time':Time,'X': X, 'Y': Y, 'Gas1L': Gas1L, 'Gas1R': Gas1R})
    result = result.groupby(['X', 'Y'])[['Gas1L', 'Gas1R']].mean().reset_index()
    return result


def transform_m_cm(df_column):
    df_column = df_column * 100
    #df_column = df_column+abs(df_column.min())
    df_column = df_column.astype(int)
    return df_column


def transform_m_dm(df_column):
    df_column = df_column * 10
    #df_column = df_column+abs(df_column.min())
    df_column = df_column.astype(int)
    return df_column

def transform_gas_concentration(df_column):
    df_column = df_column.max() - df_column
    return df_column




if __name__ == '__main__':
    main()