import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import logs.logger as logger
import logging
import parameters
import numpy as np



LOGS_SAVE = False
WINDOW_SIZE=[320,320]
NAMES=["F","F_1.5s","F_60cm","F_fastGas","F_Gas","F_less10cm","F_NoGas"]
EXAMPLE=NAMES[0]
TRANFORM_TO_CENTER=True


def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_evaluate")    
    df = load_csv(parameters.PARAMETERS[1],EXAMPLE)
    plot_gdm(df)

def load_csv(flightpath, file=None):       
    if flightpath == "first":
        files = os.listdir("data")
        files = files[0]
        file_path=f"data/{files}"
    elif file:
        file_path=f"data/GSL_{flightpath}_0.3_{file}.csv"        
    else:
        file_path=f"data/GSL_{flightpath}_0.3_C.csv"
    logging.info(f"[DATA] Loading CSV from: {file_path}.")         
    try:    
        df = pd.read_csv(file_path)
        logging.info(f"[DATA] Columns: {', '.join(df.columns)}")
        return df
    except:
        logging.error(f"[DATA] Could not load file: {file_path}")
        return None
    

def plot_gdm(df, window_size=WINDOW_SIZE):
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
    plt.colorbar(img1, ax=ax1)
    #plt.colorbar(img2, ax=ax2)
    plt.show()
    return imageGasL, imageGasR

 
    

def transform_column(df):
    X=transform_m_dm(df_column=df['X'])
    Y=transform_m_dm(df_column=df['Y'])
    Z=df['Z']*100
    Time = df['Time']
    Gas1L=transform_gas_concentration(df_column=df['Gas1L'])
    Gas1R=transform_gas_concentration(df_column=df['Gas1R'])
    result = pd.DataFrame({'Time':Time,'X': X, 'Y': Y, 'Gas1L': Gas1L, 'Gas1R': Gas1R})
    result = result.groupby(['X', 'Y'])[['Gas1L', 'Gas1R']].mean().reset_index()
    return result

    old_x=-1
    old_y=-1
    average_amount=0
    average_value_L=0
    average_value_R=0
    for i, x in enumerate(X):
        y = Y[i]
        if x==old_x and y == old_y:
            average_amount+=1
            average_value_L+=Gas1L[i]
            average_value_R+=Gas1R[i]
        else:        
            old_x,old_y=x,y
            if average_amount>0:
                average_value_L=average_value_L/average_amount
                average_value_R=average_value_R/average_amount
                result[x,y,0]=average_value_L
                result[x,y,1]=average_value_R
            else:
                result[x,y,0]=Gas1L[i]
                result[x,y,1]=Gas1R[i]
            average_value_L=0
            average_value_R=0
            average_amount=0
    #result = pd.DataFrame({'Time':Time,'X': X_copy, 'Y': Y_copy, 'Gas1L': Gas1L, 'Gas1R': Gas1R})
    #result['X'].min
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