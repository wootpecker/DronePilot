import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import logs.logger as logger
import logging
import parameters


LOGS_SAVE = True
FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage","TestPositioning"]

def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_evaluate")    
    df = load_csv(parameters.PARAMETERS[1])
    plot_values(df)

def load_csv(file):       
    if file == "first":
        files = os.listdir("data")
        files = files[0]
        file_path=f"data/{files}"
    else:
        file_path=f"data/GSL_{file}_0.3.csv"
    logging.info(f"[DATA] Loading CSV from: {file_path}.")         
    try:    
        df = pd.read_csv(file_path)
        logging.info(f"[DATA] Columns: {', '.join(df.columns)}")
        return df
    except:
        logging.error(f"[DATA] Could not load file: {file_path}")
        return None
    





def plot_values(df, window_size=[300,300]):
    logging.info("Plottting.")    
    df_plot=transform_column(df=df)
    X=df_plot['X']
    Y=df_plot['Y']
    Time=df_plot['Time']
    image=torch.Tensor(size=window_size)
    for i in range(len(X)):
        #print(X.iloc[i], Y.iloc[i], Time.iloc[i])
        image[X.iloc[i], Y.iloc[i]]=Time.iloc[i]
    image.unsqueeze(-1)        
    plt.imshow(image, cmap='turbo', origin="lower")
    #df.plot()
    plt.show()

def transform_column(df):
    X=transform_m_cm(df_column=df['X'])
    Y=transform_m_cm(df_column=df['Y'])
    Time = df['Time']

    old_x=-1
    old_y=-1

    X_copy=X.copy()
    Y_copy=Y.copy()
    for i, x in enumerate(X):
        y = Y[i]
        if x==old_x and y == old_y:
            X_copy.drop(i, inplace=True)
            Y_copy.drop(i, inplace=True)
            Time.drop(i, inplace=True)
           # X.drop(i)
           #Y.drop(i)
           # Time.drop(i)
        else:        
            old_x,old_y=x,y         
    result = pd.DataFrame({'X': X_copy, 'Y': Y_copy, 'Time':Time})
    #result['X'].min
    return result

def transform_m_cm(df_column):
    if df_column.max() < 10:
        df_column = df_column * 100
        df_column = df_column+abs(df_column.min())
    df_column = df_column.astype(int)
    return df_column


if __name__ == '__main__':
    main()