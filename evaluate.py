import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import logs.logger as logger
import logging
import parameters


LOGS_SAVE = False
FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage","TestPositioning"]
WINDOW_SIZE=[400,400]

def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_evaluate")    
    df = load_csv(parameters.PARAMETERS[1],"test3")
    #plot_flightpath(df)
    plot_gdm(df)

def load_csv(flightpath, file=None):       
    if flightpath == "first":
        files = os.listdir("data")
        files = files[0]
        file_path=f"data/{files}"
    elif file:
        file_path=f"data/GSL_{flightpath}_0.3_{file}.csv"        
    else:
        file_path=f"data/GSL_{flightpath}_0.3.csv"
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
        #print(X.iloc[i], Y.iloc[i], Time.iloc[i])
        #if Gas1L.iloc[i]==1 or Gas1R.iloc[i]==1:
        #    imageGasL[X.iloc[i], Y.iloc[i]]=1/float(Gas1L.iloc[i]+0.1)            
        #    imageGasR[X.iloc[i], Y.iloc[i]]=1/float(Gas1R.iloc[i]+0.1)
        ##elif Gas1L.iloc[i]<3 or Gas1R.iloc[i]<3:
        #    #print(Gas1R.iloc[i],Gas1L.iloc[i])
        ##print(Gas1R.iloc[i],Gas1L.iloc[i])
        #else:
        #    imageGasL[X.iloc[i], Y.iloc[i]]=1/float(Gas1L.iloc[i])         
        #    imageGasR[X.iloc[i], Y.iloc[i]]=1/float(Gas1R.iloc[i])
            imageGasL[X.iloc[i], Y.iloc[i]]=Gas1L.iloc[i]       
            imageGasR[X.iloc[i], Y.iloc[i]]=Gas1R.iloc[i]

        #imageGasL[X.iloc[i], Y.iloc[i]]=Gas1L.max()/Gas1L.iloc[i]
    imageGasL=imageGasL.unsqueeze(-1)
    imageGasR=imageGasR.unsqueeze(-1)   
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(imageGasL, origin="lower")
    ax1.set_title('Gas Distribution Left')
    ax2.imshow(imageGasR, origin="lower")
    ax2.set_title('Gas Distribution Right')
    #fig,ax=plt.subplot(1, 2, 1)
    #fig.add_subplot
    #plt.imshow(imageGasL, cmap='turbo', origin="lower")
    #plt.imshow(imageGasR, cmap='turbo', origin="lower")

    #df.plot()
    plt.show()





def plot_flightpath(df, window_size=WINDOW_SIZE):
    logging.info("Plottting Flightpath.")    
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
    plt.title("Flightpath")
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
        #print(Z[i])
        if x==old_x and y == old_y:# or Z[i]<20:
            X_copy.drop(i, inplace=True)
            Y_copy.drop(i, inplace=True)
            Time.drop(i, inplace=True)
            Gas1L.drop(i, inplace=True)
            Gas1R.drop(i, inplace=True)
           # X.drop(i)
           #Y.drop(i)
           # Time.drop(i)
        else:        
            old_x,old_y=x,y         
    result = pd.DataFrame({'Time':Time,'X': X_copy, 'Y': Y_copy, 'Gas1L': Gas1L, 'Gas1R': Gas1R})
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