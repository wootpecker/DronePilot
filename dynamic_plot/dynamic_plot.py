import pandas as pd
import matplotlib.pyplot as plt
import torch
import logs.logger as logger
import logging
import visualize
import matplotlib.animation as animation
import parameter_input_console

LOGS_SAVE = False
FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage","TestPositioning"]
PARAMETERS=[0.3,"Snake",3]#height,flightpath,distance

def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_pilot") 
    dynamic_plot(parameter_input_console.PARAMETERS[1],window_size=[300,300])   
    
def dynamic_plot(flightpath="TestPositioning",window_size=[100,100]):  
    logger.logging_config(logs_save=LOGS_SAVE)  
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], "b-", label="Log Data")
    ani = animation.FuncAnimation(fig, update_plot, fargs=(fig, ax, flightpath, window_size), interval=200)
    plt.show()








def update_plot(frame, fig, ax, flightpath="TestPositioning", window_size=[200,200]):
    df = visualize.load_csv(flightpath)
    if df is not None and not df.empty:
        logging.info("Plottting.")  
        ax.clear()
        df_plot=transform_column(df=df)
        X=df_plot['X']
        Y=df_plot['Y']
        Time=df_plot['Time']
        image=torch.Tensor(size=window_size)
        for i in range(len(X)):
            image[X.iloc[i], Y.iloc[i]]=Time.iloc[i]
        image.unsqueeze(-1)
        im = ax.imshow(image, cmap='turbo', origin="lower")
        ax.set_xlabel("X in cm")
        ax.set_ylabel("Y in cm")
        ax.set_title(f"Crazyflie Flightpath: {flightpath}")


def transform_column(df):
    X=transform_m_mm(df_column=df['X'])
    Y=transform_m_mm(df_column=df['Y'])
    Time = df['Time']
    result = pd.DataFrame({'X': X, 'Y': Y, 'Time':Time})
    return result

def transform_m_mm(df_column):
    if df_column.max() < 10:
        df_column = df_column * 100
        df_column = df_column+abs(df_column.min())
    df_column = df_column.astype(int)
    return df_column


if __name__ == '__main__':
    main()
