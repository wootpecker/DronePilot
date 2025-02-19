import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import logs.logger as logger
import logging
import evaluate
import matplotlib.animation as animation

LOGS_SAVE = False
FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage","TestPositioning"]

def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_pilot") 
    dynamic_plot()   
    
def dynamic_plot(flightpath="TestPositioning"):    
    df = evaluate.load_csv(flightpath)
    plot_values(df)







def plot_values(df, window_size=[100,100]):
    logging.info("Plottting.")    
    df_plot=transform_column(df=df)
    X=df_plot['X']
    Y=df_plot['Y']
    Time=df_plot['Time']
    image=torch.Tensor(size=window_size)
    for i in range(len(X)):
        image[X.iloc[i], Y.iloc[i]]=Time.iloc[i]
    image.unsqueeze(-1)        
    plt.imshow(image)
    #df.plot()
    plt.show()

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





csv_file = "logconfig.csv"
# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlabel("Time")
ax.set_ylabel("Log Value")
ax.set_title("Live Log Plot")

# Initialize an empty line object
(line,) = ax.plot([], [], "b-", label="Log Data")

def read_csv():
    """Reads the latest data from the CSV file."""
    try:
        df = pd.read_csv(csv_file)  # Read the CSV file
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
def update(frame):
    """Updates the plot with new data."""
    df = read_csv()
    if df is not None and not df.empty:
        ax.clear()
        
        # Assuming the CSV has 'time' and 'log_value' columns
        ax.plot(df["time"], df["log_value"], "b-")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Log Value")
        ax.set_title("Live Log Plot")
        ax.legend(["Log Data"])

# Use FuncAnimation to update the plot every 100 ms
ani = animation.FuncAnimation(fig, update, interval=100)

plt.show()