# DronePilot

DronePilot is a Python-based toolkit for generating flight paths and measuring gas concentrations for the Crazyflie nano-drone. Furthermore, it includes visualiziation and evaluations of these measurements. It provides utilities for creating various flight path patterns (such as S-shape and Cage) and integrates them with drone control.

## Main Features

- Drone piloting logic with multiple patterns and asynchronous data save([crazyflie_pilot.py](crazyflie_pilot.py))
- Evaluation and Vizualization of the measurements with machine learning model([ml_predictions.py](ml_predictions.py))
- Visualization of all taken flights for both model's prediction([plot_predictions.py](plot_predictions.py))
- Visualization of single flight in cm for evaluating flight time and gas measurements ([visualize.py](visualize.py))

## Helper

- Visualization of single flight in dm for evaluating ML model input ([transform_measurements.py](transform_measurements.py))
- Utility functions ([utils.py](utils.py))
- Crazyflie finder and initializer ([crazyflie_init.py](crazyflie_init.py))
- Flightpath creation with various patterns ([flightpaths.py](flightpaths.py))
- Model building ([model_builder.py](model_builder.py))
- Parameters for manual input ([parameter_input_console.py](parameter_input_console.py))

## Directory Structure

- `data/` - Data of gas measurements taken by the nano-drone
- `dynamic_plot/` - Helper scripts for dynamically plotting the drone's flightpath (has been removed due to limited capacity)
- `logs/` - Evaluation logs
- `model/` - Models to predict gas source location, create and add files from https://tubcloud.tu-berlin.de/s/yN3GjMwsJ8QRSom 
- `results/` - Plots of flightpath and experiment

## Getting Started

1. **Install dependencies**  <br/>
   Make sure you have Python 3.8+ installed. Install required packages:<br/>
   pip install -r requirements.txt<br/>
   My advice: install pytorch from https://pytorch.org/get-started/locally/ for enabling GPU

2. **Take gas measurements**<br/>
    Set testing parameters and take measurments

3. **Evaluate flight**<br/>
    Evaluate flight with:
    python visualize.py

4. **Make predictions**<br/>
    Evaluate flights with ML models and plot results:<br/>
    python ml_predictions.py

5. **Evaluate results**<br/>
    
    Evaluate experiment results by calculating avg. Euclidean distance and plotting with:<br/>
    python plot_predictions.py
