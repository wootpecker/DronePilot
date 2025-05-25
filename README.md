# DronePilot

DronePilot is a Python-based toolkit for generating, visualizing, and evaluating flight paths for drones. It provides utilities for creating various flight path patterns (such as S-shape and cage), plotting them, and integrating with drone control and evaluation workflows.

## Main Features
- Drone piloting logic with multiple patterns ([create_dataset.py](create_dataset.py))

- Generate S-shaped and cage flight paths with customizable parameters.
- Visualize flight paths using Matplotlib and PyTorch.
- Utilities for drone initialization, parameter management, and result plotting.
- Modular codebase for easy extension and integration.

## Project Structure
crazyflie_init.py # Drone initialization routines crazyflie_pilot.py # Drone piloting logic dynamic_plot.py # Dynamic plotting utilities evaluate.py # Evaluation of flight paths and predictions flightpaths.py # Flight path generation and visualization model_builder.py # Model building utilities parameters.py # Parameter management plotresults.py # Plotting results predictions.py # Prediction logic predictions_ML.py # Machine learning-based predictions utils.py # Utility functions data/ # Data files logs/ # Log files model/ # Saved models results/ # Results and outputs


## Main Features

- Main training class ([train_model.py](train_model.py))
- Prediction and evaluation ([predictions.py](predictions.py))

## Helper

- Data transformation and augmentation ([data_transformations.py](data_transformations.py), [data_transformations_with_adequate_input.py](data_transformations_with_adequate_input.py))
- Dataset creation ([create_dataset.py](create_dataset.py))
- Model building ([model_builder.py](model_builder.py))
- Custom data loaders ([model_dataloader.py](model_dataloader.py))
- Engine for training and testing step ([engine.py](engine.py), [engine_encdec.py](engine_encdec.py))
- Utility functions ([utils.py](utils.py))

## Directory Structure

- `data/` - Raw and processed datasets, add files from https://tubcloud.tu-berlin.de/s/yN3GjMwsJ8QRSom
- `helper/` - Helper scripts for creating dataset
- `logs/` - Training and evaluation logs
- `model/` - Saved models and checkpoints
- `plots/` - Scripts for plots of data augmentation visualization
- `results/` - Prediction results and evaluation metrics

## Getting Started

1. **Install dependencies**  <br/>
   Make sure you have Python 3.8+ installed. Install required packages:<br/>
   pip install -r requirements.txt<br/>
   My advice: install pytorch from https://pytorch.org/get-started/locally/ for enabling GPU

2. **Prepare the dataset**<br/>
    Place your raw data in the DroneSearch/data directory (both original and datasets_tensor)<br/>
    Skip to training or run:<br/>
    python create_dataset.py

3. **Train the model**<br/>
    Train model with:<br/>
    python train_model.py<br/>
    Alternatively, place one or more CPU or GPU models into the DroneSearch/model directory (while retaining the folder structure) to bypass the training phase and proceed to make predictions

4. **Make predictions**
    Choose parameters or use default values by running:<br/>
    python predictions.py
