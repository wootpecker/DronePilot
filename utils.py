"""
utils.py

This module provides utility functions for loading CSV data and retrieving benchmark and ML model results used for drone accuracy analysis.

-----------------------------
Functions:
- load_csv(flightpath, file=None):
    Loads CSV files from the data directory based on the specified flight path and file type.
    Supports loading all files, specific folders, or individual files.

- get_results(model_type):
    Returns prediction and benchmark results for different model types ("UnetS" or "VGG-8").

-----------------------------
Dependencies:
- os, logging, pandas

-----------------------------
Usage:
Import this module to use its data loading and result retrieval utilities in other scripts:
"""
import os
import logging
import pandas as pd



def load_csv(flightpath, file=None): 
    if file == "all":
        #files = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
        folders = [f for f in os.listdir("data") if os.path.isdir(os.path.join("data", f))]
        dfs = []
        for folder in folders:
            files = [f for f in os.listdir(os.path.join("data", folder)) if os.path.isfile(os.path.join("data", folder, f))]
            file_path = [os.path.join(f"data/{folder}", f) for f in files]
            
            for f in file_path:
                logging.info(f"[DATA] Loading CSV from: {f}.")
                try:    
                    df = pd.read_csv(f)
                    logging.info(f"[DATA] Columns: {', '.join(df.columns)}")
                    dfs.append(df)
                except:
                    logging.error(f"[DATA] Could not load file: {f}")
                    return None
        return dfs
    elif file == "A" or file == "A_invers" or file == "B" or file == "B_invers":
        files = [f for f in os.listdir(f"data/{file}") if os.path.isfile(os.path.join(f"data/{file}", f))]
        dfs = []

        file_path = [os.path.join(f"data/{file}", f) for f in files]
        
        for f in file_path:
            logging.info(f"[DATA] Loading CSV from: {f}.")
            try:    
                df = pd.read_csv(f)
                logging.info(f"[DATA] Columns: {', '.join(df.columns)}")
                dfs.append(df)
            except:
                logging.error(f"[DATA] Could not load file: {f}")
                return None
        return dfs
    elif file == "first":
        folders = [f for f in os.listdir("data") if os.path.isdir(os.path.join("data", f))]
        folder = folders[0]
        files = [f for f in os.listdir(os.path.join("data", folder)) if os.path.isfile(os.path.join("data", folder, f))]
        file = files[0]
        file_path = f"data/{folder}/{file}"
    elif file:
        file_path=f"data/GSL_{flightpath}_0.3_{file}.csv"        
    else:
        file_path=f"data/GSL_{flightpath}_0.3_C.csv"
    logging.info(f"[DATA] Loading CSV from: {file_path}.")         
    try:    
        df = pd.read_csv(file_path)
        logging.info(f"[DATA] Columns: {', '.join(df.columns)}")
        df = [df]
        return df
    except:
        logging.error(f"[DATA] Could not load file: {file_path}")
        return None
    




def get_results(model_type):
    if model_type == "UnetS":  
        logging.info("Loading U-NetS Results")      
        pred_a=[[5, 14], [5, 11], [7, 17], [7, 14], [3, 16], [4, 14]]
        pred_b=[[11, 16], [11, 18], [11, 20], [11, 16], [13, 15], [15, 17], [13, 12], [13, 16]]
        pred_a_inv=[[0, 17], [5, 18], [0, 16], [4, 19], [1, 16], [0, 14], [0, 16], [3, 16]]
        pred_b_inv=[[7, 13], [7, 15], [6, 11], [7, 12], [6, 12], [6, 16], [6, 12], [6, 14]]
    else:  
        logging.info("Loading VGG-8 Results")              
        pred_a=[[5, 14], [5, 11], [7, 17], [7, 14], [4, 16], [4, 14]]
        pred_b=[[11, 16], [10, 18], [12, 19], [11, 17], [13, 14], [15, 18], [13, 12], [13, 16]]
        pred_a_inv=[[1, 17], [5, 18], [1, 16], [4, 18], [1, 16], [2, 14], [1, 17], [2, 16]]
        pred_b_inv=[[7, 14], [7, 15], [7, 12], [7, 13], [6, 13], [7, 15], [6, 11], [6, 14]]   
    benchmark_a=[[5, 13], [5, 11], [6, 17], [7, 14], [3, 16], [4, 14]]
    benchmark_b=[[11, 16], [11, 18], [11, 19], [11, 16], [13, 15], [15, 17], [13, 11], [13, 15]]
    benchmark_a_inv=[[1, 17], [5, 18], [1, 16], [4, 19], [1, 16], [1, 14], [1, 16], [1, 15]]
    benchmark_b_inv=[[7, 14], [7, 15], [7, 12], [7, 13], [6, 13], [7, 16], [6, 12], [6, 14]]
    return pred_a, pred_a_inv, pred_b , pred_b_inv, benchmark_a, benchmark_a_inv, benchmark_b, benchmark_b_inv    