"""
flightpaths.py

This module provides functions to generate and visualize various flight path patterns for drone navigation.
It includes S-shaped (snake) and cage patterns, as well as utilities for plotting and converting flight paths to coordinates.

-----------------------------
Constants:
- FLIGHTPATH (list): List of available flight paths for the drone.

-----------------------------
Functions:
- main(): Test entry point for visualizing snake and cage patterns.

- flightpath_to_coordinates(flightpath, window_shape, distance, pad, start_zero):
    Converts flight path names to coordinate lists in m, based on specified parameters.

- generate_coordinates_s_shape(dataset_GDM, distance, pad, start_zero):
    Generates coordinates in an S-shaped pattern with vertical passes, starting from the top or bottom.

- generate_coordinates_cage(dataset_GDM, distance, pad):
    Generates coordinates for a cage flight path by combining two S-shaped patterns.

- generate_coordinates_s_shape_rotate():
    Generates coordinates in an S-shaped pattern. Mainly used for the cage pattern.

- generate_coordinates_s_shape_rotate_up():
    Generates coordinates in an S-shaped pattern. Mainly used for the cage pattern.

- test_snake(): Generates and visualizes S-shaped flight paths with varying distances.

- test_cage(): Generates and visualizes cage flight paths with varying distances.

- plot_more_images():
    Plots multiple images in a grid layout.

-----------------------------
Dependencies:
- numpy, matplotlib, math, torch

-----------------------------
Usage:
Import this module and use flightpath_to_coordinates to create flight paths for drones or simulations.
Run this script directly to visualize example patterns, with:
python flightpaths.py

"""


import numpy as np
import matplotlib.pyplot as plt
import math
import torch


FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage","TestPositioning"]


def main():
   # test_snake()
    test_cage()

    
def flightpath_to_coordinates(flightpath,window_shape=[12,12],distance=5,pad=2,start_zero=True):
    if flightpath==FLIGHT_PATH[2]:
        coordinates=generate_coordinates_s_shape(window_shape,distance=distance,pad=pad,start_zero=start_zero)
    elif flightpath==FLIGHT_PATH[3]:
        coordinates=generate_coordinates_cage(window_shape,distance=distance,pad=pad)
    else:
        return

    for value in coordinates:
            value[0]=(10*value[0]+5)/100
            value[1]=(10*value[1]+5)/100
    return coordinates


def generate_coordinates_s_shape_rotate(dataset_GDM,distance=1,pad=1,start_zero=True):
    """
    Generates a list of coordinates in an S-shaped pattern within the given dataset dimensions.
    Args:
        dataset_GDM (list): A list containing the dimensions of the dataset.
        distance (int, optional): The vertical distance of each horizontal pass. Defaults to 3.
        pad (int, optional): The padding to apply to the edges of the dataset. Defaults to 2.
        start_zero (bool, optional): Whether to start the pattern from the left side. Defaults to True.
    Returns:
        list: A list of [x, y] coordinates representing the S-shaped pattern.
    """
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    coordinates = []
    x,y=pad,pad
    while(x<=width-pad):
        #left to right
        if(start_zero):
            while(y<=height-pad):
                coordinates.append([x,y])
                y+=1
            y-=1
        else:
            y=height-pad
            while(y>=pad):                
                coordinates.append([x,y])
                y-=1
            y+=1
        x_max = min(x + distance, width - pad)        
        while(x<x_max):
            x+=1
            coordinates.append([x,y])

        start_zero= not start_zero
        x+=1
    #print(len(coordinates))
    return coordinates


def generate_coordinates_s_shape_rotate_up(dataset_GDM,distance=1,pad=1,start_zero=True):
    """
    Generates a list of coordinates in an S-shaped pattern within the given dataset dimensions.
    Args:
        dataset_GDM (list): A list containing the dimensions of the dataset.
        distance (int, optional): The vertical distance of each horizontal pass. Defaults to 3.
        pad (int, optional): The padding to apply to the edges of the dataset. Defaults to 2.
        start_zero (bool, optional): Whether to start the pattern from the left side. Defaults to True.
    Returns:
        list: A list of [x, y] coordinates representing the S-shaped pattern.
    """
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    coordinates = []
    x,y=width-pad,height-pad
    while(x>=pad):
        #right to left
        if(start_zero):
            while(y>=pad):
                coordinates.append([x,y])
                y-=1
            y+=1
        else:
            y=pad
            while(y<=height-pad):                
                coordinates.append([x,y])
                y+=1
            y-=1
        x_min = max(x - distance, pad)        
        while(x>x_min):
            x-=1
            coordinates.append([x,y])

        start_zero= not start_zero
        x-=1
    #print(len(coordinates))
    return coordinates



def generate_coordinates_s_shape(dataset_GDM,distance=1,pad=1,start_zero=True):
    """
    Generates a list of coordinates in an S-shaped pattern within the given dataset dimensions.
    Args:
        dataset_GDM (list): A list containing the dimensions of the dataset.
        distance (int, optional): The vertical distance of each horizontal pass. Defaults to 3.
        pad (int, optional): The padding to apply to the edges of the dataset. Defaults to 2.
        start_zero (bool, optional): Whether to start the pattern from the left side. Defaults to True.
    Returns:
        list: A list of [x, y] coordinates representing the S-shaped pattern.
    """
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    coordinates = []
    x,y=pad,pad
    while(y<=height-pad):
        #left to right
        if(start_zero):
            while(x<=width-pad):
                coordinates.append([x,y])
                x+=1
            x-=1
        else:
            x=width-pad
            while(x>=pad):                
                coordinates.append([x,y])
                x-=1
            x+=1
        y_max = min(y + distance, height - pad)        
        while(y<y_max):
            y+=1
            coordinates.append([x,y])

        start_zero= not start_zero
        y+=1
    #print(len(coordinates))
    return coordinates

def generate_coordinates_cage(dataset_GDM,distance=3,pad=2):
    coordinatesfirst=generate_coordinates_s_shape(dataset_GDM,distance=distance,pad=pad,start_zero=True)
    if coordinatesfirst[-1][0]>dataset_GDM[-1]/2:
        coordinatessecond=generate_coordinates_s_shape_rotate_up(dataset_GDM,distance=distance,pad=pad,start_zero=True)
    else:
        coordinatessecond=generate_coordinates_s_shape_rotate(dataset_GDM,distance=distance,pad=pad,start_zero=False)
    if(coordinatessecond[0][0]==coordinatesfirst[len(coordinatesfirst)-1][0] and coordinatessecond[0][1]==coordinatesfirst[len(coordinatesfirst)-1][1]):
        coordinatessecond.pop(0)
    coordinatesfirst.extend(coordinatessecond)
    return coordinatesfirst



def test_snake():
    distances=14
    koordinaten=[]
    title=[]
    for x in range(2,distances):
        gdm=np.zeros([1,40,40])
        #print(gdm.shape)
        coordinates=generate_coordinates_s_shape(gdm.shape,distance=x,pad=2,start_zero=True)
        #print(coordinates)
        values=np.arange(0,len(coordinates))
        for i, coordinate in enumerate(coordinates):
            gdm[0,coordinate[0], coordinate[1]] = values[i]
        
        koordinaten.append(gdm)
        title.append(f"Distance: {x}")
    
    plot_more_images(torch.tensor(np.array(koordinaten)),title=title)

def test_cage():
    distances=14
    koordinaten=[]
    title=[]
    for x in range(2,distances):
        gdm=np.zeros([1,40,40])
        #print(gdm.shape)
        coordinates=generate_coordinates_cage(gdm.shape,distance=x,pad=2)
        
        #print(coordinates)
        values=np.arange(0,len(coordinates))
        for i, coordinate in enumerate(coordinates):
            gdm[0,coordinate[0], coordinate[1]] = values[i]
        
        koordinaten.append(gdm)
        title.append(f"Distance: {x}")
    
    plot_more_images(torch.tensor(np.array(koordinaten)),title=title)
    


def plot_more_images(images, title="", save=False):
    """
    Plots multiple images.
    
    Args:
        images (Tensor): The images to plot.
        title (str): The title of the plot.
    """
    images=images.squeeze().unsqueeze(-1)

    if(len(images)<6):
      fig, axes = plt.subplots(len(images), 1, figsize=(18, 18))
    else:      
      fig, axes = plt.subplots(4, math.ceil(len(images)/4), figsize=(18, 18))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
          break
        ax.imshow(images[i], cmap='turbo',origin="lower")        
        #ax.axis('off')
        plt.suptitle(title[i])
    plt.tight_layout()
    plt.show()







if __name__ == '__main__':
    main()