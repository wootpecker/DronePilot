
import numpy as np
import matplotlib.pyplot as plt
import math
import torch



def main():
   # test_snake()
    test_cage()

def test_snake():
    distances=14
    koordinaten=[]
    title=[]
    for x in range(2,distances):
        gdm=np.zeros([1,40,40])
        #print(gdm.shape)
        coordinates=generate_coordinates_s_shape(gdm.shape,distance=x,pad=2,start_left=True)
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
    coordinates_to_flightpath(coordinates)
    
    # Setup a plot 
    
def coordinates_to_flightpath(coordinates):
    temp=coordinates
    for value in temp:
            value[0]=(10*value[0]+5)/100
            value[1]=(10*value[1]+5)/100
            print(f"x: {value[0]}, y: {value[1]}")
    print(coordinates)
    #return coordinates

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


def generate_coordinates_s_shape_rotate(dataset_GDM,distance=1,pad=1,start_left=True):
    """
    Generates a list of coordinates in an S-shaped pattern within the given dataset dimensions.
    Args:
        dataset_GDM (list): A list containing the dimensions of the dataset.
        distance (int, optional): The vertical distance of each horizontal pass. Defaults to 3.
        pad (int, optional): The padding to apply to the edges of the dataset. Defaults to 2.
        start_left (bool, optional): Whether to start the pattern from the left side. Defaults to True.
    Returns:
        list: A list of [x, y] coordinates representing the S-shaped pattern.
    """
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    coordinates = []
    x,y=pad,pad
    while(x<=width-pad):
        #left to right
        if(start_left):
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

        start_left= not start_left
        x+=1
    #print(len(coordinates))
    return coordinates


def generate_coordinates_s_shape_rotate_up(dataset_GDM,distance=1,pad=1,start_left=True):
    """
    Generates a list of coordinates in an S-shaped pattern within the given dataset dimensions.
    Args:
        dataset_GDM (list): A list containing the dimensions of the dataset.
        distance (int, optional): The vertical distance of each horizontal pass. Defaults to 3.
        pad (int, optional): The padding to apply to the edges of the dataset. Defaults to 2.
        start_left (bool, optional): Whether to start the pattern from the left side. Defaults to True.
    Returns:
        list: A list of [x, y] coordinates representing the S-shaped pattern.
    """
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    coordinates = []
    x,y=width-pad,height-pad
    while(x>=pad):
        #right to left
        if(start_left):
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

        start_left= not start_left
        x-=1
    #print(len(coordinates))
    return coordinates



def generate_coordinates_s_shape(dataset_GDM,distance=1,pad=1,start_left=True):
    """
    Generates a list of coordinates in an S-shaped pattern within the given dataset dimensions.
    Args:
        dataset_GDM (list): A list containing the dimensions of the dataset.
        distance (int, optional): The vertical distance of each horizontal pass. Defaults to 3.
        pad (int, optional): The padding to apply to the edges of the dataset. Defaults to 2.
        start_left (bool, optional): Whether to start the pattern from the left side. Defaults to True.
    Returns:
        list: A list of [x, y] coordinates representing the S-shaped pattern.
    """
    width,height=dataset_GDM[-2]-1,dataset_GDM[-1]-1
    coordinates = []
    x,y=pad,pad
    while(y<=height-pad):
        #left to right
        if(start_left):
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

        start_left= not start_left
        y+=1
    #print(len(coordinates))
    return coordinates

def generate_coordinates_cage(dataset_GDM,distance=3,pad=2):
    print(f"cage, distance: {distance}")
    coordinatesfirst=generate_coordinates_s_shape(dataset_GDM,distance=distance,pad=pad,start_left=True)

    print(f"coordinatesfirst[-1][1]: {coordinatesfirst[-1][1]}")
    print(f"coordinatesfirst[-1][0]: {coordinatesfirst[-1][0]}")
    print(f"dataset_GDM[-1]/2:{dataset_GDM[-1]/2}")
    if coordinatesfirst[-1][0]>dataset_GDM[-1]/2:
        #print(distance)
        coordinatessecond=generate_coordinates_s_shape_rotate_up(dataset_GDM,distance=distance,pad=pad,start_left=True)
    else:
        coordinatessecond=generate_coordinates_s_shape_rotate(dataset_GDM,distance=distance,pad=pad,start_left=False)
    #coordinatessecond=[]
    coordinatesfirst.extend(coordinatessecond)
    return coordinatesfirst






if __name__ == '__main__':
    main()