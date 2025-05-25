import torch
import matplotlib.pyplot as plt
import flightpaths
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
import utils
import logging
from logs import logger


HYPER_PARAMETERS = {
                "SAVE_DATASET": False,
                "TRANSFORM": False,
                "MODEL_TYPES": ["VGG8", "UnetS"],
                "LOGS_SAVE": False,
                "AMOUNT_SAMPLES": 16,
                "WINDOW_SIZE": [64,64]
  }

TESTING_PARAMETERS = {
                "MODEL_TO_TEST": 0,
}

def main():
    logger.logging_config(logs_save=HYPER_PARAMETERS['LOGS_SAVE'], filename="crazyflie_predictions")
    model_type = HYPER_PARAMETERS['MODEL_TYPES'][TESTING_PARAMETERS['MODEL_TO_TEST']]
    plot_prediction_A(model_type)
    plot_prediction_B(model_type)
    #plot_predictions2(model_type)


def plot_prediction_A(model_type):
    flight_path = flightpaths.flightpath_to_coordinates("Snake",[20,20],4,1)
    X=torch.zeros((HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1]))
    flight_path=[[int(x * 10),int(y*10)] for x,y in flight_path]
    flight_path=np.array(flight_path)
    source_location_b=[9, 15]
    source_location_a=[4, 15]

    # Calculate Euclidean distance
    pred_a, pred_a_inv, pred_b, pred_b_inv, benchmark_a, benchmark_a_inv, benchmark_b, benchmark_b_inv = utils.get_results(model_type)
    avg_euclidean_distance = calculate_euclidean_distance(pred_a, pred_a_inv, pred_b, pred_b_inv, source_location_a, source_location_b)
    benchmark_euclidean_distance = calculate_euclidean_distance(benchmark_a, benchmark_a_inv, benchmark_b, benchmark_b_inv, source_location_a, source_location_b)
    logging.info(f"Average Euclidean distance: {avg_euclidean_distance}")
    logging.info(f"Benchmark Euclidean distance: {benchmark_euclidean_distance}")


    fig, axes = plt.subplots(1, 1, figsize=(7, 7),tight_layout=True) #tight_layout=True)
    X_img=X[0:21,0:21]

    # Image A
    #axes.set_title('Position A', fontsize=16)
    img1= axes.imshow(X_img, cmap="viridis", origin="lower",alpha=0.9)
    img1_flightpath = axes.scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10, label='Intended Flight Path')
    source_location_1=axes.plot(source_location_a[0], source_location_a[1], marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    wind_direction_1_right=axes.annotate('', xy=(source_location_a[0]+2,source_location_a[1]), xytext=(source_location_a[0], source_location_a[1]), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)
    wind_direction_1_left=axes.annotate('', xy=(source_location_a[0]-2,source_location_a[1]), xytext=(source_location_a[0], source_location_a[1]), arrowprops=dict(arrowstyle='->', color='lime', lw=2, mutation_scale=15), zorder=1)

 
    #Predictions
    added=[]
    for idx, pred in enumerate(pred_a):        
        if pred in added:
            source_1_text = axes.text(pred[0]-1, pred[1], f'{idx+1}', color='white', fontsize=10)
        else:
            
            source_1_text = axes.text(pred[0], pred[1], f'{idx+1}', color='white', fontsize=10)
            added.append(pred)
        if idx == 0:
            source_1 = axes.plot(pred[0], pred[1], marker='x', color='deepskyblue', markersize=10, linestyle='None', label='Model Prediction')
        else:
            source_1 = axes.plot(pred[0], pred[1], marker='x', color='deepskyblue', markersize=10, linestyle='None')
       
    
    # Inverse
    for idx, pred in enumerate(pred_a_inv):        
        if pred in added:
            if pred[0] == 0:
                source_1_text = axes.text(pred[0], pred[1]-1, f'{idx+1}', color='white', fontsize=10)
            else:
                if model_type == "UnetS":
                    source_1_text = axes.text(pred[0]-1, pred[1], f'{idx+1}', color='lime', fontsize=10)
                else:
                    source_1_text = axes.text(pred[0]-1, pred[1], f'{idx+1}', color='white', fontsize=10)
        else:
            source_1_text = axes.text(pred[0], pred[1], f'{idx+1}', color='white', fontsize=10)
            added.append(pred)
            source_1 = axes.plot(pred[0], pred[1], marker='x', color='lime', markersize=10, linestyle='None')

 #legend
    handles, labels = axes.get_legend_handles_labels()
    arrow = plt.arrow(0, 0, 0, 0, color='deepskyblue', label='Wind Direction')
    handles.append(arrow)
    labels.append('Wind Direction')
    axes.legend(loc='lower right',handles=handles, fontsize=14, labels=labels,handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),})

    
    axes.set_ylabel(f"y (dm)", fontsize=14)
    #for ax in axes:
        #ax.label_outer() 
    axes.set_ylabel(f"y (dm)", fontsize=14)
    axes.set_xlabel(f"x (dm)", fontsize=14)
    #axes.label_outer() 
    #plt.savefig(f"results/{model_type}_predictions.pdf", dpi=300)
    plt.show()





def plot_prediction_B(model_type):
    flight_path = flightpaths.flightpath_to_coordinates("Snake",[20,20],4,1)
    X=torch.zeros((HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1]))
    flight_path=[[int(x * 10),int(y*10)] for x,y in flight_path]
    flight_path=np.array(flight_path)
    source_location_b=[9, 15]
    source_location_a=[4, 15]

    # Calculate Euclidean distance
    pred_a, pred_a_inv, pred_b, pred_b_inv, benchmark_a, benchmark_a_inv, benchmark_b, benchmark_b_inv = utils.get_results(model_type)
    avg_euclidean_distance = calculate_euclidean_distance(pred_a, pred_a_inv, pred_b, pred_b_inv, source_location_a, source_location_b)
    benchmark_euclidean_distance = calculate_euclidean_distance(benchmark_a, benchmark_a_inv, benchmark_b, benchmark_b_inv, source_location_a, source_location_b)
    logging.info(f"Average Euclidean distance: {avg_euclidean_distance}")
    logging.info(f"Benchmark Euclidean distance: {benchmark_euclidean_distance}")

    fig, axes = plt.subplots(1, 1, figsize=(7, 7),tight_layout=True)
    X_img=X[0:21,0:21]

    # Image B
    img1= axes.imshow(X_img, cmap="viridis", origin="lower",alpha=0.9)
    img1_flightpath = axes.scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10, label='Intended Flight Path')
    source_location_1=axes.plot(source_location_b[0], source_location_b[1], marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    wind_direction_1_right=axes.annotate('', xy=(source_location_b[0]+2,source_location_b[1]), xytext=(source_location_b[0], source_location_b[1]), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)
    wind_direction_1_left=axes.annotate('', xy=(source_location_b[0]-2,source_location_b[1]), xytext=(source_location_b[0], source_location_b[1]), arrowprops=dict(arrowstyle='->', color='lime', lw=2, mutation_scale=15), zorder=1)

    #Predictions
    added=[]
    for idx, pred in enumerate(pred_b):        
        if pred in added:
            source_1_text = axes.text(pred[0]-1, pred[1], f'{idx+1}', color='white', fontsize=10)
        else:
            source_1_text = axes.text(pred[0], pred[1], f'{idx+1}', color='white', fontsize=10)
            added.append(pred)
        if idx == 0:
            source_1 = axes.plot(pred[0], pred[1], marker='x', color='deepskyblue', markersize=10, linestyle='None', label='Model Prediction')
        else:
            source_1 = axes.plot(pred[0], pred[1], marker='x', color='deepskyblue', markersize=10, linestyle='None')

    # Inverse
    for idx, pred in enumerate(pred_b_inv):        
        if pred in added:
            if pred[0] == 0:
                source_1_text = axes.text(pred[0], pred[1]-1, f'{idx+1}', color='white', fontsize=10)
            else:
                source_1_text = axes.text(pred[0]-1, pred[1], f'{idx+1}', color='white', fontsize=10)
        else:
            source_1_text = axes.text(pred[0], pred[1], f'{idx+1}', color='white', fontsize=10)
            added.append(pred)
            source_1 = axes.plot(pred[0], pred[1], marker='x', color='lime', markersize=10, linestyle='None')

    # legend
    handles, labels = axes.get_legend_handles_labels()
    arrow = plt.arrow(0, 0, 0, 0, color='deepskyblue', label='Wind Direction')
    handles.append(arrow)
    labels.append('Wind Direction')
    axes.legend(loc='lower right',handles=handles, fontsize=14, labels=labels,handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),})

    axes.set_ylabel(f"y (dm)", fontsize=14)
    axes.set_xlabel(f"x (dm)", fontsize=14)
    plt.show()

def plot_predictions2(model_type):
    flight_path = flightpaths.flightpath_to_coordinates("Snake",[20,20],4,1)
    X=torch.zeros((HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1]))
    flight_path=[[int(x * 10),int(y*10)] for x,y in flight_path]
    flight_path=np.array(flight_path)
    source_location_b=[9, 15]
    source_location_a=[4, 15]

    # Calculate Euclidean distance
    pred_a, pred_a_inv, pred_b, pred_b_inv, benchmark_a, benchmark_a_inv, benchmark_b, benchmark_b_inv = utils.get_results(model_type)
    avg_euclidean_distance = calculate_euclidean_distance(pred_a, pred_a_inv, pred_b, pred_b_inv, source_location_a, source_location_b)
    benchmark_euclidean_distance = calculate_euclidean_distance(benchmark_a, benchmark_a_inv, benchmark_b, benchmark_b_inv, source_location_a, source_location_b)
    logging.info(f"Average Euclidean distance: {avg_euclidean_distance}")
    logging.info(f"Benchmark Euclidean distance: {benchmark_euclidean_distance}")
   



    fig, axes = plt.subplots(1, 2, figsize=(10, 5),constrained_layout=True) #tight_layout=True)
    X_img=X[0:21,0:21]

    # Image A
    axes[0].set_title('Position A', fontsize=16)
    img1= axes[0].imshow(X_img, cmap="viridis", origin="lower",alpha=0.9)
    img1_flightpath = axes[0].scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10, label='Intended Flight Path')
    source_location_1=axes[0].plot(source_location_a[0], source_location_a[1], marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    wind_direction_1_right=axes[0].annotate('', xy=(source_location_a[0]+2,source_location_a[1]), xytext=(source_location_a[0], source_location_a[1]), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)
    wind_direction_1_left=axes[0].annotate('', xy=(source_location_a[0]-2,source_location_a[1]), xytext=(source_location_a[0], source_location_a[1]), arrowprops=dict(arrowstyle='->', color='lime', lw=2, mutation_scale=15), zorder=1)

 
    #Predictions
    added=[]
    for idx, pred in enumerate(pred_a):        
        if pred in added:

                source_1_text = axes[0].text(pred[0]-1, pred[1], f'{idx+1}', color='white', fontsize=10)
        else:
            source_1_text = axes[0].text(pred[0], pred[1], f'{idx+1}', color='white', fontsize=10)
            added.append(pred)
        source_1 = axes[0].plot(pred[0], pred[1], marker='x', color='deepskyblue', markersize=10, linestyle='None')
       
    
    # Inverse
    for idx, pred in enumerate(pred_a_inv):        
        if pred in added:
            if pred[0] == 0:
                source_1_text = axes[0].text(pred[0], pred[1]-1, f'{idx+1}', color='white', fontsize=10)
            else:
                if model_type == "UnetS":
                    source_1_text = axes[0].text(pred[0]-1, pred[1], f'{idx+1}', color='lime', fontsize=10)
                else:
                    source_1_text = axes[0].text(pred[0]-1, pred[1], f'{idx+1}', color='white', fontsize=10)
        else:
            source_1_text = axes[0].text(pred[0], pred[1], f'{idx+1}', color='white', fontsize=10)
            added.append(pred)
            source_1 = axes[0].plot(pred[0], pred[1], marker='x', color='lime', markersize=10, linestyle='None')





    # Image B
    axes[1].set_title('Position B', fontsize=16)    
    img2 = axes[1].imshow(X_img, cmap="viridis", origin="lower",alpha=0.9)
    img2_flightpath = axes[1].scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10, label='Intended Flight Path')
    source_location_2=axes[1].plot(source_location_b[0],source_location_b[1], marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    wind_direction_2_right=axes[1].annotate('', xy=(source_location_b[0]+2,source_location_b[1]), xytext=(source_location_b[0],source_location_b[1]), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)
    wind_direction_2_left=axes[1].annotate('', xy=(source_location_b[0]-2,source_location_b[1]), xytext=(source_location_b[0],source_location_b[1]), arrowprops=dict(arrowstyle='->', color='lime', lw=2, mutation_scale=15), zorder=1)

    #Predictions
    added=[]
    for idx, pred in enumerate(pred_b):        
        if pred in added:
            source_1_text = axes[1].text(pred[0]-1, pred[1], f'{idx+1}', color='white', fontsize=10)
        else:
            source_1_text = axes[1].text(pred[0], pred[1], f'{idx+1}', color='white', fontsize=10)
            added.append(pred)
        if idx == 0:
            source_1=axes[1].plot(pred[0], pred[1], marker='x', color='deepskyblue', markersize=10, linestyle='None',label='Model Prediction')
        else:    
            source_1 = axes[1].plot(pred[0], pred[1], marker='x', color='deepskyblue', markersize=10, linestyle='None')

    # Inverse
    for idx, pred in enumerate(pred_b_inv):          
        if pred in added:
            source_1_text = axes[1].text(pred[0]-1, pred[1], f'{idx+1}', color='white', fontsize=10)
        else:
            source_1_text = axes[1].text(pred[0], pred[1], f'{idx+1}', color='white', fontsize=10)
            added.append(pred)
        source_1 = axes[1].plot(pred[0], pred[1], marker='x', color='lime', markersize=10, linestyle='None')
            


    #legend
    handles, labels = axes[1].get_legend_handles_labels()
    arrow = plt.arrow(0, 0, 0, 0, color='deepskyblue', label='Wind Direction')
    handles.append(arrow)
    labels.append('Wind Direction')
    axes[1].legend(loc='lower right',handles=handles, fontsize=14, labels=labels,handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),})

    
    axes[0].set_ylabel(f"y (dm)", fontsize=14)
    for ax in axes:
        ax.label_outer() 
        ax.set_ylabel(f"y (dm)", fontsize=14)
        ax.set_xlabel(f"x (dm)", fontsize=14)
        ax.label_outer() 
    plt.savefig(f"results/{model_type}_predictions.pdf", dpi=300)
    plt.show()









def plot_predictions3():
    flight_path = flightpaths.flightpath_to_coordinates("Snake",[20,20],4,1)
    X=torch.zeros((HYPER_PARAMETERS['WINDOW_SIZE'][0],HYPER_PARAMETERS['WINDOW_SIZE'][1]))
    flight_path=[[int(x * 10),int(y*10)] for x,y in flight_path]
    flight_path=np.array(flight_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),constrained_layout=True) #tight_layout=True)
    source_location=[9, 15]
    wind_arrow=[7, 15]
    X_img=X[0:20,0:20]

    # Image A
    axes[0].set_title('Position A', fontsize=16)
    img1= axes[0].imshow(X_img, cmap="viridis", origin="lower",alpha=0.9)
    img1_flightpath = axes[0].scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10, label='Intended Flight Path')
    source_location_1=axes[0].plot(4, 15, marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    wind_direction_1_right=axes[0].annotate('', xy=(6, 15), xytext=(4, 15), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)
    wind_direction_1_left=axes[0].annotate('', xy=(2, 15), xytext=(4, 15), arrowprops=dict(arrowstyle='->', color='lime', lw=2, mutation_scale=15), zorder=1)

    #Predictions
    source_1=axes[0].plot(6, 13, marker='x', color='deepskyblue', markersize=10, linestyle='None',label='Model Prediction')
    source_1_text=axes[0].text(6, 13, '1', color='white', fontsize=10)
    source_2=axes[0].plot(6, 11, marker='x', color='deepskyblue', markersize=10, linestyle='None',label='Model Prediction')
    source_2_text=axes[0].text(6, 11, '2', color='white', fontsize=10)
    source_3=axes[0].plot(3, 16, marker='x', color='deepskyblue', markersize=10, linestyle='None',label='Model Prediction')
    source_3_text=axes[0].text(3, 16, '3', color='white', fontsize=10)    
    source_4=axes[0].plot(5, 14, marker='x', color='deepskyblue', markersize=10, linestyle='None',label='Model Prediction')
    source_4_text=axes[0].text(5, 14, '4', color='white', fontsize=10)
    # Inverse
    source_1=axes[0].plot(1, 16, marker='x', color='lime', markersize=10, linestyle='None',label='Model Prediction')
    source_1_text=axes[0].text(1, 16, '1', color='white', fontsize=10)    
    source_2=axes[0].plot(1, 14, marker='x', color='lime', markersize=10, linestyle='None',label='Model Prediction')
    source_2_text=axes[0].text(1, 14, '2', color='white', fontsize=10)




    # Image B
    axes[1].set_title('Position B', fontsize=16)    
    img2 = axes[1].imshow(X_img, cmap="viridis", origin="lower",alpha=0.9)
    img2_flightpath = axes[1].scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10, label='Intended Flight Path')
    source_location_2=axes[1].plot(source_location[0],source_location[1], marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)
    wind_direction_2_right=axes[1].annotate('', xy=(11, 15), xytext=(source_location[0],source_location[1]), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)
    wind_direction_2_left=axes[1].annotate('', xy=(wind_arrow[0],wind_arrow[1]), xytext=(source_location[0],source_location[1]), arrowprops=dict(arrowstyle='->', color='lime', lw=2, mutation_scale=15), zorder=1)
    
    #Predictions
    source_1=axes[1].plot(1, 16, marker='x', color='deepskyblue', markersize=10, linestyle='None',label='Model Prediction')
    source_1_pred=axes[1].text(1, 16, '1', color='white', fontsize=10)
    source_2=axes[1].plot(1, 14, marker='x', color='deepskyblue', markersize=10, linestyle='None')
    source_2_pred=axes[1].text(1, 14, '2', color='white', fontsize=10)    
    # Inverse
    source_1=axes[1].plot(4, 15, marker='x', color='lime', markersize=10, linestyle='None')
    source_1_text=axes[1].text(4, 15, '1', color='white', fontsize=10)  
    source_2=axes[1].plot(6, 16, marker='x', color='lime', markersize=10, linestyle='None')
    source_2_text=axes[1].text(6, 16, '2', color='white', fontsize=10)      



    # Image C
    axes[2].set_title('Position C', fontsize=16)    
    img3 = axes[2].imshow(X_img, cmap="viridis", origin="lower",alpha=0.9)
    img3_flightpath = axes[2].scatter(flight_path[:,1],flight_path[:,0], color="yellow",alpha=0.7, s=10, label='Intended Flight Path')
    source_location_3=axes[2].plot(9, 10, marker='D', color='red', markersize=8,linestyle='None', label='Source Location', zorder=2)

    #Predictions
    source_1=axes[2].plot(1, 11, marker='x', color='deepskyblue', markersize=10, linestyle='None',label='Model Prediction')
    source_1_pred=axes[2].text(1, 11, '1', color='white', fontsize=10)
    source_2=axes[2].plot(1, 14, marker='x', color='deepskyblue', markersize=10, linestyle='None')
    source_2_pred=axes[2].text(1, 14, '2', color='white', fontsize=10)    
    # Inverse
    source_1=axes[2].plot(4, 15, marker='x', color='lime', markersize=10, linestyle='None')
    source_1_text=axes[2].text(4, 15, '1', color='white', fontsize=10)  
    source_2=axes[2].plot(6, 16, marker='x', color='lime', markersize=10, linestyle='None')
    source_2_text=axes[2].text(6, 16, '2', color='white', fontsize=10)      

    wind_direction_2_right=axes[2].annotate('', xy=(11, 10), xytext=(9, 10), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2, mutation_scale=15), zorder=1)
    wind_direction_2_left=axes[2].annotate('', xy=(7, 10), xytext=(9, 10), arrowprops=dict(arrowstyle='->', color='lime', lw=2, mutation_scale=15), zorder=1)

    #legend
    handles, labels = axes[2].get_legend_handles_labels()

    arrow = plt.arrow(0, 0, 0, 0, color='deepskyblue', label='Wind Direction')
    handles.append(arrow)
    labels.append('Wind Direction')
    axes[2].legend(loc='lower right',handles=handles, labels=labels,handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),})


    
    axes[0].set_ylabel(f"y (dm)", fontsize=14)
    for ax in axes:
        ax.label_outer() 
        ax.set_ylabel(f"y (dm)", fontsize=14)
        ax.set_xlabel(f"x (dm)", fontsize=14)
        ax.label_outer() 
    plt.show()




def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p


def euclidean_distance(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def calculate_euclidean_distance(pred_a, pred_a_inv, pred_b, pred_b_inv, source_location_a, source_location_b):
    sum_dist_pred_a = sum(euclidean_distance(pred, source_location_a) for pred in pred_a)
    sum_dist_pred_a_inv = sum(euclidean_distance(pred, source_location_a) for pred in pred_a_inv)
    sum_dist_pred_b = sum(euclidean_distance(pred, source_location_b) for pred in pred_b)
    sum_dist_pred_b_inv = sum(euclidean_distance(pred, source_location_b) for pred in pred_b_inv)
    avg_euclidean_distance = sum([sum_dist_pred_a, sum_dist_pred_a_inv, sum_dist_pred_b, sum_dist_pred_b_inv])/ sum([len(pred_a),len(pred_a_inv),len(pred_b),len(pred_b_inv)])
    return avg_euclidean_distance



if __name__ == '__main__':
    main()