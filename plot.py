import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch


def main():
    # read in all files in folder
    path = "models/"
    list_of_files = os.listdir(path)
    all_models = []
    names = []
    for file in list_of_files:
        # Separate the name by _
        name = file.split("_")
        if name[3] == "5" and name[5] == "hist.pt":
            all_models.append(file)
            names.append(name[2])

    models = []
    for model in all_models:
        models.append(torch.load(path + model))
    
    xaxis = np.array([i for i in range(1, len(models[0]) + 1)])
    # model to numpy 

    # plot all models
    plt.figure()
    for model,name in zip(models,names):
        # toplot
        plt.plot(np.array(model))
    plt.legend(names)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    #plt.show()
    plt.savefig("retrain.png")






if __name__ == '__main__':
    main()