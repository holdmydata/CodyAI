## Lazy start to the evaluation script ##
## This script will be used to evaluate the model automatically after training run. ##

## To Do:
## - Finish the automatic evaluation script
## - Add more evaluation metrics?


import pandas as pd

file = "GPT/validation/losses.csv"

data = pd.read_csv(file)
## create graph of losses
import matplotlib.pyplot as plt
import numpy as np

print(data.head(5))

def remove_tensor(data):
    
    for i in range(len(data)):
        #strip 'tensor()' from the string
        data['train_losses'][i] = data['train_losses'][i][7:-1]
        data['val_losses'][i] = data['val_losses'][i][7:-1]
        #remove blank rows
        if data['train_losses'][i] == '':
            data = data.drop(i)
    data.to_csv(file, index=False)
    return data

#save the data back to the csv file
#data.to_csv(file, index=False)

def plot_loss(data):
    plt.plot(data['epoch'], data['train_losses'], label='train_loss')
    plt.plot(data['epoch'], data['val_losses'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('Log Loss')
    plt.title(f'Model Run - {data["datetime"][0]}, epochs = 5000, learn_rate = {data["learning_rate"][0]}, batch_size = {data["batch_size"][0]}, block_size = {data["block_size"][0]}, num_heads = {data["num_heads"][0]}, num_layers = {data["num_layers"][0]}',wrap=True)
    plt.legend()
    plt.show()


#remove_tensor(data)
plot_loss(data)
