import numpy as np
import pandas as pd
from neuralnetwork import Dense
def save(network):
    weights = {}
    bias = {}
    for layer in network:
        if isinstance(layer,Dense):
            weights[layer.name] = layer.weights
            bias[layer.name] = layer.bias
    print(weights)
    for weight in weights:
        #for weights
        dw = pd.DataFrame(weights[weight])
        filename = f"Model_Weights/{weight}_weights.csv"
        dw.to_csv(filename, index=False, header=False)  # Save without headers for simplicity
        print(f"Saved {weight} to {filename}")
        #for biases
        db = pd.DataFrame(bias[weight])
        filename = f"Model_Weights/{weight}_bias.csv"
        db.to_csv(filename, index=False, header=False)  # Save without headers for simplicity
        print(f"Saved {weight} to {filename}")