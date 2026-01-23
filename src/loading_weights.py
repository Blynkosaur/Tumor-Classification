import numpy as np
import pandas as pd
from neuralnetwork import Dense

def load_weights(network):
    for layer in network:
        if isinstance(layer,Dense):
            #weights
            filename = f"Model_Weights/{layer.name}_weights.csv"
            data = pd.read_csv(filename, header=None).values
            weight = np.array(data)
            layer.weights = weight
            print("Loaded weights:", layer.name)
            #bias
            filename = f"Model_Weights/{layer.name}_bias.csv"
            data = pd.read_csv(filename, header=None).values
            bias = np.array(data)
            layer.bias = bias
            print("Loaded bias:", layer.name)
