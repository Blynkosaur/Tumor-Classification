import numpy as np
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input):
        pass
    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self,input_size,output_size,name):
        self.name = name
        # Xavier/Glorot initialization for better training stability
        limit = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.normal(0, limit, (output_size, input_size))
        self.bias = np.zeros((output_size, 1))
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights,self.input) + self.bias
    def backward(self, output_gradient, learning_rate):
        # weights_gardient = np.dot(output_gradient,self.input.T)
        # self.weights -= learning_rate*weights_gardient
        # self.bias -= learning_rate * output_gradient
        # return np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Apply momentum
        momentum = 0.9
        self.weight_momentum = momentum * self.weight_momentum - learning_rate * weights_gradient
        self.bias_momentum = momentum * self.bias_momentum - learning_rate * output_gradient
        
        # Update weights and bias
        self.weights += self.weight_momentum
        self.bias += self.bias_momentum
        
        return input_gradient

class Activation(Layer):
    def __init__(self,activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient,self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self,name):
        self.name = name
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1- np.tanh(x) ** 2
        super().__init__(tanh,tanh_prime)
class Sigmoid(Activation):
    def __init__(self, name):
        self.name = name
        def sigmoid(x):
            # Clip x to prevent overflow
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s*(1-s)
        super().__init__(sigmoid,sigmoid_prime)
class ReLU(Activation):
    def __init__(self, name):
        self.name = name
        # ReLU function: returns x if x > 0, else returns 0
        relu = lambda x: np.maximum(0, x)
        
        # ReLU derivative: returns 1 if x > 0, else returns 0
        relu_prime = lambda x: np.where(x > 0, 1, 0)
        
        super().__init__(relu, relu_prime)
def mse(y_hat, y_true):
    return (np.mean(np.power((y_hat- y_true),2)))/2

def mse_prime(y_true,y_pred):
    return (y_pred-y_true)/np.size(y_true)

def predict(network,input,verbose = False):
    output = input
    for layer in network:
        if verbose:
            print(layer.name, output.shape)
        output = layer.forward(output)
    return output
def train(network,alpha,epochs,loss, loss_prime,x_train,y_train,prin = True):
    
    for time in range(epochs):
        error = 0 
        for x, y in zip(x_train,y_train):
            # print(x,y)
            prediction = predict(network,x)
            
            error += loss(y,prediction)

            #gradient descent 
            gradient = loss_prime(y, prediction)
            for layer in network[::-1]:
                gradient = layer.backward(gradient,alpha)
            
        error /= len(x)
        if prin:
            print(f"{time + 1}/{epochs}, error={error}")
def bce(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_prime(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * np.size(y_true))  # Fixed derivative


