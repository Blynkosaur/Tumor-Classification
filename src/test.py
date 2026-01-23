import numpy as np
from neuralnetwork import predict
def test(network, test_x, test_y):
    total = len(test_x)
    correct = 0
    positive = 0  # actual positive cases
    predicted_positive = 0  # predicted positive cases
    true_positive = 0  # correctly predicted positive cases
    print(total)
    # Process all samples first
    for i in range(len(test_x)):
        
        y_hat = predict(network, test_x[i])
        # print(y_hat[0][0])
        # Count actual positives
        if test_y[i][0][0] == 1:
            
            positive += 1
            
        # Convert prediction to binary
        if y_hat >= 0.5:
            y_hat = 1
            predicted_positive += 1
        else:
            y_hat = 0
            
        # Count correct predictions and true positives
        if y_hat == test_y[i][0][0]:
            correct += 1
            if y_hat == 1:
                true_positive += 1
        accuracy = correct / total
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0
        recall = true_positive / positive if positive > 0 else 0
    return accuracy, precision, recall