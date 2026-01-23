import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path,features, test_size):
    data = pd.read_csv(file_path)
    data['diagnosis'] = data['diagnosis'].map({"M":1,"B":0})
    train,test = train_test_split(data, test_size= test_size, random_state= 42, shuffle= True)
    train_x = train[features]
    train_y = train['diagnosis']
    test_x = test[features]
    test_y = test['diagnosis']
    
    return train_x,train_y,test_x,test_y, len(train_x)
def standardize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std