# %%
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("../data/breast-cancer.csv")

# %%
data.head()

# %%
data = data.drop(columns="id")

# %%
data.head()

# %%
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# %%
data.head()

# %%
mean_features = list(data.columns[1:11])
se_features = list(data.columns[11:21])
worst_features = list(data.columns[21:31])

# %%
se_features.append("diagnosis")
worst_features.append("diagnosis")

# %%
corr = data[worst_features].corr()

# %%
corr

# %%
prediction_vars = [
    "radius_mean",
    "perimeter_mean",
    "area_mean",
    "compactness_mean",
    "concave points_mean",
    "concavity_mean",
    "fractal_dimension_se",
    "radius_se",
    "radius_worst",
    "perimeter_worst",
    "area_worst",
    "concave points_worst",
]

# %% [markdown]
# # Model Training

# %%

train, test = train_test_split(data, test_size=0.15, random_state=1)

# %%
train_x = train[prediction_vars]
train_y = train["diagnosis"]
test_x = test[prediction_vars]
test_y = test["diagnosis"]

# %%


# %%

# %%
svm = SVC(kernel="rbf", C=600, random_state=1)
svm.fit(train_x, train_y)

# %%
yhat = svm.predict(test_x)

# %%
total = len(test_y)
print(total)

# %%
count = 0
test_y = list(test_y)


# %%
count = 0
for i in range(len(yhat)):
    if yhat[i] == test_y[i]:
        count += 1
print(count / 86)

# %%
print(precision_score(test_y, yhat))
print(recall_score(test_y, yhat))

# %%
