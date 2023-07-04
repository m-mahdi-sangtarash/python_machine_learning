import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


dataframe = pd.read_excel("Practice-43-2-data.xlsx")
x = dataframe[["Gender", "Age", "Smoke"]]
y = dataframe["Cancer"]
error = []

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

for i in range (1, 20):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(xtrain, ytrain)
    ypredict = model.predict(xtest)
    error.append(np.mean(ypredict != ytest))

plt.plot(range(1, 20),error)
plt.xlabel("n_neighbors")
plt.ylabel("error")
plt.show()

model = KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)
ypredict = model.predict([[0, 45, 1]])
print(ypredict)