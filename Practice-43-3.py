import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataframe = pd.read_excel("practice-43-3-data.xlsx")
x = dataframe[["Read_per", "Review"]]
y = dataframe["Result"]
error_lst = []

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

for i in range(1, 27):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(xtrain, ytrain)
    ypredict = model.predict(xtest)
    error_lst.append(np.mean(ypredict != ytest))

plt.plot(range(1, 27), error_lst)
plt.xlabel("n_neighbors")
plt.ylabel("error")
plt.show()

model = KNeighborsClassifier(n_neighbors=4)
model.fit(xtrain, ytrain)
ypredict = model.predict([[40, 2]])
print(ypredict)