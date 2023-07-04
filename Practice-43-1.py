import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


dataframe = pd.read_excel("Practice-43-1-data.xlsx")
x = dataframe[["Age", "Gender", "Salary_level", "Perfume_price"]]
y = dataframe["Purchase_status"]
error = []

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

for i in range(1, 22):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(xtrain, ytrain)
    ypredict = model.predict(xtest)
    error.append(np.mean(ypredict != ytest))

plt.plot(range(1, 22), error)
plt.xlabel("n_neighbors")
plt.ylabel("error")
plt.show()
model = KNeighborsClassifier(n_neighbors=17)
model.fit(xtrain, ytrain)
ypredict = model.predict([[1, 26, 5, 250]])
print(ypredict)
