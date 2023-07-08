import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import OPTICS


dataframe = pd.read_excel("Practice-49-data.xlsx")
data = dataframe.iloc[:, [3, 4]].values
model = OPTICS(min_samples=4)
labels = model.fit_predict(data)
print(labels)

plt.scatter(data[labels == -1, 0], data[labels == -1, 1], s=20, c="red")
plt.scatter(data[labels == 1, 0], data[labels == 1, 1], s=20, c="blue")
plt.scatter(data[labels == 2, 0], data[labels == 2, 1], s=20, c="green")
plt.scatter(data[labels == 3, 0], data[labels == 3, 1], s=20, c="yellow")
plt.scatter(data[labels == 4, 0], data[labels == 4, 1], s=20, c="purple")
plt.scatter(data[labels == 5, 0], data[labels == 5, 1], s=20, c="orange")
plt.scatter(data[labels == 6, 0], data[labels == 6, 1], s=20, c="gray")
plt.scatter(data[labels == 7, 0], data[labels == 7, 1], s=20, c="black")
plt.scatter(data[labels == 8, 0], data[labels == 8, 1], s=20, c="brown")
plt.scatter(data[labels == 9, 0], data[labels == 9, 1], s=10, c="yellow")
plt.scatter(data[labels == 10, 0], data[labels == 10, 1], s=10, c="purple")
plt.scatter(data[labels == 11, 0], data[labels == 11, 1], s=10, c="orange")
plt.scatter(data[labels == 12, 0], data[labels == 12, 1], s=10, c="gray")
plt.scatter(data[labels == 13, 0], data[labels == 13, 1], s=10, c="purple")
plt.scatter(data[labels == 14, 0], data[labels == 14, 1], s=10, c="orange")
plt.scatter(data[labels == 15, 0], data[labels == 15, 1], s=10, c="gray")
plt.show()
