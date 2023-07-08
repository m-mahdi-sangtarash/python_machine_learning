import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

dataframe = pd.read_excel("Practice-48-data.xlsx")

neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(dataframe)
distance, indices = neighbors_fit.kneighbors(dataframe)

distance = np.sort(distance, axis=0)
distance = distance[:, -1]
plt.plot(distance)
plt.show()

model = DBSCAN(eps=15.41, min_samples=2)
model.fit(dataframe)
print(model.labels_)
