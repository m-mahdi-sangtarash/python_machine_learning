import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataframe = pd.read_excel("Practice-47-data.xlsx")

se = []

for i in range(1, 15):
    model = KMeans(n_clusters=i)
    model.fit(dataframe)
    se.append(model.inertia_)

plt.plot(range(1, 15), se)
plt.xlabel("n_cluster")
plt.ylabel("elbow")
plt.show()