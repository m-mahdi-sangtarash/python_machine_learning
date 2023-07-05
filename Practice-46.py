import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataframe = np.array([[120, 50],
                      [60, 20],
                      [145, 65],
                      [130, 45],
                      [50, 15]])

#elbow

se = []

for i in range(1, 6):
    model = KMeans(n_clusters=i, max_iter=(300))
    model.fit(dataframe)
    se.append(model.inertia_)

plt.plot(range(1, 6), se)
plt.xlabel("n_clusters")
plt.ylabel("elbow")
plt.show()


#silhouette

silhouette_coefficients = []

for k in range(2, 5):
    model = KMeans(n_clusters=k)
    model.fit(dataframe)
    score = silhouette_score(dataframe, model.labels_)
    silhouette_coefficients.append(score)

plt.plot(range(2, 5), silhouette_coefficients)
plt.xticks(range(2, 5))
plt.xlabel("n_clusters")
plt.ylabel("silhouette")
plt.show()

