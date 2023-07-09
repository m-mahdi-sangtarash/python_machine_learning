from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


(x, y) = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = KMeans(n_clusters=4)
model.fit(x_train, y_train)
error = model.inertia_

print("_____________Before optimization_____________\n")
print("train accuracy: ", accuracy_score(y_train, model.predict(x_train)))
print("test accuracy: ", accuracy_score(y_test, model.predict(x_test)))

scale_x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2).fit_transform(scale_x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = KMeans(n_clusters=4)
model.fit(x_train, y_train)


print("_____________After optimization_____________\n")
print("train accuracy: ", accuracy_score(y_train, model.predict(x_train)))
print("test accuracy: ", accuracy_score(y_test, model.predict(x_test)))