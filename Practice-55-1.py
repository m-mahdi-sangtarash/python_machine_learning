from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


(x, y) = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=11)
model.fit(x_train, y_train)

print("_____________Before optimization_____________\n")
print("train accuracy: ", accuracy_score(y_train, model.predict(x_train)))
print("test accuracy: ", accuracy_score(y_test, model.predict(x_test)))

scale_x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2).fit_transform(scale_x)

x_train, x_test, y_train, y_test = train_test_split(scale_x, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=11)
model.fit(x_train, y_train)

print("\n_____________After optimization_____________\n")
print("train accuracy: ", accuracy_score(y_train, model.predict(x_train)))
print("test accuracy: ", accuracy_score(y_test, model.predict(x_test)))