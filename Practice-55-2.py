from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

(x, y) = load_breast_cancer(return_X_y=True)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

print("_____________Before optimization_____________\n")
print("train accuracy: ", accuracy_score(ytrain, model.predict(xtrain)))
print("test accuracy: ", accuracy_score(ytest, model.predict(xtest)))

scale_x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2).fit_transform(scale_x)

xtrain, xtest, ytrain, ytest = train_test_split(scale_x, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

print("_____________After optimization_____________\n")
print("train accuracy: ", accuracy_score(ytrain, model.predict(xtrain)))
print("test accuracy: ", accuracy_score(ytest, model.predict(xtest)))

