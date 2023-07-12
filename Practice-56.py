from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

dataframe = load_wine()
x = dataframe.data
y = dataframe.target
target_names = dataframe.target_names

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)

print("_____________Before resacle_____________\n")
print("accuracy: ", accuracy_score(ytest, y_pred))

scale_x = StandardScaler().fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(scale_x, y, test_size=0.2)

model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)

print("\n_____________After resacle_____________\n")
print("accuracy: ", accuracy_score(ytest, y_pred))

