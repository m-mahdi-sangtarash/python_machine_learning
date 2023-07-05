import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

dataframe = pd.read_excel("practice-44-2-data.xlsx")
x = dataframe[["Gender", "Age", "Smoke"]]
y = dataframe["Cancer"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
ypredict = model.predict([[0, 46, 1]])
print("Cancer: ", ypredict)

plot_tree(model)
plt.show()