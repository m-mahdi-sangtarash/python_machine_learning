import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

dataframe = pd.read_excel("Practice-44-1-data.xlsx")
x = dataframe[["Age", "Gender", "Salary_level", "Perfume_price"]]
y = dataframe["Purchase_status"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
ypredict = model.predict([[1, 26, 5, 250]])
print("Purchase status: ", ypredict)

plot_tree(model)
plt.show()