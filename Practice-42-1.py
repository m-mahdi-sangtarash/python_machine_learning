import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

dataframe = pd.read_excel("practice-42-1-data.xlsx")
x = dataframe[["Sex", "Age", "Smoke"]]
y = dataframe["Cancer"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
model = LogisticRegression()
model.fit(xtrain, ytrain)
ypredict = model.predict(xtest)

print(pd.DataFrame({"ytest": ytest, "ypredict": ypredict}))
print(confusion_matrix(ytest, ypredict))
print(classification_report(ytest, ypredict))

print(model.predict([[0, 45, 1]]))