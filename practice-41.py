import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_excel("house-p.xlsx")
x = data[["meter", "room", "floor", "year", "zone"]]
y = data["price"]

model = LinearRegression()
model.fit(x, y)
ypred = model.predict([[70, 1, 2, 17, 2]])
print(ypred)
