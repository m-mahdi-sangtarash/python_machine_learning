import pandas as pd
from sklearn.decomposition import PCA

dataframe = pd.read_excel("Practice-52-data.xlsx")

model = PCA(n_components=0.95)
x_pca = model.fit_transform(dataframe)
print(x_pca)
print("di: ", dataframe.shape[1])
print("df: ", x_pca.shape[1])