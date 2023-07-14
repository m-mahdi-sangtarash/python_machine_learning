from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

digits = load_digits()

fig = plt.figure(figsize=(6, 6))
plt.subplots_adjust(bottom=0, right=1, left=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary)
    ax.text(0, 7, str(digits.target[i]))
plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=0)

model = KMeans()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

fig = plt.figure(figsize=(6, 6))
plt.subplots_adjust(bottom=0, right=1, left=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(xtest.reshape(-1, 8, 8)[i], cmap=plt.cm.binary)

    if ypred[i] == ytest[i]:
        ax.text(0, 7, str(ypred[i]), color="green")
    else:
        ax.text(0, 7, str(ypred[i]), color="red")

plt.show()

print("_________Before PCA_________\n")
print("accuracy: ", accuracy_score(ytest, ypred))

pca = PCA(n_components=4)
proj = pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
plt.colorbar()
plt.show()

