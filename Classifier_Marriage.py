import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


marriage_data = pd.read_csv("marriage.csv",names=None)
marriage_data.columns=["x"+str(i) for i in range(1, 56)]
marriage_data = marriage_data.rename({'x55': 'y'}, axis=1)
X = marriage_data.loc[:, marriage_data.columns != 'y']
y = marriage_data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


nb_model = GaussianNB(priors = None, var_smoothing = 1e-03).fit(X_train, y_train)
lr_model = LogisticRegression().fit(X_train, y_train)
knn_model = KNeighborsClassifier().fit(X_train,y_train)

y_pred_nb = nb_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

print("Naive Bayes Accuracy: ",accuracy_score(y_test, y_pred_nb))
print("Logistic Regression Accuracy: ",accuracy_score(y_test, y_pred_lr))
print("KNN Accuracy: ",accuracy_score(y_test, y_pred_knn))


pca_comp = PCA(n_components=2).fit_transform(X)

pca_X = pd.DataFrame(data = pca_comp, columns = ['PC1', 'PC2'])
pca_X_train, pca_X_test, pca_y_train, pca_y_test = train_test_split(pca_X, y, test_size=0.2, random_state=99)

pca_nb_model = GaussianNB(priors = None, var_smoothing = 1e-03).fit(pca_X_train, pca_y_train)
pca_lr_model = LogisticRegression().fit(pca_X_train, pca_y_train)
pca_knn_model = KNeighborsClassifier().fit(pca_X_train,pca_y_train)


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
h = .02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
x_min, x_max = pca_X.iloc[:, 0].min() - 1, pca_X.iloc[:, 0].max() + 1
y_min, y_max = pca_X.iloc[:, 1].min() - 1, pca_X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))


knn_Z = pca_knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
knn_Z = knn_Z.reshape(xx.shape)

nb_Z = pca_nb_model.predict(np.c_[xx.ravel(), yy.ravel()])
nb_Z = nb_Z.reshape(xx.shape)

lr_Z = pca_lr_model.predict(np.c_[xx.ravel(), yy.ravel()])
lr_Z = lr_Z.reshape(xx.shape)


plt.figure()
plt.pcolormesh(xx, yy, nb_Z, cmap=cmap_light)
plt.scatter(pca_X.iloc[:, 0], pca_X.iloc[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Naive Bayes")
plt.savefig("NB_Marriage.png")
plt.clf()


plt.figure()
plt.pcolormesh(xx, yy, lr_Z, cmap=cmap_light)
plt.scatter(pca_X.iloc[:, 0], pca_X.iloc[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Logistic Regression")
plt.savefig("LR_Marriage.png")
plt.clf()


plt.figure()
plt.pcolormesh(xx, yy, knn_Z, cmap=cmap_light)
plt.scatter(pca_X.iloc[:, 0], pca_X.iloc[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("KNN")
plt.savefig("KNN_Marriage.png")
plt.clf()