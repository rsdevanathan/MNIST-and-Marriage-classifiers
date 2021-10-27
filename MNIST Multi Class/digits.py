import numpy as np
import scipy.io
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)




digits_mat = scipy.io.loadmat('mnist_10digits.mat')


train_data = pd.DataFrame(np.hstack((digits_mat['xtrain'], digits_mat['ytrain'].T)))
train_data_sample = train_data.sample(n=5000)

test_data = pd.DataFrame(np.hstack((digits_mat['xtest'], digits_mat['ytest'].T)))

Xtrain,ytrain = train_data_sample.iloc[:, 0:784],train_data_sample.iloc[:,784]
Xtest,ytest = test_data.iloc[:, 0:784],test_data.iloc[:,784]

Xtrain = Xtrain.div(255)

Xtest = Xtest.div(255)


####MedianTrick
#print("Median Trick")
medtrick_X = Xtrain.sample(n=1000).to_numpy()

medtric_dist = scipy.spatial.distance.pdist(medtrick_X)
M= np.median(medtric_dist**2)
sigma = np.sqrt(M/2)
gamma = 1/(2*(sigma**2))
#print("gamma",gamma)


svm_model = SVC(kernel='linear').fit(Xtrain, ytrain)
kern_svm_model = SVC(kernel='rbf',gamma=gamma).fit(Xtrain, ytrain)
lr_model = LogisticRegression().fit(Xtrain, ytrain)
#knn_model_base = KNeighborsClassifier()
#knn_model_gs = GridSearchCV(knn_model_base, dict(n_neighbors = list(range(5,15))), cv=5).fit(Xtrain,ytrain)
#knn_model = knn_model_gs.best_estimator_
#print(knn_model_gs.best_params_)  #### Output --- > {'n_neighbors': 5}

# As per the above grid search 5 is chosen as n_neighbors and the model is fitted below. Gridsearch part is commented in final code to improve the performance.
knn_model = KNeighborsClassifier(n_neighbors = 5).fit(Xtrain, ytrain)

nn_model = MLPClassifier(hidden_layer_sizes=(20, 10)).fit(Xtrain,ytrain)



modellist = [svm_model,kern_svm_model,lr_model,knn_model,nn_model]
modelname = ["SVM","Kernal_SVM","LR","KNN","NN"]

for name,model in zip(modelname,modellist):
    y_pred = model.predict(Xtest)
    print("Classification report for "+name+" classifier")
    print(metrics.classification_report(ytest, y_pred))
    disp = metrics.plot_confusion_matrix(model, Xtest, ytest,values_format='')
    disp.figure_.suptitle("Confusion Matrix for " + name)
    plt.savefig("CM_ " + name + ".png")