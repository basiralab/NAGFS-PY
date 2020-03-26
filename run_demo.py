#Recoded by Dogu Can ELCI, Istanbul Technical University

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from matplotlib import mlab
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nxviz.plots import CircosPlot
import networkx as nx
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
from ScoreFeaturesAcrossRuns import Score_features
from NAGFS import NAGFS
from simulateData import simulate_data

#* * (1) Some variables are described.
mu1 = 0.9 #Mean value of the first Gaussian distribution
sigma1 = 0.4 #Standard deviation value of the first Gaussian distribution
mu2 = 0.7 #Mean value of the second Gaussian distribution
sigma2 = 0.6 #Standard deviation value of the first Gaussian distribution
Nf = 5 #number of selected features
displayResults = 0
predict_list=[] #total predicts list from all iterations
ind_array=np.empty((0,Nf),int)
#* *

#* * (2) Random connectivity matrixes are created as user inputs.
Data=simulate_data(mu1,sigma1,mu2,sigma2) #simulate all data as connectivity matrixes
#Data=[Featurematrix,X,Label]
# * *


# * * (3) This part include  seperating samples by each classes.
#This part will used for plotting gaussian distribution of datas of each classes.
number=()
number2=()
#Data[0] (Featurematrix) and Data[2](LabelMatrix) has same order and  same index, so they can be seperated featurematrix
#with using labelmatrix.
for i in range(len(Data[2])):
    if Data[2][i]==1:
        number=number+(i,)
data_class1=Data[0][number,:]

for i in range(len(Data[2])):
    if Data[2][i]==-1:
        number2=number2+(i,)
data_class2=Data[0][number2,:]

#Gaussian Distribution needs 1x(NxK) matrixes, so they are converted to this shape.
data_class11=np.concatenate(data_class1)
data_class22=np.concatenate(data_class2)
# * *



#MAIN LOOP---All iteration circulations are here.

# * * (4) In this part, Samples are seperated from each other one by one as a training and testing sets for each iteration.
for i in range(0,(len(Data[2]))):
    print("Iteration number :", i+1)

    #We create a list which contain regular numbers like [0,1,2,3,....,n]->n is a number of total samples
    #So this list is used in iteration for leave one out.
    general_index = []
    for j in range(len(Data[2])):
        general_index.append(j)

    #In first iteration(i=0) 0 is deleted from general_index list, so remains are used as training samples and
    #deleted one is used as testing sample.
    general_index.remove(i)
    train_data=Data[1][general_index,:,:]
    train_feature_data=Data[0][general_index,:]
    train_Labels=Data[2][general_index]
    test_data=Data[1][i,:,:]
    test_feature_data = Data[0][i, :]
    test_Label=Data[2][i]
# * *

# * * (5) NAGFC Function
    #This function is used to seperate training samples into 3 clusters as their similarities and find representative loca atlasses
    #of each center of clusters as AC11, AC12 etc. and then fuse it and find general representative atlasses of all Class-1 samples and all Class-2 samples as AC1 and AC2.

    AC1 , AC2 ,ind = NAGFS(train_data,train_Labels,Nf,displayResults)

# * *

# * * (6) 5 most discriminative features are added in ind_array for each iteration.
    ind1=np.ravel(ind)
    ind_array=np.append(ind_array,[ind1.reshape(Nf)],axis=0)
# * *


# * * (7)
    #Most discriminitive features are determined before with NAGFS function.In this part, all feature
    #columns except these discriminitive feature columns in train and test sets are deleted.(For example, after this part
    #train set turn to [X,5] shape and test set turn to [1,5] shape.)
    delete_list=[]
    cont2=True
    cont4=True
    for i in range(train_feature_data.shape[1]):
        for j in range(len(ind)):
            if i==ind[j]:
                cont2=False
            else:
                continue
        if cont2==True:
            delete_list.append(i) #for delete unnecessary columns-features
        else:
            cont2=True

    train_set=np.delete(train_feature_data,delete_list, 1)

    for i in range(len(test_feature_data)):
        for j in range(len(ind)):
            if i == ind[j]:
                cont4 = False
            else:
                continue
        if cont4 == True:
            delete_list.append(i)
        else:
            cont4 = True
    test_set = np.delete(test_feature_data, delete_list)
    test_set=test_set.reshape(-1,1)
    test_set=test_set.transpose()

#* *

# * * (8)
    #In this part, after all feature reduction, Test samples are classified with SVM classifier and predictions of test set of each iterations are gotten and added to
    #predict list.
    clf=SVC(kernel="linear",C=1)
    clf.fit(train_set,train_Labels)
    pred=clf.predict(test_set)
    predict_list.append(pred)

# * *

# * * (9) Finding Accuracy, Sensitivity and Specificity scores.
conf=confusion_matrix(Data[2],predict_list)
TN = conf[0][0]
FN = conf[1][0]
TP = conf[1][1]
FP = conf[0][1]
TPR = TP / (TP + FN) # Sensitivity
TNR = TN / (TN + FP) # Specificity
ACC = (TP+TN)/(TP+FP+FN+TN) # Accuracy
print("Confusion Matrix: ")
print(conf)
print("* * * * * ")
print("Accuracy Score: ",ACC)
print("Sensitivity Score: ",TPR)
print("Specificity Score: ",TNR)
# * *

# * * (10) Score_index function is used to find 5 most discriminative features across all iterations by scoring them.
Score_index=Score_features(ind_array)
# * *


# * * (11) In this part, 5 most discriminative features which are across all cross-validation runs are plotted in matrix.
aa11=Data[1][0]
aa22=Data[0][0]
last_coor=[]
for i in range(len(Score_index)):
    key1=True
    for j in range(aa11.shape[0]):
        for k in range(aa11.shape[1]):
            if aa22[Score_index[i]]==aa11[j][k]:
                if key1:
                    last_coor.append([j,k])
                    key1=False
                else:
                    continue
topScoreFeatures=np.zeros((aa11.shape[0],aa11.shape[1]))
s=0
ss=0
for i in range(len(Score_index)):
    topScoreFeatures[last_coor[i][0]][last_coor[i][1]]=  aa22[Score_index[s]]
    topScoreFeatures[last_coor[i][1]][last_coor[i][0]] = aa22[Score_index[s]]
    ss+=1
    if ss==2:
        s+=1
        ss=0


#Plotting feature matrix
plt.imshow(topScoreFeatures)
plt.title('NAGFS Most Discriminative Features Across All Cross-Validation Runs')
plt.colorbar()
plt.show()

# * *

# * * (11) Plotting gaussian distribution of both classes
#For Class-1
n, bins, patches = plt.hist(data_class11, 15, normed=1, facecolor='green', alpha=0.75)
y = mlab.normpdf( bins, mu1, sigma1)
l = plt.plot(bins, y, 'g--', linewidth=3)
plt.xlabel('')
plt.ylabel('')
plt.grid(True)

#For Class-2
n, bins, patches = plt.hist(data_class22,15, normed=1, facecolor='blue', alpha=0.75)
y = mlab.normpdf( bins, mu2, sigma2)
l = plt.plot(bins, y, 'b--', linewidth=3)
plt.xlabel('')
plt.ylabel('')
plt.title("Class-specific simulated data distribution (2 classes)".title())
plt.grid(True)
plt.show()

#* *

#* * (12) Plotting circular graph of top Nf discriminative features across all cross-validation runs
node_list=[]
edge_list=[]

for i in range(aa11.shape[0]):
    node=i+1
    node_list.append(node)
for i in range(len(last_coor)):

    last_coor[i][0] += 1
    last_coor[i][1] += 1

    edge_list.append(last_coor[i])
G = nx.Graph()
G.add_nodes_from(node_list)
G.add_edges_from(edge_list)
color_list=["a", "b", "c", "d", "e"]
for n, d in G.nodes(data=True):
    G.node[n]["class"] = node_list[n-1]

c = CircosPlot(graph=G,node_labels=True,
    node_label_rotation=True,
    fontsize=30,
    group_legend=True,
    figsize=(7, 7),node_color="class")

c.draw()
plt.title("circular graph of top Nf discriminative features across all cross-validation runs".title())
plt.show()


#* *
