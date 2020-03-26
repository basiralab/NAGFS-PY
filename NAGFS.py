"""
 Main function of NAGFS framework for a fast and accurate classification.
 Details can be found in the original paper: https://www.scieC2edirect.com/scieC2e/article/abs/pii/S1361841519301367
 Islem Mhiri and Islem Rekik. "Joint input_ brain network atlas estimation and feature selection for neurological disorder diagnosis
with application to autism",Medical Image Analysis, 2019, p. 101596.

   ---------------------------------------------------------------------
     This file contains the implementation of three key steps of our NAG-FS framework:
     (1) Estimation of a centered and representative input_ network atlas,
     (2) Discriminative connectional biomarker identification and
     (3) disease classification:

                        [AC1,AC2,ind] = NAGFS(train_data,train_Labels,Nf,displayResults)

                 Inputs:

                          train_data: ((n-1) × m × m) tensor stacking the symmetric matrices of the training subjects
                                      n-->the total number of subjects
                                      m-->the number of nodes

                          train_Labels: ((n-1) × 1) vector of training labels (e.g., -1, 1)
                          Nf: Number of selected features

                          displayResults: Boolean variables [0, 1].
                                    if displayResults = 1 ==> display(Atlas of group 1, Atlas of group 2, top features matrix and the circular graph)
                                    if displayResults = 0 ==> no display
                 Outputs:
                         AC1: (m × m) matrix storing the atlas of Class 1

                         AC2: (m × m) matrix storing the atlas of Class 2

                         ind: (Nf × 1) vector storing the indices of the top disciminative features


     To evaluate our framework we used Leave-One-Out cross validation strategy.



To test NAGFS on random data, we defined the function 'simulateData' where the size of the dataset is chosen by the user.
 ---------------------------------------------------------------------
     Copyright 2020 Dogu Can ELCI, Istanbul Technical University.

     Please cite the above paper if you use this python code.
     All rights reserved.

"""



# noinspection PyUnresolvedReferences
import scipy as sc
# noinspection PyUnresolvedReferences
import sklearn as sk
import numpy as np

# noinspection PyUnresolvedReferences
import SIMLR
# noinspection PyUnresolvedReferences
import snf
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt

def NAGFS(train_data,train_Labels,Nf,displayResults):

    XC1 = np.empty((0, train_data.shape[2],train_data.shape[2]), int)
    XC2 = np.empty((0, train_data.shape[2],train_data.shape[2]), int)

# * * (5.1) In this part, Training samples which were chosen last part are seperated as class-1 and class-2 samples.
    for i in range(len(train_Labels)):

        if (train_Labels[i] == 1):
            XC1=np.append(XC1,[train_data[i,:,:]],axis=0)
        else:
            XC2=np.append(XC2,[train_data[i,:,:]],axis=0)

# * *

# * * (5.2) SIMLR functions need matrixes which has 1x(N*N) shape.So all training matrixes are converted to this shape.

#For C1 group
    k=np.empty((0,XC1.shape[1]*XC1.shape[1]),int)
    for i in range(XC1.shape[0]):
        k1=np.concatenate(XC1[i]) #it vectorizes all NxN matrixes
        k=np.append(k,[k1.reshape(XC1.shape[1]*XC1.shape[1])],axis=0)


# For C2 group
    kk = np.empty((0, XC2.shape[1] * XC2.shape[1]), int)
    for i in range(XC2.shape[0]):
        kk1 = np.concatenate(XC2[i])
        kk = np.append(kk, [kk1.reshape(XC2.shape[1] * XC2.shape[1])], axis=0)


# * *

# * * (5.3) SIMLR(Single-Cell Interpretation via Multi Kernel Learning) is used to clustering of samples of 2 classes into 3 clusters.

#For C1 group
    #[t1, S2, F2, ydata2,alpha2] = SIMLR(kk,3,2);
    simlr = SIMLR.SIMLR_LARGE(3,4,0) #This is how we initialize an object for SIMLR.The first input is number of rank (clusters) and the second input is number of neighbors.The third one is an binary indicator whether to use memory-saving mode.You can turn it on when the number of cells are extremely large to save some memory but with the cost of efficiency.
    S1, F1,val1, ind1 = simlr.fit(k)
    y_pred_X1 = simlr.fast_minibatch_kmeans(F1,3) #This SIMLR function predicts training 1x(N*N) samples what they belong.
    #to first, second or third clusters.(0,1 or 2)

    # For C2 group
    simlr = SIMLR.SIMLR_LARGE(3,4,0)
    S2, F2,val2, ind2 = simlr.fit(kk)
    y_pred_X2 = simlr.fast_minibatch_kmeans(F2,3)

# * *

# * * (5.4) Training samples are placed into their predicted clusters for Class-1 and Class-2 samples.
#For XC1, +1 k
    Ca1 = np.empty((0, XC1.shape[2],XC1.shape[2]), int)
    Ca2 = np.empty((0, XC1.shape[2],XC1.shape[2]), int)
    Ca3 = np.empty((0, XC1.shape[2],XC1.shape[2]), int)

    for i in range(len(y_pred_X1)):
        if y_pred_X1[i]==0:
            Ca1=np.append(Ca1,[XC1[i,:,:]],axis=0)
            Ca1=np.abs(Ca1)
        elif y_pred_X1[i]==1:
            Ca2=np.append(Ca2,[XC1[i,:,:]],axis=0)
            Ca2 = np.abs(Ca2)
        elif y_pred_X1[i]==2:
            Ca3=np.append(Ca3,[XC1[i,:,:]],axis=0)
            Ca3 = np.abs(Ca3)

#For XC2, -1 kk
    Cn1 = np.empty((0, XC2.shape[2],XC2.shape[2]), int)
    Cn2 = np.empty((0, XC2.shape[2],XC2.shape[2]), int)
    Cn3 = np.empty((0, XC2.shape[2],XC2.shape[2]), int)

    for i in range(len(y_pred_X2)):
        if y_pred_X2[i]==0:
            Cn1=np.append(Cn1,[XC2[i,:,:]],axis=0)
            Cn1 = np.abs(Cn1)
        elif y_pred_X2[i]==1:
            Cn2=np.append(Cn2,[XC2[i,:,:]],axis=0)
            Cn2 = np.abs(Cn2)
        elif y_pred_X2[i]==2:
            Cn3=np.append(Cn3,[XC2[i,:,:]],axis=0)
            Cn3 = np.abs(Cn3)

# * *

    #SNF PROCESS
# * * (5.5) SNF(Similarity Network Fusion) is used for create a local centered network atlas which is the best representative matrix
#of other similar matrixes.In this process, for every class, there are 3 clusters, so snf create 3 representative-center
#matrixes for both classes.After that it create 1 general representative matrixes of 3 matrixes.
#So finally there are 2 general representative matrixes.

    #Ca1
    class1=[]
    if Ca1.shape[0]>1:
        for i in range(Ca1.shape[0]):
            class1.append(Ca1[i,:,:])
        affinity_networks = snf.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC11=snf.snf(affinity_networks,K=20) #First local network atlas for C1 group
        class1=[]
    else:
        AC11=Ca1[0]

    #Ca2
    class1=[]
    if Ca2.shape[0]>1:
        for i in range(Ca2.shape[0]):
            class1.append(Ca2[i,:,:])
        affinity_networks = snf.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC12=snf.snf(affinity_networks,K=20) #Second local network atlas for C1 group
        class1 = []
    else:
        AC12=Ca2[0]

    #Ca3
    class1=[]
    if Ca3.shape[0]>1:
        for i in range(Ca3.shape[0]):
            class1.append(Ca3[i,:,:])
        affinity_networks = snf.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC13=snf.snf(affinity_networks,K=20) #Third local network atlas for C1 group
        class1 = []
    else:
        AC13=Ca3[0]

    #Cn1
    if Cn1.shape[0]>1:
        class1=[]
        for i in range(Cn1.shape[0]):
            class1.append(Cn1[i,:,:])
        affinity_networks = snf.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC21=snf.snf(affinity_networks,K=20) #First local network atlas for C2 group
        class1 = []
    else:
        AC21=Cn1[0]

    #Cn2
    class1=[]
    if Cn2.shape[0]>1:
        for i in range(Cn2.shape[0]):
            class1.append(Cn2[i,:,:])
        affinity_networks = snf.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC22=snf.snf(affinity_networks,K=20) #Second local network atlas for C2 group
        class1 = []
    else:
        AC22=Cn2[0]

    #Cn3
    class1=[]
    if Cn3.shape[0]>1:
        for i in range(Cn3.shape[0]):
            class1.append(Cn3[i,:,:])
        affinity_networks = snf.make_affinity(class1, metric='euclidean', K=20, mu=0.5)
        AC23=snf.snf(affinity_networks,K=20)  #Third local network atlas for C2 group
        class1 = []
    else:
        AC23=Cn3[0]


    #A1
    AC1 = snf.snf([AC11, AC12, AC13], K=20)  #Global network atlas for C1 group

    #A2
    AC2 = snf.snf([AC21, AC22, AC23], K=20)  #Global network atlas for C2 group

# * *


# * * (5.6) In this part, most 5 discriminative connectivities are determined and their indexes are saved in ind array.

    D0=np.abs(AC1-AC2) #find differences between AC1 and AC2
    D=np.triu(D0) #Upper triangular part of matrix
    D1=D[np.triu_indices(AC1.shape[0],1)] #Upper triangular part of matrix
    D1=D1.transpose()
    D2=np.sort(D1)  #Ranking features
    D2=D2[::-1]
    Dif=D2[0:Nf] #Extract most 5 discriminative connectivities
    D3=[]
    for i in D1:
        D3.append(i)
    ind=[]
    for i in range(len(Dif)):
        ind.append(D3.index(Dif[i]))
# * *

# * * (5.7) Coordinates of most 5 disriminative features are determined for plotting for each iteration if displayresults==1.

    coord=[]
    for i in range(len(Dif)):
        for j in range(D0.shape[0]):
            for k in range(D0.shape[1]):
                if Dif[i]==D0[j][k]:
                    coord.append([j,k])

    topFeatures=np.zeros((D0.shape[0],D0.shape[1]))
    s=0
    ss=0
    for i in range(len(Dif)*2):
        topFeatures[coord[i][0]][coord[i][1]]=Dif[s]
        ss+=1
        if ss==2:
            s+=1
            ss=0
    if displayResults==1:
        plt.imshow(topFeatures)
        plt.colorbar()
        plt.show()
# * *

    return AC1 , AC2 ,ind


















