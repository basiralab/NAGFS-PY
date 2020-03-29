# NAGFS in Python (Network Atlas-Guided Feature Selection)
NAGFS (Network Atlas-Guided Feature Selection) for a fast and accurate graph data classification code, recoded by Dogu Can ELCI. Please contact elci15@itu.edu.tr for further inquiries. Thanks. 

While typical feature selection (FS) methods aim to identify the most discriminative features in the original feature space for the target classification task, feature extraction (FE) methods cannot track the original features as they extract new discriminative features via projection. Hence, FS methods are more convenient for clinical applications for biomarker discovery. However, existing FS methods are generally challenged by space, time, scalability, and reproducibility. To address these issues, we design a simple but effective feature selection method, which identifies the most discriminative features by comparing healthy and disordered *brain network atlases to learn*.

![NAGFS pipeline](http://basira-lab.com/nagfs_0/)

# Detailed proposed NAGFS pipeline

This work has been published in the Journal of Medical Image Analysis 2020. **Network Atlas-Guided Feature Selection (NAG-FS)** is a network atlas-based connectomic feature selection method for a fast and accurate classification. Our learning-based framework comprises three key steps. (1) Estimation of a centered and representative network atlas, (2) Discriminative connectional biomarker identification, (3) Disease classification. Experimental results and comparisons with the state-of-the-art methods demonstrate that NAG-FS can achieve the best results in terms of classification accuracy and overall computational time. We evaluated our proposed framework from ABIDE preprocessed dataset (http://preprocessed-connectomes-project.org/abide/). 

More details can be found at: https://www.sciencedirect.com/science/article/pii/S1361841519301367 or https://www.researchgate.net/publication/337092350_Joint_Functional_Brain_Network_Atlas_Estimation_and_Feature_Selection_for_Neurological_Disorder_Diagnosis_With_Application_to_Autism

![NAGFS pipeline](http://basira-lab.com/nagfs_1/)

![NAGFS pipeline](http://basira-lab.com/nagfs_2/)


# Libraries to pre-install in Python

We used the following codes from others as follows:

SIMLR code from https://github.com/bowang87/SIMLR_PY 

SNF code from https://github.com/rmarkello/snfpy

# Demo

NAFGS is coded in Python 3. GPU is not needed to run the code.

In this repository, we release the NAGFS source code trained and tested on a simulated heterogeneous graph data drawn from 2 Gaussian distributions as shown below:

![NAGFS pipeline](http://basira-lab.com/nagfs_3/) 

**Data preparation**

We simulated random graph dataset drawn from two Gaussian distributions, each representing a data class, using the function simulateData.py. The number of class 1 graphs, the number of class 2 graphs and the number of nodes (must be >20) are fixed by the operator. Also, the operator should choose the normal distribution parameters (the mean mu and the standard deviation sigma).

To train and evaluate NAG-FS code for classification on other datasets, you need to upload your data as a structure including:

• A Feature matrix of size (N × M) stacking the symmetric matrices of all subjects. N denotes the total number of subjects and M denotes the number of features.<br/>
• A tensor of size (N × m ×m) containing N symmetric connectivity matrices, each of size (m × m). N denotes the number of subjects and m denotes the number of nodes (regions of interest in our case).<br/>
• A label list including the label of each subject in the dataset (healthy or disordered in our work).<br/>

The NAG-FS outputs are:

• A matrix of size (m × m) storing the network atlas of group 1. <br/>
• A matrix of size (m × m) storing the network atlas of group 2. <br/>
• A vector of size (Nf × 1) stacking the indices of the top discriminative features. <br/>

**Train and test NAG-FS**

To evaluate our framework, we used leave-one-out cross validation strategy.

To try our code, you can use: run_demo.py


# Example Results

If you set the number of samples (i.e., graphs) from class 1 to 20, from class 2 to 20, and the size of each graph to 50 (nodes), you will get the following outputs when running the demo with default parameter setting:

![NAGFS pipeline](http://basira-lab.com/nagfs_py_results/) 


# Related references

Similarity Network Fusion (SNF): Wang, B., Mezlini, A.M., Demir, F., Fiume, M., Tu, Z., Brudno, M., HaibeKains, B., Goldenberg, A., 2014. Similarity network fusion for aggregating data types on a genomic scale. [http://www.cogsci.ucsd.edu/media/publications/nmeth.2810.pdf] (2014) [https://github.com/maxconway/SNFtool].

Single‐cell Interpretation via Multi‐kernel LeaRning (SIMLR): Wang, B., Ramazzotti, D., De Sano, L., Zhu, J., Pierson, E., Batzoglou, S.: SIMLR: a tool for large-scale single-cell analysis by multi-kernel learning. [https://www.biorxiv.org/content/10.1101/052225v3] (2017) [https://github.com/bowang87/SIMLR_PY].


# Please cite the following paper when using NAG-FS:

@article{mhiri2019joint,
  title={Joint Functional Brain Network Atlas Estimation and Feature Selection for Neurological Disorder Diagnosis With Application to Autism},<br/>
  author={Mhiri, Islem and Rekik, Islem},<br/>
  journal={Medical Image Analysis},<br/>
  volume={60},<br/>
  pages={101596},<br/>
  year={2020},<br/>
  publisher={Elsevier}<br/>
}<br/>

Paper link on ResearchGate:
https://www.researchgate.net/publication/337092350_Joint_Functional_Brain_Network_Atlas_Estimation_and_Feature_Selection_for_Neurological_Disorder_Diagnosis_With_Application_to_Autism

# License
Our code is released under MIT License (see LICENSE file for details).






