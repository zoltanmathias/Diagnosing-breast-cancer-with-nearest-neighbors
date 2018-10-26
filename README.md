# Diagnosing-breast-cancer-with-nearest-neighbors
Applying k-NN (nearest neighbors) algorithm to a dataset to predict whether a tumor is malignant or benign. 

This is achieved with the use of Python scikit-learn and it's sklearn, numpy, and pandas libraries.

The dataset used can be found here: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

The features in the dataset are already on a scale from 1-10. There are however empty spaces in the data that are denoted by '?'. The entire sample can be deleted from the dataset, or the empty features can be modified so they become obvious outliers. The data is split 75/25 training data and test data. This can be easily adjusted in the script. 

The code is quite short and demonstrates the effectiveness of scikit-learn as a machine learning tool. I am very impressed with the accuracy with which the algorithm is able to predict the outcome of a tumor. Roughly about 95%. I say roughly because k-NN is dependent on the value chosen for the k, and I've noticed a +/-1% difference during different runs of the code.
