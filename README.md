# K-Fold-Assignment

## Description
The objective of K-Fold Cross-Validation is assesing the performance of K-Nearest Neighbor on a given dataset, in this case, the MNIST dataset. 
* The MNIST dataset is a standard benchmark in machine learning for handwritten digit recognition. This dataset provides a robust and well-known set of data for testing classification algorithms.

By splitting the data into multiple folds and training/testing the model on each fold, it ensures that the model’s performance is not dependent on a particular train-test split. This provides a more generalizable estimate of the model’s accuracy.

The KFold object has 10 splits, which means the data is divided into 10 parts, and the model is trained and tested 10 times, each time with a different fold as the test set and the remaining folds as the training set. This helps in obtaining a comprehensive performance metric.
```
kfold=KFold(n_splits=10, random_state=11, shuffle=True)
```

cross_val_score computes the accuracy of the KNN classifier across each fold and returns an array of accuracy scores. This array gives insight into the model’s performance, including potential variance in accuracy across different subsets of the data.
```
scores = cross_val_score(knn, mnist[0], mnist[1], cv=kfold)
```

## Results
Observing the scores tuple, we get the following result
```
array([0.97171429, 0.97257143, 0.97057143, 0.97185714, 0.97071429, 0.97185714, 0.976, 0.97314286, 0.97157143, 0.97557143])
```
The results indicate that each fold's accuracy is approximately 97%, showing that the KNN classifier performs very well on the MNIST dataset. This high accuracy suggests that the model is highly effective at recognizing handwritten digits. The accuracy scores are consistent across the 10 folds, with values ranging between 0.9706 and 0.9760. This consistency indicates that the model’s performance is stable and not highly variable across different subsets of the data.

Calulating the mean and standard deviation, we get:
```
Mean accuracy: 97.255714%
Accuracy St Dev: 0.176872%
```
These statistics indicate that the model not only has high accuracy but also low variability, which implies reliability and robustness in its performance.
