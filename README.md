# Credit_Risk_Analysis

## Resources
- Anaconda 4.11.0
- Jupyter Notebook 6.0.3
- Python 3.7.6
- Numpy
- Pandas
- scikit-learn
- Data:  LoanStats_2019Q1.csv

## Project Overview

Apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, you’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once finsihed, I evaluated the performance of these models and made a written recommendation on whether they should be used to predict credit risk. 

Project consisted of three technical analysis deliverables and a written report as follows:

- Deliverable 1: Use Resampling Models to Predict Credit Risk
- Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
- Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
- Deliverable 4: A Written Report on the Credit Risk Analysis (README.md)

## Use Resampling Models to Predict Credit Risk

Using the imbalanced-learn and scikit-learn libraries, I evaluated three machine learning models by using resampling to determine which is better at predicting credit risk. First, I used the oversampling RandomOverSampler and SMOTE algorithms, and then I used the undersampling ClusterCentroids algorithm. Using these algorithms, I resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

### Naive Random Oversampling

![Naive ROS](https://github.com/PatriciaCB1/Credit_Risk_Analysis/blob/main/Images/Naive%20Random%20Oversampler.png) 

## Use the SMOTEENN algorithm to Predict Credit Risk

Using knowledge of the imbalanced-learn and scikit-learn libraries, I used a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the SMOTEENN algorithm, I resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

### SMOTE Oversampling

![Oversampling](https://github.com/PatriciaCB1/Credit_Risk_Analysis/blob/main/Images/SMOTE%20Oversampling.png) 

### Undersampling

![Undersampling](https://github.com/PatriciaCB1/Credit_Risk_Analysis/blob/main/Images/Undersampling.png)

### Combination (Over and Under) Sampling

![Combination](https://github.com/PatriciaCB1/Credit_Risk_Analysis/blob/main/Images/Combination%20(Over%20and%20Under)%20Sampling.png)


## Use Ensemble Classifiers to Predict Credit Risk

Using knowledge of the imblearn.ensemble library, I trained and compared two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluated each model. Using both algorithms, I resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

### Balanced Random Forest Classifier

![Balanced Random Forest](https://github.com/PatriciaCB1/Credit_Risk_Analysis/blob/main/Images/Balanced%20Random%20Forest%20Classifier.png)

### Easy Ensemble Classifier
![Easy Ensemble Classifier](https://github.com/PatriciaCB1/Credit_Risk_Analysis/blob/main/Images/Easy%20Ensemble%20Classifier.png)

## Results

### Naive Random Oversampling
- Balanced accuracy score: 65%
- High Risk Precision score:  1%
- Recall score:  66%

### SMOTE Oversampling
- Balanced accuracy score: 61%
- High Risk Precision score:  1%
- Recall score: 65%

### Undersampling
- Balanced accuracy score:  61%
- High Risk Precision score:  1%
- Recall score: 48%

### Combination (Over and Under) Sampling
- Balanced accuracy score: 52%
- High Risk Precision score:  1%
- Recall score: 58%

### Balanced Random Forest Classifier
- Balanced accuracy score: 80%
- High Risk Precision score:  3%
- Recall score: 88%

### Easy Ensemble Classifier
- Balanced accuracy score: 93%
- High Risk Precision score:  8%
- Recall score: 94%


## Summary

I would not recommend the first four models on the basis of their low balanced accuracy scores.  Although the balanced random forest classifer has higher accuracy the precesion for High Risk is still low at 3%.  The easy ensemble classifier performed the best with a balanced accuracy score of 93%, however with high risk precision at only 7% and high risk accuracy at 91% it cannot be recommended.  I would not recommend any of these models be used for credit risk analysis.  

