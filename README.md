# Fraud Detection Algorithms with SMOTE
[![Python](https://img.shields.io/badge/python-3.8-informational)](https://docs.python.org/3/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the implementation of several traditional Machine Learning algorithms on the Kaggle Transaction Fraud.
The description of the dataset can be found below and the dataset can be downloaded from [**"Credit Card Fraud Dataset"**](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download). 

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Dataset](#dataset)
* [Model](#model)
* [Configuration](#configuration)
* [Usage](#usage)
* [Results](#usage)




### Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

### Model

The Machine Learning models used are Random Forest Regression, Logistic Regression, and Decision Tree Rgression



### Configuration

The clone the repository:
```shell script
git clone git@github.com:giovannicatalani/FraudDetection.git
```
The model uses the **ImbLearn** library.


### Usage

### Results


