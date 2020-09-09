# :robot: Supervised and unsupervised Machine Learning with Python

These exercises are based on [Datacamp's Supervised Learning with scikit-learn](https://learn.datacamp.com/courses/supervised-learning-with-scikit-learn) course. This is what the description says:
```
Machine learning is the field that teaches machines and computers to learn from existing data to make predictions on new data: Will a tumor be benign or malignant? Which of your customers will take their business elsewhere? Is a particular email spam? In this course, you'll learn how to use Python to perform supervised learning, an essential component of machine learning. You'll learn how to build predictive models, tune their parameters, and determine how well they will perform with unseen data—all while using real world datasets. You'll be using scikit-learn, one of the most popular and user-friendly machine learning libraries for Python.
```

## Summary

### Types of Machine Learning
- Supervised learning: Predict the target variable, given the predictor variables with labeled data, finds patterns for a specific prediction task
    - Classication: Target variable consists of categories, e.g. cancer tumor benign or cancerous
    - Regression: Target variable is continuous
- Unsupervised learning: find patterns from unlabeled data without prediction task in mind
    - Clustering: e.g. grouping customers into distinct categories by purchase
    - Dimension reduction: e.g. compressing data using purchase patterns


### Overfitting and underfitting

### EDA = Exploratory Data Analysis
to explore data use `df.info()`, `df.head()` and `df.describe()`



### Test-train split
In machine learning, you often split the data you have to training data and testing data. With training data you *train* your model and then you test how well the trained model is working by using *test* data.

Basic syntax to do this is:
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)`

Here `0.3` means that 30% of the original data is used for testing and 70% for training.

### Cross-validation


### Grid search

### Regularization
Lasso and Ridge Regression

### Data preprocessing

### Missing data
When many values in your dataset are missing, if you drop them, you may end up throwing away valuable information along with the missing data. It's better instead to develop an imputation strategy. This is where domain knowledge is useful, but in the absence of it, you can impute missing values with the **mean** or the **median** of the row or column that the missing value is in.

### Normalizing data

Scale: 
```
from sklearn.preprocessing import scale
X_scaled = scale(X)
```

## Cheatsheet
`EDA` exploratory data analysis

### Classification

### Regression
`ridge = Ridge(alpha=0.5, normalize=True)`
`lasso = Lasso(alpha=0.5, normalize=True)`

### Import

**Library**
`import pandas as pd`
`from sklearn.linear_model import Ridge`
`from sklearn.model_selection import cross_val_score`

**dataframe (CSV)**
`df = pd.read_csv('dataset_name.csv')`


## Unsupervised learning

- `Single linkage` In single linkage, the distance between clusters is the distance between the closest points of the clusters.
- `Complete linkage` In complete linkage, the distance between clusters is the distance between the furthest points of the cluster