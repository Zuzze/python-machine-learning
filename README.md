# :robot: Machine Learning with Python

These exercises are based on [Supervised Learning with scikit-learn](https://learn.datacamp.com/courses/supervised-learning-with-scikit-learn) and [Unsupervised Learning in Python](https://learn.datacamp.com/courses/unsupervised-learning-in-python) courses by Datacamp. This is what the description says:

```
Supervised Learning

Machine learning is the field that teaches machines and computers to learn from existing data to make predictions on new data: Will a tumor be benign or malignant? Which of your customers will take their business elsewhere? Is a particular email spam? In this course, you'll learn how to use Python to perform supervised learning, an essential component of machine learning. You'll learn how to build predictive models, tune their parameters, and determine how well they will perform with unseen data—all while using real world datasets. You'll be using scikit-learn, one of the most popular and user-friendly machine learning libraries for Python.


Unsupervised Learning

Say you have a collection of customers with a variety of characteristics such as age, location, and financial history, and you wish to discover patterns and sort them into clusters. Or perhaps you have a set of texts, such as wikipedia pages, and you wish to segment them into categories based on their content. This is the world of unsupervised learning, called as such because you are not guiding, or supervising, the pattern discovery by some prediction task, but instead uncovering hidden structure from unlabeled data. Unsupervised learning encompasses a variety of techniques in machine learning, from clustering to dimension reduction to matrix factorization. In this course, you'll learn the fundamentals of unsupervised learning and implement the essential algorithms using scikit-learn and scipy. You will learn how to cluster, transform, visualize, and extract insights from unlabeled datasets, and end the course by building a recommender system to recommend popular musical artists.
```

## 1 Overview

### 1.1 Types of Machine Learning

#### Supervised learning

_Predict_ the target variable, given the predictor variables with _labeled_ data, finds patterns for a specific prediction task

- Classication: Target variable consists of categories, e.g. cancer tumor benign or cancerous
- Regression: Target variable is continuous

#### Unsupervised learning

find patterns from unlabeled data without prediction task in mind, see whether observations fit into distinct groups based on their similarities

- Clustering: e.g. grouping customers into distinct categories by purchase
- Dimension reduction: e.g. compressing data using purchase patterns

### 1.2 Import dataset

X = data (feature array)
y = target (target array)

```
import pandas as pd
voting_data = pd.read_csv('voting_data.csv')
X = voting_data.drop('party', axis=1).values
y = voting_data['party'].values

```

### 1.3 EDA = Exploratory Data Analysis

to explore data use `df.keys()`, `df.info()`, `df.head()`. For Visual EDA, use for example scattrplot from Pandas:

```
 pd.plotting.scatter_matrix(df, c = y, figsize = [8, 8],                           s=150, marker = 'D')
```

### 1.4 Test-train split

In machine learning, you often split the data you have to training data and testing data. With training data you _train_ your model and then you test how well the trained model is working by using _test_ data.

Basic syntax to do this is:

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```

Here `0.3` means that 30% of the original data is used for testing and 70% for training.

## 2 Supervised Learning

### 2.1 Classification

Target variable consists of categories, e.g. yes/no, female/male, US/FI/CH

#### k-Nearest Neighbors

Predict the label of a data point by looking at the ‘k’ closest labeled data points

```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# get accuracy of the prediction
knn.score(X_test, y_test)
```

Smaller k = more complex model = can lead to overtting

#### Confusion matrix

- A way to evaluate the accuracy of classification predictions via True positive (tp)/False Negative (fn)/False Positive(fp)/True Negative(tn) =>
- Accuracy = (tp + tn) / (tp + tn + fp + fn)

```
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

knn = KNeighborsClassifier(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


```

### 2.2 Regression

Target variable is continuous, eg. price, height etc.

```
import numpy as npfrom sklearn.linear_model
import LinearRegression

y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms),                                    max(X_rooms)).reshape(-1, 1)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space),             color='black', linewidth=3)

plt.show()

```

#### 2.2.1 Linear Regression

Linear regression minimizes a loss function and chooses a coefcient for each feature variable.

y = a<sub>1</sub>x<sub>1</sub> + a<sub>2</sub>x<sub>1</sub> + ... + a<sub>n</sub>x<sub>n</sub> + b, where

- y = target
- x = single feature
- a, b = parameters of model

Linear regression on all features:

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y,    test_size = 0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test)

```

##### Cross-validation

Model performance is dependent on way the data is split => "fold" data in different sections and train data for these separately leaving always some away (test data) => more accurate validation

```
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=5)
print(cv_results)
```

#### 2.2.2 Regularized regression

In linear regression, large coefficients can lead to overfitting => Penalizing large coefficients = Regularization

##### Ridge regression

- Adjust linear loss function: ax + b => ax + b + alpha \* a<sup>2</sup>
- Choose alpha parameter to adjust loss function
- Alpha = 0 => back to linear regression
- Very high alpha => Can lead to underfitting

```
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

```

##### Lasso regression

- Adjust linear loss function: ax + b => ax + b + alpha \* |a|
- Can be used to select important features of a dataset
- Shrinks the coefficients of less important features to exactly 0

```
from sklearn.linear_model import Lasso

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)

```

Lasso for feature selection:

```
from sklearn.linear_model import Lasso

names = boston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

```

#### 2.2.3 Logistic regression and ROC curve

- for binary classification
- Logistic regression outputs probabilities
- By default, logistic regression threshold = 0.5
- If the probability ‘p’ is greater than 0.5: The data is labeled ‘1’, If the probability ‘p’ is less than 0.5:The data is labeled ‘0’

Logistic regression in scikit-learn:

```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_splitlog

reg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,    test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

```

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

### Regularization

Regularization helps to address the problem of over-fitting training data by restricting the model's coefficients.

### Classification

### Regression

#### Linear regression

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x, y)

print("Regression coefficients: {}".format(reg.coef_))
print("Regression intercept: {}".format(reg.intercept_))
```

#### Logistic regression

```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
model.score(X_test, y_test)
```

### ROC Curve

The green curve since it's the closest to the upper left side of the plot, this could be proved by calculating area under the curve

### Bias-variance trade-off

Bias is when the models fails to capture a relationship between the data and the response, resulting in high training and testing errors.

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

- `describe()` dataframe method that will help you extract statistical summaries of your numeric columns
- `K-Means Clustering` K-means clustering groups data into relatively distinct groups by using a pre-determined number of clusters and iterating cluster assignments.

```
from scipy.cluster.vq import kmeans, vq

KMeans(
kmeans(
	df[['x_scaled', 'y_scaled']],
	2
)
```

- `Principal Component Analysis (PCA)` PCA summarizes the original dataset to one with fewer variables, called principal components, that are combinations of the original variables.

  ```
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(scaled_samples)

    pca_features = pca.transform(scaled_samples)
    print(pca_features.shape)
  ```

- `Single linkage` In single linkage, the distance between clusters is the distance between the closest points of the clusters.
- `Complete linkage` In complete linkage, the distance between clusters is the distance between the furthest points of the cluster
