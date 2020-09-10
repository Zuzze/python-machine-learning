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

## 2 :envelope: Supervised Learning

- Using machine learning techniques to build predictive models For both regression and classication problems
- Undertting and overtting
- Test-train split
- Cross-validation
- Gridsearch

### 2.1 Test-train split

In machine learning, you often split the data you have to training data and testing data. With training data you _train_ your model and then you test how well the trained model is working by using _test_ data.

Basic syntax to do this is:

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```

Here `0.3` means that 30% of the original data is used for testing and 70% for training.

### :star: 2.2 Classification

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
print(confusion_matrix(y_test, y_pred))


```

### :star: 2.3 Regression

Target variable is continuous, eg. price, height etc., but in Logistic regression output is 1 or 0.

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

#### 2.3.1 Linear Regression

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

#### 2.3.2 Regularized regression

- In linear regression, large coefficients can lead to overfitting => Penalizing large coefficients = Regularization
- Regularization helps to address the problem of over-fitting training data by restricting the model's coefficients.

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

#### 2.3.3 Logistic regression

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

##### ROC Curve

- Receiver operating characteristic curve (ROC curve) is a graphical plot that illustrates the diagnostic ability of a binary classifier system (e.g. logistic regression) as its discrimination threshold is varied.
- The ROC curve is created by plotting the _true positive_ rate (TPR) against the _false positive_ rate (FPR) at various threshold settings.

```
from sklearn.metrics import roc_curve

y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show();

```

- Larger area under the ROC curve (**AUC**) = better model

```
from sklearn.metrics import roc_auc_score

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)

```

### :star: 2.4 Hyperparameter tuning

- Linear regression: Choosing parameters
- Ridge/lasso regression: Choosing alpha
- k-Nearest Neighbors: Choosing n_neighbors
- Parameters like alpha and k: Hyperparameters
- Hyperparameters **_cannot be learned_** by fitting the model
- Try different values, fit separately, use cross-validation, choose best

#### 2.4.1 GridSearch Cross Validation

```
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
knn_cv.best_params_
knn_cv.best_score_
```

#### 2.4.1 Hold-out set

- Using ALL data for cross-validation is not ideal
- Split data into training and hold-out set at the beginning
- Perform grid search cross-validation on training set

### :star: 2.5 Data preprocessing

#### 2.5.1 Dummy Variables

Scikit-learn will not accept categorical features by default, encode categorical features numerically Convert to ‘dummy variables’

- 0: Observation was NOT that category
- 1: Observation was that category

```
import pandas as pd

df = pd.read_csv('auto.csv')
df_origin = pd.get_dummies(df)
print(df_origin.head())

# encoding to define which one to drop
df_origin = df_origin.drop('origin_Asia', axis=1)

```

#### 2.5.2 Missing data

When many values in your dataset are missing, if you drop them, you may end up throwing away valuable information along with the missing data. It's better instead to develop an imputation strategy. This is where domain knowledge is useful, but in the absence of it, you can impute missing values with the **mean** or the **median** of the row or column that the missing value is in.

```
df.insulin.replace(0, np.nan, inplace=True)

```

drop missing data

```
df = df.dropna()
df.shape(393, 9)
```

Imputing missing data: Making an educated guess about the missing values

```
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0imp.fit(X)
X = imp.transform(X)

```

Imputing within pipeline

```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
logreg = LogisticRegression()
steps = [('imputation', imp), ('logistic_regression', logreg)]
pipeline = Pipeline(steps)X_train, X_test, y_train, y_test = train_test_split(X, y,    test_size=0.3, random_state=42)

```

#### 2.5.3 Normalizing data

Features on larger scales can unduly inuence the model => We want features to be on a similar scale

Ways to normalize your data:

- Standardization: Subtract the mean and divide by variance
- All features are centered around zero and have variance one
- Can also subtract the minimum and divide by the range
- Minimum zero and maximum one
- Can also normalize so the data ranges from -1 to +1

```
from sklearn.preprocessing import scale

X_scaled = scale(X)
np.mean(X), np.std(X)
np.mean(X_scaled), np.std(X_scaled)

```

CV and scaling in a pipeline

```
from sklearn.preprocessing import StandardScaler

steps = [('scaler', StandardScaler()),       (('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)parameters = {knn__n_neighbors: np.arange(1, 50)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
knn_unscaled.score(X_test, y_test)

```

## 3 Unsupervised Learning

- Supervised learning finds patterns for a prediction task whereas unsupervised learning finds patterns in data without specific prediction task in mind
- **Dimension** = Number of features
- When dimension too high to visualize, unsupervised learning gives insight

#### 3.1 Glossary

##### K-means clustering

- `k-means clustering` groups data into relatively distinct groups (clusters) by using a pre-determined number of clusters and iterating cluster assignments.

```
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
KMeans(algorithm='auto', ...)
labels = model.predict(samples)
```

- `Inertia` measures clustering quality, the lower the better

##### Hierarchical Clustering

- `Hierarchical clustering` each element begins as a separate cluster, At each step, the two closest clusters are merged (agglomerative hierarchical clustering). Can be visualised with dendrogram.
- `Single linkage` In single linkage, the distance between clusters is the distance between the closest points of the clusters.
- `Complete linkage` In complete linkage, the distance between clusters is the distance between the furthest points of the cluster

##### t-SNE for 2-dimensional map

- t-distributed stochastic neighbor embedding
- Maps samples to 2D space (or 3D)
- Map approximately preserves nearness of samples
- Different every time

##### Principal Component Analysis

- `Principal Component Analysis (PCA)` PCA summarizes the original dataset to one with fewer variables, called principal components, that are combinations of the original variables.
- More efficient storage and computation
- Remove less-informative "noise" features
- aligns data with axes
- `Pearson correlation` = Measures linear correlation of features, value between -1 and 1, value 0 = no correlation
- `Principal components` = directions of variance
- `Intrinsic dimension` = number of features needed to approximate the dataset, can be detected with PCA

  ```
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(scaled_samples)

    pca_features = pca.transform(scaled_samples)
    print(pca_features.shape)
  ```

##### Non-negative matrix factorization (NMF)

- Dimension reduction technique
- NMF models are interpretable (unlike PCA)
- All sample features must be non-negative (>= 0)
- Must specify number of components e.g. NMF(n_components=2)

```
from sklearn.decomposition import NMF
model = NMF(n_components=2)
```

##### Cosine similarity

- Used in recommendation systems to recommend similar articles/songs/etc
- similar elements have similar NMF feature values
- Calculated as the angle between the lines
- Higher values means more similar
- Maximum value is 1, when angle is 0 ̊
- `Bias-variance trade-off`

##### Bias

is when the models fails to capture a relationship between the data and the response, resulting in high training and testing errors.

###### describe()

dataframe method that will help you extract statistical summaries of your numeric columns
