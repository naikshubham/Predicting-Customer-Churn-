# Predicting-Customer-Churn
Churn is when a customer stops doing business or ends a relationship with a company. It’s a common problem across a variety of industries, from telecommunications to cable TV to SaaS, and a company that can predict churn can take proactive action to retain valuable customers and get ahead of the competition. 

### Exploratory Data Analysis
- Data used here comes from a Cellular usage dataset that consists of records of actual cell phone customers and features.
- Features of interest includes specific features to a customer's cell service, like **`voice mail`** , **`international calling`**, **`cost for the service**`, **`customer usage`**, **`customer churn`**
- Here **churn** is defined as the customer cancelling their cellular plan at a given point in time and is encoded in the dataset as **`no`** and **`yes`**.
- In EDA we understand the features of the dataset, compute summary statistics.

#### Grouping and summarizing data
- Our goal is to classify whether or not a new customer will churn. This model thus has two outcomes, `yes` : customer will churn or `no` : customer will not churn
- We can use EDA to identify differences between these 2 classes. `Do churners call customer service more often?` or `Does one state have more churners compared to another?`. These are some questions we can ask of the data.
- To answers these questions we need to be able to group and summarize our data. To group the data pandas has a method called **`.groupby()`**

### Exploring data using visualizations
- **`Visualizing data in python`** : `seaborn` library allows us to easily create informative & attractive plots, built on top of `matplotlib`

#### Visualizing the distribution of account lengths
- Many ML algorithms make assumptions about **`how the data is distributed`**, so it's **important to understand how the variables in our own dataset are distributed before we apply those algorithms**.
- A **histogram** is an effective way to visualize the distribution of a variable, and we can create one using seaborn's **distplot** function (distribution plot)

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(telco['Account Length'])
plt.show()
```

<p align="center">
  <img src="data/vis_1.JPG" width="350" title="visualization">
</p>

- Here it resembles a bell curve, also known as normal distribution.It turns out that many things we measure in the real-world are well approximated by the normal distribution, and many models actually make the assumption that our data is normally distributed.
- Lets now visualize the differences in account length between churners and non-churners. Effective way to do it using `box plot`, we can do it using seaborn's boxplot function

```python
sns.boxplot(x='Churn',
            y='Account_Length',
            data = telco,
            sym="")
plt.show()
```
<p align="center">
  <img src="data/boxplot.JPG" width="350" title="boxplot">
</p>

- There doesn't appear to be any noticeful difference in account length. 
- The line in the middle of each box represents the median. The colored boxes represent the middle 50% of the account lengths for each group. The values here range from the 25th to the 75th percentile and gives a sense for the spread of the distribution. The floating points represents outliers, which we can remove using the  **`sym`** parameter

#### Adding a third variable
- Seaborn allows us to easily add a third variable to the plot. For e.g, we might be intersted in visualizing whether the "International Plan" feature has an impact on Account Length or Churn.We can add this info to the plot by specifying the **`hue`** parameter.

<p align="center">
  <img src="data/hue.JPG" width="350" title="hue">
</p>

- From the plot,it looks like as far as predicting churn goes it does not matter whether or not a customer had an international plan.

### Data Preprocessing for modelling
- Some assumptions that the model make : **that the features are normally distributed** , **that the features are on the same scale**
- Certain ML algorithms make assumptions about the data. If the features in our dataset do not meet these assumptions, then the results of the model won't be reliable.
- That's why the data-preprocessing stage is so critical.

#### Data-types
- Many ML algorithms so accept numeric datatypes.So if any of the features are categorical, they will need to be first encoded numerically.
- We can look at the data types in the telco Dataframe using its `dtype` attribute.Numeric columns have datatypes as int64 or float64 while any columns that includes text are encoded as object.

#### Encoding binary features
- Some features have two values `yes` and `no`, to encode them numerically we can use `no` as `0` & `yes` as `1`, using either **`replace()`** method or scikit-learn's LabelEncoder function.

```python
telco['Intl_Plan'].replace({'no':0, 'yes':1})

# or

from sklearn.preprocessing import LabelEncoder

LabelEncoder().fit_transform(telco['Intl_Plan'])
```
- The `State` feature is a bit more complex to represent numerically, because there are so many states. We can assign a number to each state ` 0 for Kansas`, `1 for Ohio`, `2 for New Jersey` and so on.
- But assigning arbitary numbers like this is dangerous, **as it implies some form of ordering in the states**. This would make sense for a feature that had categories like 'low', 'medium', 'high', but in this case it doesnt make sense to order states, and doing so would make model less effective.
- Instead, we can encode states using **`one hot encoding`**. This creates new binary features.

#### Feature scaling
- Another imp preprocessing step is feature scaling. Most models require features to be on the same scale, but this is rarely true in case of real world data.
- In our dataset, the `Intl_Calls` feature ranges from 0 to 20, while the `Night_Mins` feature ranges from 23 to 395.
- So we need to rescale our data and ensure all our features are on the same scale.We'll do this using a process known as **`standardization`**, which centers the distribution around the mean of the data and calculates the number of standard deviations away from the mean each point is.

```python
from sklearn.preprocessing import StandardScaler

df = StandarScaler().fit_transform(df)
```

### Feature selection and engineering
- Datasets often have features that provide no predictive power and need to be dropped prior to modeling. Features that can be dropped include unique identifiers such as phone numbers, social security numbers and account numbers.

#### Dropping correlated features
- Features that are highly correalted with other features can also be dropped, **as they provide no additional information to the model**.
- The **.corr()** method allows us to explore the correlation between the features in the dataset. In our dataset, (Day_Mins, Eve_Mins, Night_Mins and Intl_Mins) are highly correlated with (Day_Charge, Eve_Charge, Night_Charge and Intl_Charge) respectively.
- Intutively, it makes sense that these features should be correlated, and from modeling standpoint, we can improve the performance of our models by removing these redundant features.
- This process of choosing which features to use in our model is known as **`feature selection`**. Besides selecting features, we often need to create new features to help improve model performance. This is known as **`feature engineering`**
- Consulting with the business and subject matter experts can lead to additional features, and should be a crucial step for every data science workflow. Together with feature selection, feature engineering is a critical step that can add a lot of value to your final model.

#### Examples of feature engineering
- One example of a new feature we could create is `Total Minutes`, which combines Day Minutes, Evening Minutes, Night Minutes and International minutes. Or we can create a new feature that is the ratio between Minutes and Charge.

```python
telco['Day_Cost'] = telco['Day_Mins'] / telco['Day_Charge']
```

### Making Predictions
- Goal is to predict whether or not customer will churn depending on various features. Target : **`Churn`**, Supervised Machine Learning problem.
- First we need to decide which model we want to use. This is often one of the most difficult questions faced by data scientists, and the answer often is, it depends.
- For classification problems the good baseline model to begin with is logistic regression.If offers simplicity and interpretability, however it is not flexible enough to capture more complex relationsips in our dataset.
- Random Forests are a good next step - they have high performance but offer limited interpretability. Support Vector Machines are another option.

```python
from sklearn.svm import SVC

svc = SVC()

svc.fit(telco[features], telco['target'])

prediction = svc.predict(new_customer)
print(prediction)
```

### Evaluating Model performance

#### Accuracy
- One possible metric:accuracy : total num of correct predictions/total number of data points
- Train on the training set and test using test set.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(telco['data'], telco['Churn'], test_size=0.2, random_state = 42)

from sklearn.svm import SVC
svc= SVC()
svc.fit(X_train, y_train)
svc.predict(X_test)

# compute accuracy
svc.score(X_test, y_test)
```

- Random state argument allows us to specify a seed.Setting a seed ensures that our results are reproducible, as all splits will be same seed and will be identical.

#### Improving our model
- Overfitting : Model fits the training data too closely, and doesn't generalize well on unseen data.
- Underfitting : Does not capture the patterns in the training data.
- Need to find the right balance between overfitting and underfitting.

### Model Metrics

#### Imbalanced classes
- In our churn datasets there are 6 times more non-churners as compared to churners. This might have an impact on the performance of the model, as it might always learn to predict the majority class(in this case that the customers will not churn). There are techniques like upsampling and downsampling that we can use to balance the classes.
- When working with imbalanced classes, **`accuracy is not a very useful metric`**. If we were to build a classifier that **always predicted that a customer would not churn it would be very accurate(about 85% accurate in this case).** yet it would completely fail at identifying the customers who will actually churn. This creates the need for metrics that are more nuanced like **confusion matrix, precision and recall**
- **Precision** : If our model had high precision, it would mean that there were **not many false positives**- that is, not many non-churners who were classified as churners(few false positives).
- **Recall/Sensitivity** :(True positives/(True Positives + False Negatives)) A high recall in this case means that the model correctly classified most churners.
- **Which metric we focus on optimizing often depends on the business case(use case).** 
- **If the offers we make to keep potential churners is high in relation to the value of losing customers, we want our model to have high precision or minimize false positives. Conversely, if losing customers we didnt think would churn is more expensive than an offer to keep a customer that would not have churned than we want a high recall or minimize false negatives**
- Both recall and precision are far more valuable than just looking at accuracy.

### Other model metrics
The **receiver operating characteristics curve(ROC curve**), that allows us to visualize the performance of our classifer.

#### Probability thresholds
- Every prediction our classifer makes has an associated probability.By default in scikit-learn, if this probability is above 50%, then our model would predict the data point as belonging to the positive class, and if it is lower than 50%, it would predict the negative class. The default probabilty threshold here is 50%
- What if we vary this threshold and for each threshold plot the model's TPR against the FPR, we get what is known as the **ROC curve**
- This curve allows us to diagnose the performance of the model at different thresholds. Y-axis is the TPR and ranges from 0 to 1 and the X-axis is the FPR which also ranges from 0 to 1.
- A perfect classifier will have a TPR of 1 and FPR of 0.This information can be used to extract another useful metric: **the area under the ROC curve, known as the AUC.**
- A better performing model will have a large area under the curve

<p align="center">
  <img src="data/AUC.JPG" width="350" title="AUC">
</p>

- The ROC curve of a model that randomly guesses would be a random line, and the AUC would be 0.5

<p align="center">
  <img src="data/random_roc.JPG" width="350" title="ROC">
</p>

- Therefore, having an AUC of above 0.5 would be better then randomly guessing.While an AUC above 0.7 or 0.8 would indicate an well performing model.
- Together with the ROC curve the AUC allows us to easily evaluate and compare the performance of different classifiers. Inorder to plot the ROC, we need to first calcuate the probabilities that our model generates for each data point.
- scikit learn has a `predict_proba()` method for this purpose. Each column contains the probabilities for the respective target values.
- Since our positive class is "1", we are interested in the second column, and we can select that as shown below

```python
logreg.predict_proba(X_test)[:,1]
y_pred_prob = logreg.predict_proba(X_test)[:,1]
```

#### ROC curve in sklearn

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

import matplotlib.pyplot as plt

plt.plot(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot([0, 1], [0, 1], "k--")
plt.show()
```
- To calculate the AUC sklearn has `roc_auc_score` function


#### Precision-recall curve
Another way to evaluate model performance is using a precision-recall curve, which shows the tradeoff between precision and recall for different thresholds.

### Tuning model
- Choose best hyperparameters that leads to a best performing model.
- **Gridsearch** : is an bruteforce search to find the hyperparameter of interest. Essentially it tries a range of different hyperparameter values, fits model seperately using these different values, and then returns the hyperparameter that leads to the best model fit.
- scikit-learns technique of GridSearch uses cross validation to ensure the model is tested on unseen data.

#### Grid search in sklearn

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators' : np.arange(10, 51)} # dict of hyperparameters
clf_cv = GridSearchCV(RandomClassifier(), param_grid)
clf_cv.fit(X, y)
clf_cv.best_params_
clf_cv.best_score_
```

### Feature Importances
- Tree based methods such as Random forest allows us to calculate feature importances, which are scores representing how much each feature contributes to a prediction.
- Visualizing these feature importances is an effective way to communicate results to stakeholders.It can inform which features are important in driving churn and which features can be removed from the model.

#### Interpretability vs accuracy
- Different models have different strengths and weaknesses.For e.g, deep learning techniques tends to perform better in many domains, but the neural networks that underlie these techniques are very hard to interpret.
- In contrast, simpler techniques like logistic regression might not perform as well, but are easier to interpret, and this interpretability might be the difference between whether or not we buy-in from stakeholders, do we should'nt underestimate its importance.
- The value of a churn model is not only in helping us identify which customers are at risk of churning, but also what the drivers of churn are.

#### Random forest feature importances

```python
random_forest = RandomForestClassifer()
random_forest.fit(X_train, y_train)
random_forest.feature_importances_
```

### Adding New Features
- Adding additional features or data sources are easy way to improve model performance.Customer touch points are important sources of data to incorporate in churn modeling workflow. These sources provide a 360 degree view of the customer and should be inluded in our list of features.
- Data sources that we could incorporate into our model include: customer service call notes, web logs, email campaign, social networks, transcations and cell tower signal strength.Including these data sources can greatly improve model performance.

#### Additonal churn features
- We are doing this to potentially avoid underfitting by providing additional information to train the model with. Features that we will use are `Region code, Total charges, Total minutes, minutes per call, cost per call and total calls`










































































































































































