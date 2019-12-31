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
- Goal is to predict whether or not customer will churn depending on various features.
- 




































































