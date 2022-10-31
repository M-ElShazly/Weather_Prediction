# Weather_Prediction

This repo contains the building, training, and testing of a Random Forest model to predict weather information based on historical data. 


## 1. Objectives

Build a model able to forecast weather information based on historical temparatrue values.

## 2. Data Sources

The data used in the repo is available through [This Link](https://github.com/M-ElShazly/Weather_Prediction/blob/main/GlobalTemp.csv), also as an attachement on this repo. 

# Code Steps & Explanation

1. **Importing Data:** 


```python
import pandas as pd
global_temp = pd.read_csv("GlobalTemp.csv")
```
2. **EDA:**
> Rather than a dedicated section for EDA, I'm inturrupting the repo every now and then with an `EDA` section to show case relevant information for better comperehension.

```python
print(global_temp.shape)
print(global_temp.columns)
print(global_temp.info())
print(global_temp.isnull().sum())
```
>This outputs the shape, columns and information about the data, also allows us to see if there's missing values. 

3. **Data Preparation:**

- Converting Temparature from celsius to fahrenheit which we're calling `converttemp` as follow:

    > Obviously this is an optional step, but since I was experimenting with all possible variations, added it here.


```python
def converttemp(x):
    x = (x * 1.8) + 32
    return float(x)
```
<br>
 
- Data Wrangling

For data wrangling, we're `dropping the columns we're not going to use`, `converting into datetime format`, `indexing by Year`, and `removing rows with null values` 


```python
def wrangle(df):
    df = df.copy()
    df = df.drop(columns=["LandAverageTemperatureUncertainty", "LandMaxTemperatureUncertainty","LandMinTemperatureUncertainty", "LandAndOceanAverageTemperatureUncertainty"], axis=1)
    df["LandAverageTemperature"] = df["LandAverageTemperature"].apply(converttemp)
    df["LandMaxTemperature"] = df["LandMaxTemperature"].apply(converttemp)
    df["LandMinTemperature"] = df["LandMinTemperature"].apply(converttemp)
    df["LandAndOceanAverageTemperature"] = df["LandAndOceanAverageTemperature"].apply(converttemp)
    df["dt"] = pd.to_datetime(df["dt"])
    df["Month"] = df["dt"].dt.month
    df["Year"] = df["dt"].dt.year
    df = df.drop("dt", axis=1)
    df = df.drop("Month", axis=1)
    df = df.set_index(["Year"])
    df = df.dropna()
    return df
global_temp = wrangle(global_temp)
print(global_temp.head()) 
```
- **EDA:**

Let's visualise the correlation between these variable that we kept in our data set. 
We're performting this step after data wrangling to make sure our results are true and consistent specifically because _NULL_ values would have inturrupted correlation. 

To visualise correlation we're using `seaborn`, and `matplotlib`.

```python
import seaborn as sns
import matplotlib.pyplot as plt
corrMatrix = global_temp.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
```
**OUTPUT:**

![Heatmap](https://user-images.githubusercontent.com/103464869/198922396-ea3e6337-a096-4d73-9312-79d3a071320f.jpg)

> This heatmap shows the predictable high correlation between the variable we're using here. 

- **Separating Target and Prediction Varible (x,y):**



```python
target = "LandAndOceanAverageTemperature"
y = global_temp[target]
x = global_temp[["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]]
```

- **Train/Test Split:**


```python
from sklearn.model_selection import train_test_split
xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.25, random_state=42)
print(xtrain.shape)
print(xval.shape)
print(ytrain.shape)
print(yval.shape)
```

- **Establishing a Baseline Mean Absolute Error:**

```python
from sklearn.metrics import mean_squared_error
ypred = [yval.mean()] * len(yval)
print("Baseline MAE: ", round(mean_squared_error(yval, ypred), 5))
```

- **Training the `Random Forest Model`:**


```python
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
forest = make_pipeline(
    SelectKBest(k="all"),
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        random_state=77,
        n_jobs=-1
    )
)
forest.fit(xtrain, ytrain)
```
- **Evaluting Model Accuracy:**

```python
import numpy as np
errors = abs(ypred - yval)
mape = 100 * (errors/ytrain)
accuracy = 100 - np.mean(mape)
print("Random Forest Model: ", round(accuracy, 2), "%")
```
 
**OUTPUT:**

   _Random Forest Model:  96.61 %_

3. - Clone the repo using this command in your terminal `git clone https://github.com/M-ElShazly/Weather_Prediction.git` 