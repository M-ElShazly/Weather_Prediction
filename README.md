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
2. **Exploratory Data Analysis:**

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
print(global_temp.head(20))
    
```

