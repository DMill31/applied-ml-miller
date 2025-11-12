# Predicting a Continuous Target with Regression
**Author:** Dan Miller
**Date:** November 12th, 2025
**Objective:** Be able to predict a continuous numeric target, fare, using features in the Titanic dataset

## **Introduction**

This project explores the seaborn Titanic dataset. With the goal of being able to predict a continuous target with regression, linear regression as well as some alternative models will all be created and evaluated in terms of predicting fare, the price of the journey. The data will be explored and prepped before the models are created. There will be a summary at the end to discuss the findings.

## Imports

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
```

## Section 1. Import and Inspect the Data

```python
# Load the Titanic dataset
titanic = sns.load_dataset("titanic")

# Inspect the first few rows of the dataset
titanic.head()
```

       survived  pclass  sex   age  sibsp  parch     fare  embarked  class    who  \
    0         0       3    0  22.0      1      0   7.2500       2.0  Third    man   
    1         1       1    1  38.0      1      0  71.2833       0.0  First  woman   
    2         1       3    1  26.0      0      0   7.9250       2.0  Third  woman   
    3         1       1    1  35.0      1      0  53.1000       2.0  First  woman   
    4         0       3    0  35.0      0      0   8.0500       2.0  Third    man   
    
       adult_male deck  embark_town alive  alone  family_size  
    0        True  NaN  Southampton    no  False            2  
    1       False    C    Cherbourg   yes  False            2  
    2       False  NaN  Southampton   yes   True            1  
    3       False    C  Southampton   yes  False            2  
    4        True  NaN  Southampton    no   True            1  

## Section 2. Data Exploration and Preparation

```python
# Impute missing values for age using median
titanic["age"] = titanic["age"].fillna(titanic["age"].median())

# Drop rows with missing fare
titanic = titanic.dropna(subset=["fare"])

# Create a new feature for family size
titanic["family_size"] = titanic["sibsp"] + titanic["parch"] + 1

# Fill missing values in the embarked column with the most common value
titanic["embarked"] = titanic["embarked"].fillna(titanic["embarked"].mode()[0])

# Map categorical variable embarked to numerical values
titanic["embarked"] = titanic["embarked"].map({"C": 0, "Q": 1, "S": 2})
```