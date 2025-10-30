# **Titanic Data Features**
**Author:** Dan Miller 
**Date:** October 30th, 2025  
**Objective:** Explore the Titanic dataset to determine what factors impacted survival

## **Introduction**

This project explores the Titanic dataset available to us from the seaborn library. We will explore its features and determine if it needs to be cleaned or transformed. Once prepping the data is complete, we will select the features to be used as predictors and our target. In this notebook, 5 features will be used as predictors to predict the target variable survived. Lastly, we create training and testing sets, then examine those in terms of size and distribution.

## **Section 1. Import and Inspect the Data**


```python
# Imports
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

```


```python
# Load the data
titanic = sns.load_dataset('titanic')
```

Display the first few rows of the dataset


```python
print(titanic.head(10))
```
