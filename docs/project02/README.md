# Project 2
# Miller Titanic Dataset Data Features

## Project Overview
This project focuses on exploring the Titanic dataset's data and seeing what features impacted survivability

---
## Steps

### 1. Import and Inspect the Data

### 1.1 Import the necessary libraries
Numerous python libraries were needed, including pandas, seaborn, sklearn

### 1.2 Load the Dataset
The dataset can be found in seaborn, so we only needed to load the dataset using the .load_dataset() function.

### 1.3 Check for Missing Values
There were a handful of missing values, most notably the feature 'deck' was missing the most.  Thankfully, there were fourteen other features we could work with.

### 2. Data Exploration and Preparation

### 2.1 Explore Data Patterns and Distributions
In this section, multiple charts and graphs were created to help us see any possible relationships between the features.
Histograms, scatter charts, and scatter plots can be found in the notebook.

### 2.2 Handle Missing Values and Clean Data
As we had missing values in this dataset, we decided to fill some of the missing ones with medians and modes, depending on the feature.

### 2.3 Feature Engineering
Part of machine learning can be making new features, so here we created a new feature 'family_size' as well as converting some categorical data to numeric so that the data works better with ML algorithms.

### 3. Feature Selection and Justification

### 3.1 Feature Selection
At this point, we've seen the feature relationships, so it's time to choose our target feature as well as our predictors
Predictors:
- age
- fare
- pclass
- sex
- family_size
Target:
- survived

### 3.2 Define X and y
Here all we did was create variables X and y that we will be using in our splitting.

### 4. Splitting

### 4.1 Basic Train/Test Split
We split the dataset into training and test sets using the train_test_split() function from sklearn.

### 4.2 Stratified Train/Test Split
We split the dataset into training and test sets using a splitter created from the StratifiedShuffleSplit() function from sklearn.

### 4.3 Compare Results
In this section we compare the training and test sets between the basic split and the stratified split to see how well they resemble the original data's distribution.

---
## How to Run the Project

### 1. Open the Notebook
This project's notebook can be found at:

notebooks/project02/ml02_miller.ipynb

### 2. Activate the Virtual Environment & Select Kernel
In the terminal, run:

```shell
.\.venv\Scripts\activate
```

Once the virtual environment is up, at the top of the notebook select the kernel that goes with it

### 3. Run the Notebook
Either select the 'Run All' button at the top of the notebook or run the notebook cell by cell
