# Project 4
# Miller Predicting a Continuous Target with Regression

## Project Overview
This project focuses on being able to predict a continuous numeric target, fare, from the Titanic dataset using multiple regression models

---
## Steps

### Imports
Import numerous python libraries, including pandas, seaborn, and matplotlib

### 1. Import and Inspect the Data

### 1.1 Import the necessary libraries
Load the dataset using the .load_dataset() function from seaborn and display the first few rows.

### 2. Data Exploration and Preparation
As we had missing values in this dataset, we decided to fill some of the missing ones with medians and modes, depending on the feature.
Part of machine learning can be making new features, so here we created a new feature 'family_size' as well as converting some categorical data to numeric so that the data works better with ML algorithms.

### 3. Feature Selection and Justification
We will have four different cases of inputs for this project, and they are as follows:
Case 1:
- input feature: age
- target: fare

Case 2:
- input feature: family_size
- target: fare

Case 3:
- input features: age, family_size
- target: fare

Case 4:
- input features: pclass, embarked
- target: fare

Then the four sets of X and y variables are created.

### 4. Train a Regression Model (Linear Regression)

### 4.1 Split the data
We split the dataset into training and test sets using the train_test_split() method for each case.

### 4.2 Train and Evaluate Linear Regression Models
In this section, four LR models are created (one for each case), using the .fit() method we train them, and lastly the .predict() method is used to train them.

### 4.3 Report Performance
We print the training R2 score, test R2 score, RMSE, and MAE for each model.

### 5. Compare Alternative Models
We create other regression models on the case that performed the best (Case 4).

### 5.1 Ridge Regression (L2 penalty)
Now we create a ridge model, train it, and make predictions.

### 5.2 Elastic Net (L1 + L2 combined)
Now we create an elastic net model, train it, and make predictions.

### 5.3 Polynomial Regression
Now we create a polynomial model with degree of three, train it, and make predictions.

### 5.4 Visualize Polynomial Cubic Fit (for 1 input feature)
We choose a case with one input feature (Case 2) and plot the polynomial fit.

### 5.5 Compare All Models
We print the R2 score, RMSE, and MAE for the four types of models created on Case 4.

### 5.6 Visualize Higher Order Polynomial (for same 1 inpute case)
Using the same case as the last plot (Case 2), we up the degrees to 5 and plot the polynomial fit.

### 6. Final Thoughts and Insights

### 6.1 Summarize Findings
Summary tables are created showing the metrics of the models created and a discussion is had about how well the models performed and why.

### 6.2 Discuss Challenges
A discussion regarding the target variable, skew, and outliers is had with regard to model complexity and difficulty to work with.

---
## How to Run the Project

### 1. Open the Notebook
This project's notebook can be found at:

notebooks/project04/ml04_miller.ipynb

[notebook](https://github.com/DMill31/applied-ml-miller/blob/main/notebooks/project04/ml04_miller.ipynb)

### 2. Activate the Virtual Environment & Select Kernel
In the terminal, run:

```shell
.\.venv\Scripts\activate
```

Once the virtual environment is up, at the top of the notebook select the kernel that goes with it

### 3. Run the Notebook
Either select the 'Run All' button at the top of the notebook or run the notebook cell by cell
