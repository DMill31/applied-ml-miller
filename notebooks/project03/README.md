# Project 3
# Miller Building a Classifier

## Project Overview
This project focuses on building and evaluating three classifiers using the Titanic dataset, then comparing their performance across different feature sets in terms of predicting survival

---
## Steps

### 1. Import and Inspect the Data

### 1.1 Import the necessary libraries
Numerous python libraries were needed, including pandas, seaborn, sklearn

### 1.2 Load the Dataset
The dataset can be found in seaborn, so we only needed to load the dataset using the .load_dataset() function.

### 2. Data Exploration and Preparation

### 2.1 Handle Missing Values and Clean Data
As we had missing values in this dataset, we decided to fill some of the missing ones with medians and modes, depending on the feature.

### 2.2 Feature Engineering
Part of machine learning can be making new features, so here we created a new feature 'family_size' as well as converting some categorical data to numeric so that the data works better with ML algorithms.

### 3. Feature Selection and Justification

### 3.1 Feature Selection
We will have three different cases of inputs for this project, and they are as follows:
Case 1:
- input feature: alone
- target: survived

Case 2:
- input feature: age
- target: survived

Case 3:
- inpute features: age, family_size
- target: survived

### 3.2 Define X and y
Here we created three sets of X and y variables that we will be using in our splitting.

### 4. Train a Classification Model (Decision Tree)

### 4.1 Split the data
We split the dataset into training and test sets using the StratifiedShuffleSplit method for each case.

### 4.2 Create and Train Model (Decision Tree)
In this section, three tree models are created (one for each case) and using the .fit() method we train them.

### 4.3 Predict and Evaluate Model Performance
Now that the decision trees are trained, we can see how they do, so we make predictions and get the classification reports.

### 4.4 Report Confusion Matrix (as a heatmap)
Another way of seeing how the model performs is by creating a confusion matrix, so here we make one for each case.

### 4.5 Report Decision Tree Plot
Now we finally show the decision trees themselves by using the plot_tree method.

### 5. Compare Alternative Models (SVC, NN)

### 5.1 Train and Evaluate Model (SVC)
Now instead of decision trees, we're creating support vectors, so again we create the model using SVC(), train and predict, then get the classification reports.

### 5.2 Visualize Support Vectors
A scatter plot for each case is made showing the survived, not survived, and all the support vectors.

### 5.3 Train and Evaluate Model (Neural Network on Case 3)
Just like the previous models, we create a neural network using MLPClassifier, but this time we only make one for Case 3.  The model is trained, then we predict, print the classification report and create a confusion matrix.

### 5.4 Visualize (Neural Network on Case 3)
Here a decision surface is created to visually show how the neural network decides survival.

### 6. Final Thoughts and Insights
A summary table is shown with the stats from all the models created in this notebook.

---
## How to Run the Project

### 1. Open the Notebook
This project's notebook can be found at:

notebooks/project03/ml03_miller.ipynb

[notebook](https://github.com/DMill31/applied-ml-miller/blob/main/notebooks/project03/ml03_miller.ipynb)

### 2. Activate the Virtual Environment & Select Kernel
In the terminal, run:

```shell
.\.venv\Scripts\activate
```

Once the virtual environment is up, at the top of the notebook select the kernel that goes with it

### 3. Run the Notebook
Either select the 'Run All' button at the top of the notebook or run the notebook cell by cell
