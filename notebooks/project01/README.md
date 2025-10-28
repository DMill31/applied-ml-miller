# Project 1
# Miller California Housing Price Prediction

## Project Overview
This project focuses on creating a Linear Regression model using the California housing dataset from sklearn.

---

## Steps

### 1. Imports
Numerous python libraries were needed, including pandas, seaborn, and lots from sklearn

### 2. Load and Explore Data

### 2.1 Load the Dataset
The dataset can be found in scikit-learn, so we only needed to use the function we had already imported.

### 2.2 Check for Missing Values
Lucky for us, no values were missing from the dataset

### 3. Visualize Feature Distributions
In this section, multiple charts and graphs were created to help us see any possible relationships between the features.
Histograms, boxenplots, and scatter plots can be found in the notebook.

### 4. Feature Selection
At this point, we've seen the feature relationships, so it's time to choose our target feature as well as our predictors
Predictors:
- MedInc
- AveRooms
Target:
- MedHouseVal

### 5. Training
Now we create and train the linear regression model

### 5.1 Split the Data
We split the dataset into training and test sets, in this notebook we are going with an 80/20 split

### 5.2 Train the Model
We use the .fit() function to easily train the model on the training sets already created

### 5.3 Reports
In this section we calculate R^2, MAE, and RMSE to see how well our model performs.
Functions are already available for calculating these values so we simply need to call them.

---
## How to Run the Project

### 1. Open the Notebook
This project's notebook can be found at:

notebooks/project01/ml01.ipynb

### 2. Activate the Virtual Environment & Select Kernel
In the terminal, run:

```shell
.\.venv\Scripts\activate
```

Once the virtual environment is up, at the top of the notebook select the kernel that goes with it

### 3. Run the Notebook
Either select the 'Run All' button at the top of the notebook or run the notebook cell by cell
