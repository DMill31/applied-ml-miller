# Project 5
# Miller Ensemble Maching Learning

## Project Overview
This project focuses on being able to successfully create ensemble models to accurately predict wine quality.

---
## Steps

### Imports
Import numerous python libraries, including pandas, seaborn, and matplotlib

### 1. Load and Inspect the Data
Load the dataset into a pandas dataframe and display the first few rows.

### 2. Prepare the Data
The only change to the data made is creating new columns for the target, converting into both a categorical column and a numeric column.

### 3. Feature Selection and Justification
The target is wine quality, so we will be using the numeric column for quality created in section 2.
As for the inputs, we will be using every column except the quality columns.

Then the X and y variables are created.

### 4. Split the Data into Train and Test
We split the dataset into training and test sets using the train_test_split() method.

### 5. Evaluate Model Performance
Multiple models will be made, so we create a helper method to train and evaluate a model for us.
Then we call the helper model on two different ensemble models: AdaBoost and Bagging.
A confusion matrix as well as the training and testing accuracy and f1 scores are all shonw.

### 6. Compare Results
We create a table of results, showing the useful metrics from the models created.

### 7. Conclusions and Insights
Here a discussion on the difference in model performance is given based on their metrics.
Links to other notebooks with different ensemble models are provided, as the discussion goes over more than just the two models created here.

---
## How to Run the Project

### 1. Open the Notebook
This project's notebook can be found at:

notebooks/project05/ensemble-miller.ipynb

[notebook](https://github.com/DMill31/applied-ml-miller/blob/main/notebooks/project05/ensemble-miller.ipynb)

### 2. Activate the Virtual Environment & Select Kernel
In the terminal, run:

```shell
.\.venv\Scripts\activate
```

Once the virtual environment is up, at the top of the notebook select the kernel that goes with it

### 3. Run the Notebook
Either select the 'Run All' button at the top of the notebook or run the notebook cell by cell
