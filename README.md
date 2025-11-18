# applied-ml-miller

## About this Repository

This repository contains the beginner folders and files for Applied Machine Learning projects.  There will be consistent updates regarding future projects.

## Project 1

For week 1, the project can be found at:

- notebooks/project01/ml01.ipynb

Project 1 focuses on creating a Linear Regression model using the California housing dataset from sklearn.
- Visualizations are created to show relationships between features.
- Two predictors are chosen from the features to train the model.
- Afterwards, we evaluate the model using the calculated metrics: R^2, MAE, and RMSE.


To run the project:
```shell
uv run python notebooks/project01/ml01.ipynb
```

## Project 2

For week 2, the project can be found at:

- notebooks/project02/ml02_miller.ipynb

Project 2 focuses on exploring the data from seaborn's Titanic dataset.
- Visualizations are created to show relationships between features.
- Through feature engineering, new features are made, and missing values are handled.
- Different ways of splitting the data are explored.

To run the project:
```shell
uv run python notebooks/project02/ml02_miller.ipynb
```

## Project 3

For week 3, the project can be found at:

- notebooks/project03/ml03_miller.ipynb

Project 3 focuses on exploring different types of classification models
- Decision Trees with confusion matrices are created
- SVC with visuals showing support vectors
- A Neural Network Decision Surface is made
- A Summary is then presented to discuss the differences between the models

To run the project:
```shell
uv run python notebooks/project03/ml03_miller.ipynb
```

## Project 4

For week 4, the project can be found at:

- notebooks/project04/ml04_miller.ipynb

Project 4 focuses on models used to predict a continuous target
- Linear regression models are made for four different cases
- Ridge regression and Elsatic Net are both created for the strongest case
- Polynomial regression is made and visuals are created
- A summary is presented at the end to compare the models

To run the project:
```shell
uv run python notebooks/project04/ml04_miller.ipynb
```

## Project 5

For week 5, the project can be found at:

- notebooks/project05/ensemble-miller.ipynb

Project 5 focuses on ensemble models ability to predict wine quality
- Helper functions are created to clean the data easily
- All features aside from the target are input features
- 2 different ensemble models are created
- A summary and conclusion are at the end to show comparisons

To run the project:
```shell
uv run python notebooks/project05/ensemble-miller.ipynb
```

## WORKFLOW 1. Set Up Machine

Proper setup is critical.
Complete each step in the following guide and verify carefully.

- [SET UP MACHINE](./SET_UP_MACHINE.md)

---

## WORKFLOW 2. Set Up Project

After verifying your machine is set up, set up a new Python project.
Complete each step in the following guide.

- [SET UP PROJECT](./SET_UP_PROJECT.md)

It includes the critical commands to set up your local environment (and activate it):

```shell
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
```

**Windows (PowerShell):**

```shell
.\.venv\Scripts\activate
```

**macOS / Linux / WSL:**

```shell
source .venv/bin/activate
```

Lastly, while in VS Code, it is important that we have our correct interpreter.
1. Press Ctrl+Shift+P
2. Search for "Python: Select Interpreter"
3. Choose the Interpreter for the local .venv

---

## WORKFLOW 3. Daily Workflow

Please ensure that the prior steps have been verified before continuing.
When working on a project, we open just that project in VS Code.

### 3.1 Git Pull from GitHub

Always start with `git pull` to check for any changes made to the GitHub repo.

```shell
git pull
```

### 3.2 Run Checks as You Work

This mirrors real work where we typically:

1. Update dependencies (for security and compatibility).
2. Clean unused cached packages to free space.
3. Use `git add .` to stage all changes.
4. Run ruff and fix minor issues.
5. Update pre-commit periodically.
6. Run pre-commit quality checks on all code files (**twice if needed**, the first pass may fix things).
7. Run tests.

In VS Code, open your repository, then open a terminal (Terminal / New Terminal) and run the following commands one at a time to check the code.

```shell
git pull
uv sync --extra dev --extra docs --upgrade
uv cache clean
git add .
uvx ruff check --fix
uvx pre-commit autoupdate
uv run pre-commit run --all-files
git add .
uv run pytest
```

NOTE: The second `git add .` ensures any automatic fixes made by Ruff or pre-commit are included before testing or committing.
Running `uv run pre-commit run --all-files` twice may be helpful if the first time doesn't pass. 

<details>
<summary>Click to see a note on best practices</summary>

`uvx` runs the latest version of a tool in an isolated cache, outside the virtual environment.
This keeps the project light and simple, but behavior can change when the tool updates.
For fully reproducible results, or when you need to use the local `.venv`, use `uv run` instead.

</details>

### 3.3 Build Project Documentation

Make sure you have current doc dependencies, then build your docs, fix any errors, and serve them locally to test.

```shell
uv run mkdocs build --strict
uv run mkdocs serve
```

- After running the serve command, the local URL of the docs will be provided. To open the site, press **CTRL and click** the provided link (at the same time) to view the documentation. On a Mac, use **CMD and click**.
- Press **CTRL c** (at the same time) to stop the hosting process.

### 3.4 Execute

This project includes demo code.
Run the demo Python modules to confirm everything is working.

In VS Code terminal, run:

```shell
uv run python notebooks/project01/ml01.py
```

A new window with charts should appear. Close the window to finish the execution. 
If this works, your project is ready! If not, check:

- Are you in the right folder? (All terminal commands are to be run from the root project folder.)
- Did you run the full `uv sync --extra dev --extra docs --upgrade` command?
- Are there any error messages? (ask for help with the exact error)

### 3.5 Git add-commit-push to GitHub

Anytime we make working changes to code is a good time to git add-commit-push to GitHub.

1. Stage your changes with git add.
2. Commit your changes with a useful message in quotes.
3. Push your work to GitHub.

```shell
git add .
git commit -m "describe your change in quotes"
git push -u origin main
```

This will trigger the GitHub Actions workflow and publish your documentation via GitHub Pages.

### 3.6 Modify and Debug

With a working version safe in GitHub, start making changes to the code.

Before starting a new session, remember to do a `git pull` and keep your tools updated.

Each time forward progress is made, remember to git add-commit-push.
