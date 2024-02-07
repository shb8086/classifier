# Drug Classification

This Python script is used for the classification of drugs based on certain features. The script includes data loading, data exploration, data preprocessing, and model training.

## Requirements

- Python 3.x
- Libraries: NumPy, pandas, Matplotlib, scikit-learn, imbalanced-learn

## How to Run

1. Ensure that the required Python version and libraries are installed.
2. Place your dataset in the `dataset/` directory. The script expects the original dataset to be named `orig_dataset.csv`.
3. Run the script using a Python interpreter.
```
python script.py
```

## Script Details

The script performs the following steps:

### Data Loading

Loads the dataset from a CSV file in `dataset/orig_dataset.csv`

### Data Exploration

Displays basic information about the dataset, checks for missing values, displays the distribution of various features, and saves these distributions as plots. The plots can be find in `output/plots` directory.

### Data Preprocessing

Bins certain features into categories, performs one-hot encoding, splits the dataset into training and test sets, applies the SMOTE technique to avoid over-fitting, and scales the features. This is happening in `src/preprocessing.py` file.

### Model Training

The script includes commented-out code for training various models (Logistic Regression, K-NN, SVM, Kernel SVM, Naive Bayes, Decision Tree, Random Forest) and evaluating their performance. It also includes code for applying k-Fold Cross Validation and Grid Search to find the best model and the best parameters. This is happening in `src/classifier.py` file. The models are stored in `models` directory and models' evaluation are stores in csv file in `output/model_evaluation.csv`.
