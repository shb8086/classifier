import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score
import joblib

# Create DataFrames to store model evaluation results
df_model = pd.DataFrame(columns=["Model", "Train Accuracy", "Test Accuracy", "Test Loss", "Cross Validation Mean", "Cross Validation Variance", "Trained Model", "Model File Path"])

def train_and_evaluate_model(model_name, classifier, X_train, y_train, X_test, y_test):
  # store the fitted model data for later use
  trained_model = classifier.fit(X_train, y_train)
  # save the trained model
  model_file_path = f"models/{model_name}_model.pkl"
  joblib.dump(trained_model, model_file_path)

  # Evaluate the model using k-fold cross validation
  cross_val_results = cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring="accuracy")
  cross_val_mean = cross_val_results.mean()
  cross_val_variance = cross_val_results.std()

  # Get the accuracy, loss, and trained model
  train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
  test_accuracy = accuracy_score(y_test, classifier.predict(X_test))
  test_loss = log_loss(y_test, classifier.predict_proba(X_test))

  # Add the model evaluation results to the df_model
  df_model.loc[len(df_model)] = [model_name, train_accuracy, test_accuracy, test_loss, cross_val_mean, cross_val_variance, classifier, model_file_path]

  print(f"{model_name} model trained and saved.")


if __name__ == "__main__":
  # Load the preprocessed training and test sets
  X_train = np.load("dataset/preprocessed/X_train.npy", allow_pickle=True)
  y_train = np.load("dataset/preprocessed/y_train.npy", allow_pickle=True)
  X_test = np.load("dataset/preprocessed/X_test.npy", allow_pickle=True)
  y_test = np.load("dataset/preprocessed/y_test.npy", allow_pickle=True)
  print("Preprocessed training and test sets loaded.")

  # Models to train and evaluate
  models = [
    ("Logistic_Regression", LogisticRegression(random_state=0)),
    ("K_NN", KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)),
    ("SVM", SVC(kernel="linear", random_state=0, probability=True)),
    ("Kernel_SVM", SVC(kernel="rbf", random_state=0, probability=True)),
    ("Naive_Bayes", GaussianNB()),
    ("Decision_Tree", DecisionTreeClassifier(criterion="entropy", random_state=0)),
    ("Random_Forest", RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0))
  ]

  for model_name, model in models:
    train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test)

  # Plot the model evaluation results after k-fold cross validation
  plt.figure(figsize=(10, 6))
  plt.bar(df_model["Model"], df_model["Cross Validation Mean"], yerr=df_model["Cross Validation Variance"], color="blue")
  plt.xlabel("Model")
  plt.grid(axis="y")
  plt.ylabel("Cross Validation Mean")
  plt.title("Model Evaluation after k-fold Cross Validation")
  plt.tight_layout()
  plt.savefig("output/plots/model_evaluation_k_fold.png")
  print("Model evaluation plot saved.")

  # Save the df_model
  df_model.to_csv("output/model_evaluation.csv", index=False)
  print("Model evaluation saved.")

  print("All done.")