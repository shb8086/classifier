import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Display the distributions
def plot_distribution(df, column_name, file_name):
  plt.figure()
  plt.bar(df[column_name].unique(), df[column_name].value_counts())
  plt.title(f"{column_name.capitalize()} Distribution")
  plt.xlabel(column_name.capitalize())
  plt.ylabel("Count")
  plt.xticks(rotation=45)
  # Write counts on the bars
  if column_name != "age":
    for i, count in enumerate(df[column_name].value_counts()):
      plt.text(i, count, str(count), ha='center', va='bottom')
  plt.savefig(f"../plots/{file_name}.png")

# Display the distribution col1 based on other col2
def plot_based_distribution(df, col1, col2):
  pd.crosstab(df[col1], df[col2]).plot(kind="bar", figsize=(10, 6))
  plt.title(f'{col1.capitalize()} distribution based on {col2.capitalize()}')
  plt.xlabel(col1.capitalize())
  plt.grid(axis='y')
  plt.xticks(rotation=0)
  plt.ylabel('Frequency')
  plt.savefig(f"../plots/{col1}_{col2}_distribution_chart.png")

# Load the dataset
df = pd.read_csv("../dataset/orig_dataset.csv")

# Display information of the dataset
print("Dataset shape:")
print(df.shape, end="\n\n")
print("\nFirst 5 rows of the dataset:")
print(df.head(), end="\n\n")
print("\nLast 5 rows of the dataset:")
print(df.tail(), end="\n\n")
print("Dataset information:")
print(df.info(), end="\n\n")

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum(), end="\n\n")

print(df.gender.value_counts(), end="\n\n")
print(df.blood_pressure.value_counts(), end="\n\n")
print(df.cholesterol.value_counts(), end="\n\n")

# Display the of target values in the dataset
print("Drug type count:")
print(df.drug_type.value_counts(), end="\n\n")

# Display the statistics of the dataset
print("Statistics of the dataset:")
print(df.describe(), end="\n\n")

# Display the distribution of the drug_type (target variable)
plot_distribution(df, "drug_type", "drug_type_distribution_chart")

# Display the distribution of gender
plot_distribution(df, "gender", "gender_distribution_chart")
# Display the distribution of cholesterol
plot_distribution(df, "cholesterol", "cholesterol_distribution_chart")
# Display the distribution of blood pressure
plot_distribution(df, "blood_pressure", "blood_pressure_distribution_chart")
# Display the distribution of age
plot_distribution(df, "age", "age_distribution_chart", )

# Display the distribution of gender based on drug type
plot_based_distribution(df, "gender", "drug_type")
# Display the distribution of cholesterol based on drug type
plot_based_distribution(df, "cholesterol", "drug_type")

# Display the distribution of blood pressure based on cholesterol
plot_based_distribution(df, "blood_pressure", "cholesterol")

# Display the distribution of sodium to potassium ratio based on age and gender
plt.figure()
plt.scatter(x=df.age[df.gender=='F'], y=df.Na_to_K_rate[(df.gender=='F')], c="Red")
plt.scatter(x=df.age[df.gender=='M'], y=df.Na_to_K_rate[(df.gender=='M')], c="Blue")
plt.legend(["Female", "Male"], loc="upper right", fontsize=8)
# plt.yscale("log")
plt.grid(True)
plt.xlabel("Age")
plt.ylabel("Na_to_K")
plt.title("Na_to_K ratio based on Age and Gender")
plt.savefig("../plots/Na_to_K_ratio_based_on_Age_gender.png")

# Dataset Preparation

# Data binning
# Binning the AGE column into 10 categories
bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# 1: >0 and <=20
# 2: >20 and <=30
# 3: >30 and <=40
# 4: >40 and <=50
# 5: >50 and <=60
# 6: >60 and <=70
# 7: >70 and <=80
# 8: >80 and <=90
# 9: >90 and <=100
labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]
df["age_binned"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
# Drop the age column
df.drop("age", axis=1, inplace=True)

# Binning the Na_to_K column into 5 categories
bins=[0, 10, 20, 30, 100]
# 1: >0 and <=10
# 2: >10 and <=20
# 3: >20 and <=30
# 4: >30
labels=[1, 2, 3, 4]
df["Na_to_K_binned"] = pd.cut(df["Na_to_K_rate"], bins=bins, labels=labels, right=False)
# Drop the Na_to_K_rate column
df.drop("Na_to_K_rate", axis=1, inplace=True)

# Print the first 5 rows of the changed dataset
print("\nFirst 5 rows of the modified dataset:")
print(df.head(), end="\n\n")

# Save the modified dataset
df.to_csv("../dataset/modified_dataset.csv", index=False)
print("Modified dataset saved successfully!", end="\n\n")

# Splitting the dataset into features and target variable
X = df.drop("drug_type", axis=1)
y = df["drug_type"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display the shape of the training and test sets
print("Training set shape:")
print(X_train.shape, y_train.shape, end="\n\n")
print("Test set shape:")
print(X_test.shape, y_test.shape, end="\n\n")

# It is done in line 133
# # One-hot encoding
# X_train = pd.get_dummies(X_train)
# X_test = pd.get_dummies(X_test)

# Print the first 5 rows of the one-hot encoded training set
print("\nFirst 5 rows of the one-hot encoded training set:")
print(X_train.head(), end="\n\n")

# SMOTE Technique : Since the number of drug type ‘D’ is more than other types of drugs, oversampling is carried out to avoid overfitting.
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Display the shape of the training set after SMOTE
print("Training set shape after SMOTE:")
print(X_train.shape, y_train.shape, end="\n\n")

# Display the distribution of the drug_type (target variable) after SMOTE
plot_distribution(pd.DataFrame(y_train, columns=["drug_type"]), "drug_type", "drug_type_distribution_chart_after_SMOTE")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the preprocessed training and test sets
np.save("../dataset/preprocessed/X_train.npy", X_train)
np.save("../dataset/preprocessed/y_train.npy", y_train)
np.save("../dataset/preprocessed/X_test.npy", X_test)
np.save("../dataset/preprocessed/y_test.npy", y_test)
print("Preprocessed training and test sets saved successfully!", end="\n\n")

# # Load the preprocessed training and test sets
# X_train = np.load("../dataset/preprocessed/X_train.npy")
# y_train = np.load("../dataset/preprocessed/y_train.npy")
# X_test = np.load("../dataset/preprocessed/X_test.npy")
# y_test = np.load("../dataset/preprocessed/y_test.npy")
# print("Preprocessed training and test sets loaded successfully!", end="\n\n")




# # Models
# # Training the Logistic Regression model on the Training set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state=0)
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Training the K-NN model on the Training set
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Training the SVM model on the Training set
# from sklearn.svm import SVC
# classifier = SVC(kernel="linear", random_state=0)
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Training the Kernel SVM model on the Training set
# from sklearn.svm import SVC
# classifier = SVC(kernel="rbf", random_state=0)
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Training the Naive Bayes model on the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Training the Decision Tree Classification model on the Training set
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Training the Random Forest Classification model on the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm, end="\n\n")
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}", end="\n\n")

# # Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
# print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracies.std()*100), end="\n\n")

# # Applying Grid Search to find the best model and the best parameters
# from sklearn.model_selection import GridSearchCV
# parameters = [
#   {"n_estimators": [10, 100, 1000], "criterion": ["gini", "entropy"]}
# ]
# grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10, n_jobs=-1)
# grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print(f"Best Accuracy: {best_accuracy}", end="\n\n")
# print(f"Best Parameters: {best_parameters}", end="\n\n")

# # Save the best model
# import joblib
# joblib.dump(grid_search, "best_model.pkl")
# print("Best model saved successfully!", end="\n\n")

# # Load the best model
# best_model = joblib.load("best_model.pkl")
# print("Best model loaded successfully!", end="\n\n")

# # Predicting the Test set results using the best model
# y_pred = best_model.predict(X_test)







