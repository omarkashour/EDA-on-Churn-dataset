import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = './Customer Churn.csv'  # Replace with the path to your dataset
data = pd.read_csv(url)  # Load data

# Preprocess the data
X = data.drop(columns=['ID', 'Churn'])  # Features (exclude ID and Churn)
y = data['Churn'].map({'yes': 1, 'no': 0})  # Target variable (convert 'yes'/'no' to 1/0)

# Encode categorical variables (e.g., 'Plan', 'Status')
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for k-NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on test set for k-NN
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
roc_auc_knn = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])

# Initialize and train Naive Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict on test set for Naive Bayes
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
roc_auc_nb = roc_auc_score(y_test, nb.predict_proba(X_test)[:, 1])

# Initialize and train Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict on test set for Decision Tree
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])

# Initialize and train Logistic Regression classifier
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Predict on test set for Logistic Regression
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

# Create a summary table for model results
results_summary = {
    'Model': ['k-NN', 'Naive Bayes', 'Decision Tree', 'Logistic Regression'],
    'Accuracy': [accuracy_knn, accuracy_nb, accuracy_dt, accuracy_lr],
    'ROC AUC': [roc_auc_knn, roc_auc_nb, roc_auc_dt, roc_auc_lr],
}

# Create a DataFrame from the results
results_df = pd.DataFrame(results_summary)

# Display the table in a clean format
plt.figure(figsize=(8, 4))
sns.heatmap(results_df.set_index('Model').T, annot=True, fmt='.4f', cmap="Blues", cbar=False, linewidths=0.5)
plt.title('Model Performance Comparison')
plt.yticks(rotation=0)  # Rotate row labels to make them horizontal
plt.show()

# Optionally, you can print the raw DataFrame to view the results in the console
print(results_df)

# Results Comparison
results = {
    'Model': ['Logistic Regression', 'Naive Bayes', 'k-NN', 'Decision Tree'],
    'Accuracy': [accuracy_lr, accuracy_nb, accuracy_knn, accuracy_dt],
    'ROC AUC': [roc_auc_lr, roc_auc_nb, roc_auc_knn, roc_auc_dt]
}
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Generate confusion matrix for each model
plot_confusion_matrix(conf_matrix_knn, 'k-NN')
plot_confusion_matrix(conf_matrix_nb, 'Naive Bayes')
plot_confusion_matrix(conf_matrix_dt, 'Decision Tree')
plot_confusion_matrix(conf_matrix_lr, 'Logistic Regression')

# Plot ROC curves for each classifier
def plot_roc(fpr, tpr, label):
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {label}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random classifier line
    plt.legend()
    plt.show()

# k-NN ROC curve
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])
plot_roc(fpr_knn, tpr_knn, 'k-NN')

# Naive Bayes ROC curve
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb.predict_proba(X_test)[:, 1])
plot_roc(fpr_nb, tpr_nb, 'Naive Bayes')

# Decision Tree ROC curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt.predict_proba(X_test)[:, 1])
plot_roc(fpr_dt, tpr_dt, 'Decision Tree')

# Logistic Regression ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:, 1])
plot_roc(fpr_lr, tpr_lr, 'Logistic Regression')

# Combined ROC curve plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label='k-NN')
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes')
plt.plot(fpr_dt, tpr_dt, label='Decision Tree')
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve Comparison')
plt.legend()
plt.show()
