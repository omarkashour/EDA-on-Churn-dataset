import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# load the dataset
url = './Customer Churn.csv'
df = pd.read_csv(url)

# summary
print("============= Summary Statistics =============")
with pd.option_context('display.max_columns', 40):
    print(df.describe(include='all'))


# sns.countplot(x='Churn', data=df)
# plt.title('Distribution of Churn')
# plt.show()
#
#
# # Define age groups
# bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
# df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
#
# # Histogram for each age group
# sns.histplot(data=df, x='Age Group', hue='Churn', multiple='stack')
# plt.title('Churn by Age Group')
# plt.show()
#
#
# # Histogram for each charge amount
# sns.histplot(data=df, x='Charge Amount', hue='Churn', multiple='stack')
# plt.title('Churn by Charge Amount')
# plt.show()
#
# # Summary statistics for charge amount
# print("============= Charge Amount Statistics =============")
# print(df['Charge Amount'].describe())
#
# numeric_df = df.select_dtypes(include=['number'])
#
# # Calculate the correlation matrix
# corr_matrix = numeric_df.corr()
#
# # Heatmap of the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()


# Drop irrelevant or non-numeric columns for regression

print("================= LRM1 ==================== ")


df = df.drop(['ID', 'Status', 'Churn'], axis=1)

# Convert categorical columns to numeric using OneHotEncoding
categorical_features = ['Complains', 'Plan', 'Age Group']
numerical_features = ['Call Failure', 'Charge Amount', 'Freq. of use',
                      'Freq. of SMS', 'Distinct Called Numbers', 'Age']

# Define preprocessor for column transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Define pipeline for preprocessing and regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the dataset into features (X) and target variable (y)
X2 = df.drop(['Customer Value'], axis=1)
y = df['Customer Value']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=42)

# Train the linear regression model
pipeline.fit(X_train, y_train)

# Predict the Customer Value on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n============= Model Evaluation =============")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Predicted vs Actual Customer Value')
plt.xlabel('Actual Customer Value')
plt.ylabel('Predicted Customer Value')
plt.show()



print("======================= LRM2 =========================")
selected_features = ['Freq. of SMS', 'Freq. of use']
X2 = df[selected_features]
y = df['Customer Value']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the Customer Value on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n============= Model Evaluation =============")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Predicted vs Actual Customer Value')
plt.xlabel('Actual Customer Value')
plt.ylabel('Predicted Customer Value')
plt.show()


print("======================= LRM3 =========================")
selected_features = ['Freq. of SMS', 'Freq. of use', 'Distinct Called Numbers','Call Failure']
X2 = df[selected_features]
y = df['Customer Value']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the Customer Value on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n============= Model Evaluation =============")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Predicted vs Actual Customer Value')
plt.xlabel('Actual Customer Value')
plt.ylabel('Predicted Customer Value')
plt.show()


print("======================= KNN =========================")
