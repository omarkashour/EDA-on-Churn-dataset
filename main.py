import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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


# Split the dataset

X = df.drop(['id','Customer Value'], axis=1)
y = df['Customer Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print(f'Training set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}\n')

print("LRM1 MODEL")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_binary = (y_pred >= 0.5).astype(int)

# Evaluate the model
print("\n============= Model Evaluation =============")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))