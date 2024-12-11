import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# load the dataset
url = './Customer Churn.csv'
df = pd.read_csv(url)

# summary
print(df.describe())


sns.countplot(x='Churn', data=df)
plt.title('Distribution of Churn')
plt.show()


# Define age groups
bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Histogram for each age group
sns.histplot(data=df, x='Age Group', hue='Churn', multiple='stack')
plt.title('Churn by Age Group')
plt.show()


# Histogram for each charge amount
sns.histplot(data=df, x='Charge Amount', hue='Churn', multiple='stack')
plt.title('Churn by Charge Amount')
plt.show()

# Summary statistics for charge amount
print(df['Charge Amount'].describe())

numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Split the dataset
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f'Training set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')


