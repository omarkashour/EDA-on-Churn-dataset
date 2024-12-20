import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    # Check for missing values
    print(df.isnull().sum())

    # Convert to numeric where applicable
    numeric_columns = ['Age', 'Call Failure', 'Complains', 'Freq. of use', 'Freq. of SMS', 'Distinct Called Numbers',
                       'Charge Amount', 'Customer Value']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Convert 'Churn' to binary
    df['Churn'] = df['Churn'].map({'yes': 1, 'no': 0})

    # # Feature engineering for Age Group
    # bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    # df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Normalize or standardize numeric features
    scaler = StandardScaler()
    df[['Charge Amount', 'Freq. of use', 'Freq. of SMS', 'Customer Value']] = scaler.fit_transform(
        df[['Charge Amount', 'Freq. of use', 'Freq. of SMS', 'Customer Value']])

    return df


def summary_statistics(df):
    print("============= Summary Statistics =============")
    with pd.option_context('display.max_columns', 40):
        print(df.describe(include='all'))


def plot_distribution(df):
    sns.countplot(x='Churn', data=df)
    plt.title('Distribution of Churn')
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_churn_by_age_group(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()

    # Define age bins and labels
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

    # Assign age group labels to a new column in the copied DataFrame
    df_copy['Age Group'] = pd.cut(df_copy['Age'], bins=bins, labels=labels, right=False)

    # Histogram for each age group
    sns.histplot(data=df_copy, x='Age Group', hue='Churn', multiple='stack')
    plt.title('Churn by Age Group')
    plt.show()


def plot_churn_by_charge_amount(df):
    sns.histplot(data=df, x='Charge Amount', hue='Churn', multiple='dodge', legend=True)
    plt.title('Churn by Charge Amount')
    plt.show()


def charge_amount_statistics(df):
    print("============= Charge Amount Statistics =============")
    print(df['Charge Amount'].describe())


def plot_correlation_matrix(df):
    # One-hot encode categorical variables
    encoded_df = pd.get_dummies(df, drop_first=True)
    # Now compute correlation
    corr_matrix = encoded_df.corr()
    plt.figure(figsize=(17, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix (All Attributes)')
    plt.show()


def main(file_path):
    df = load_data(file_path)
    df = preprocess_data(df)
    summary_statistics(df)
    plot_distribution(df)
    plot_churn_by_age_group(df)
    plot_churn_by_charge_amount(df)
    charge_amount_statistics(df)
    plot_correlation_matrix(df)


