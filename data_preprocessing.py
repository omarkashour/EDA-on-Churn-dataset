import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def summary_statistics(df):
    print("============= Summary Statistics =============")
    with pd.option_context('display.max_columns', 40):
        print(df.describe(include='all'))

def plot_distribution(df):
    sns.countplot(x='Churn', data=df)
    plt.title('Distribution of Churn')
    plt.show()

def plot_churn_by_age_group(df):
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    sns.histplot(data=df, x='Age Group', hue='Churn', multiple='stack')
    plt.title('Churn by Age Group')
    plt.show()

def plot_churn_by_charge_amount(df):
    sns.histplot(data=df, x='Charge Amount', hue='Churn', multiple='stack')
    plt.title('Churn by Charge Amount')
    plt.show()

def charge_amount_statistics(df):
    print("============= Charge Amount Statistics =============")
    print(df['Charge Amount'].describe())

def plot_correlation_matrix(df):
    encoded_df = pd.get_dummies(df, drop_first=True)
    corr_matrix = encoded_df.corr()
    plt.figure(figsize=(17, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix (All Attributes)')
    plt.show()
