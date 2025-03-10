from data_preprocessing import load_data, summary_statistics, plot_distribution, plot_churn_by_age_group, plot_churn_by_charge_amount, charge_amount_statistics, plot_correlation_matrix
from model_training import preprocess_data, train_model, evaluate_model, plot_predictions_vs_actual
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
url = './Customer Churn.csv'
df = load_data(url)

# Perform data preprocessing and visualization
summary_statistics(df)
plot_distribution(df)
plot_churn_by_age_group(df)
plot_churn_by_charge_amount(df)
charge_amount_statistics(df)
plot_correlation_matrix(df)


df = df.drop('ID', axis=1)
# Prepare data for model training
preprocessor = preprocess_data(df)

# Model 1
print("================= LRM1 ====================")
X2 = df.drop(['Customer Value'], axis=1)
y = df['Customer Value']
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=42)
model = train_model(X_train, y_train, preprocessor)
y_pred = model.predict(X_test)
n = len(y_test)  # Number of observations
p = X2.shape[1]  # Number of predictors (features)

# Evaluate the model
evaluate_model(y_test, y_pred, n, p)
plot_predictions_vs_actual(y_test, y_pred)

# Model 2
print("======================= LRM2 =========================")
selected_features = ['Freq. of SMS', 'Freq. of use']
X2 = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=42)
model2 = LinearRegression()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
n = len(y_test)  # Number of observations
p = X2.shape[1]  # Number of predictors (features)

# Evaluate the model
evaluate_model(y_test, y_pred, n, p)
plot_predictions_vs_actual(y_test, y_pred)

# Model 3
print("======================= LRM3 =========================")
selected_features = ['Freq. of SMS', 'Freq. of use', 'Distinct Called Numbers','Call Failure']
X2 = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=42)
model3 = LinearRegression()
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
n = len(y_test)  # Number of observations
p = X2.shape[1]  # Number of predictors (features)

# Evaluate the model
evaluate_model(y_test, y_pred, n, p)
plot_predictions_vs_actual(y_test, y_pred)
