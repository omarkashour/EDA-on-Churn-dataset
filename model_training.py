import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)


def preprocess_data(df):
    categorical_features = ['Complains', 'Plan','Status' ,'Churn']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)

        ],
        remainder='passthrough'
    )

    return preprocessor


def train_model(X_train, y_train, preprocessor):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(y_test, y_pred, n, p):
    # Calculate basic metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Calculate Adjusted R² Score
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    print("\n============= Model Evaluation =============")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Adjusted R² Score: {adjusted_r2:.2f}")
    print(f"Explained Variance Score: {explained_var:.2f}")

def plot_predictions_vs_actual(y_test, y_pred):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title('Predicted vs Actual Customer Value')
    plt.xlabel('Actual Customer Value')
    plt.ylabel('Predicted Customer Value')
    plt.show()
