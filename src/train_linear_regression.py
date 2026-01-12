import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

def train_lin_reg():
    print("--- MODEL 1: LINEAR REGRESSION ---")
    # 1. Load Clean Data
    df = pd.read_csv('TRAFFIC_CLEANED.csv')

    # 2. Setup (X = Inputs, y = Number of Cars)
    X = df[['hour', 'day_of_week', 'is_weekend']]
    y = df['total_vehicles']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Test
    preds = model.predict(X_test)
    error = mean_absolute_error(y_test, preds)
    print(f"Average Error: +/- {error:.2f} cars")

    # 5. Save
    joblib.dump(model, 'model_lin_reg.pkl')

if __name__ == "__main__":
    train_lin_reg()