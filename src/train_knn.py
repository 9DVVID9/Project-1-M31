import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_knn():
    print("--- MODEL 2: KNN CLASSIFIER ---")
    # 1. Load Clean Data
    df = pd.read_csv('TRAFFIC_CLEANED.csv')

    # 2. Setup (X = Inputs, y = Traffic Level)
    X = df[['hour', 'day_of_week', 'is_weekend']]
    y = df['traffic_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train (k=5 is standard)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # 4. Test
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc*100:.1f}%")

    # 5. Save
    joblib.dump(model, 'model_knn.pkl')

if __name__ == "__main__":
    train_knn()