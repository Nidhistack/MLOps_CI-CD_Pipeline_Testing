import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "iris_model.pkl"

# Train a simple ML model and save it
def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved as {MODEL_PATH}")

# Load model and make a prediction
def make_prediction(sample):
    model = joblib.load(MODEL_PATH)  # Load the saved model
    prediction = model.predict([sample])
    return prediction.tolist()  # Convert numpy array to a Python list

# If running this file directly, train the model
if __name__ == "__main__":
    train_model()
