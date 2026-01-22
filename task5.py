import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load dataset
data = pd.read_csv("heart.csv")

# Convert target column to binary
data["num"] = data["num"].apply(lambda x: 1 if x > 0 else 0)

# Convert categorical columns to numeric (One-Hot Encoding)
data = pd.get_dummies(data, drop_first=True)

# Handle missing values (fill NaN with column mean)
data = data.fillna(data.mean())

# Split features and target
X = data.drop("num", axis=1)
y = data["num"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
