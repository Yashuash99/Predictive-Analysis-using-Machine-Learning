# mushroom_classifier.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Step 1: Load Dataset
df = pd.read_csv('data/mushrooms.csv')
print("First 5 rows of dataset:\n", df.head())

# Step 2: Encode Categorical Columns
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Step 3: Split Features and Target
X = df.drop('class', axis=1)
y = df['class']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions & Evaluation
y_pred = model.predict(X_test)

print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Step 8: Save the Model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/mushroom_model.pkl')
print("✅ Model saved to 'models/mushroom_model.pkl'")

# Step 9: Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh', title='Top 10 Important Features')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
