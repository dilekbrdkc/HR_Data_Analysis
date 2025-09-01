# Gerekli kütüphaneleri yükle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\dilek\Desktop\HR_Data_Analysis\WA_Fn-UseC_-HR-Employee-Attrition.csv")


df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

le = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])


X = df.drop('Attrition', axis=1)
y = df['Attrition']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
#     Decision Tree Model
# ==============================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)


dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)

print("=== Decision Tree Classifier ===")
print(classification_report(y_test, y_pred_dt))

# ==============================
#   Logistic Regression Modeli
# ==============================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)


lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))

# ==============================
#  Results 
# ==============================
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Logistic Regression'],
    'Accuracy': [dt_accuracy, lr_accuracy],
    'Precision': [dt_precision, lr_precision],
    'Recall': [dt_recall, lr_recall],
    'F1 Score': [dt_f1, lr_f1]
})

print("\n=== Model Comparison Table ===")
print(results)

# ==============================
#  Feature Importance - Decision Tree
# ==============================
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
plt.title("Top 10 Important Features (Decision Tree)")
plt.tight_layout()
plt.show()
