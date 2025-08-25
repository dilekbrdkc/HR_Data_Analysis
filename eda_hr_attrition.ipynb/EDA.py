# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the data
df = pd.read_csv(r"C:\Users\dilek\Desktop\HR_Data_Analysis\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display first 5 rows
print(df.head())

# Data Shape
print("\nData Shape:", df.shape)

# Data types and missing values
df.info()

# Target variable (Attrition) distribution
print("\nAttrition distribution (%):")
print(df['Attrition'].value_counts(normalize=True) * 100)

# Turnover rate by department (%)
turnover_by_dept = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack().fillna(0)
turnover_by_dept['Attrition Rate (%)'] = turnover_by_dept['Yes'] * 100
print("\nTurnover Rate by Department (%):")
print(turnover_by_dept)

# Visualize turnover rate by department
sns.barplot(
    data=turnover_by_dept.reset_index(),
    x='Department',
    y='Attrition Rate (%)',
    palette='Reds'
)
plt.title("Turnover Rate by Department(%)")
plt.ylabel("Turnover (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Select numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# 🔁 Regular histograms (with subplots)
num_features = len(numerical_cols)
part_size = num_features // 4 + (1 if num_features % 4 != 0 else 0)  # Max number of features in each part

for part in range(4):
    start_idx = part * part_size
    end_idx = min(start_idx + part_size, num_features)
    subset = numerical_cols[start_idx:end_idx]
    
    n_plots = len(subset)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3.5))
    axes = axes.flatten()
    
    for i, col in enumerate(subset):
        sns.histplot(df[col], bins=20, kde=False, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'{col}', fontsize=10, pad=10, weight='bold')
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel("Count", fontsize=9)
        
        # Reduce tick label sizes
        axes[i].tick_params(axis='x', labelsize=8, rotation=0)
        axes[i].tick_params(axis='y', labelsize=8)
    
    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35, left=0.05, right=0.98, bottom=0.1)
    plt.show()

# Select categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Visualize categorical variable distributions
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='Set2')
    plt.title(f"{col} Distribution", fontsize=14, weight='bold', pad=15)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Compare salary, satisfaction, age for employees who left vs stayed
selected_features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'JobSatisfaction', 'YearsAtCompany']

for feature in selected_features:
    plt.figure(figsize=(8, 5))  # Increase size a bit
    sns.boxplot(data=df, x='Attrition', y=feature, palette='coolwarm')
    plt.title(f"According to Attrition - {feature}", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Attrition", fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.tight_layout()
    plt.show()

# Correlation matrix
corr_matrix = df[numerical_cols].corr()

# Visualize correlation matrix with heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, mask=mask)
plt.title("Correlation Matrix Between Numerical Variables")
plt.tight_layout()
plt.show()

# Turnover rate by some important categorical variables
cat_cols_for_attrition = ['BusinessTravel', 'JobRole', 'MaritalStatus', 'EducationField', 'Gender']

for col in cat_cols_for_attrition:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='Attrition', palette='Set1')
    plt.title(f"According to Attrition - {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Check for missing values
missing = df.isnull().sum()
print("\nMissing Data:")
print(missing[missing > 0])

# Fill missing data with the mean
df.fillna(df.mean(), inplace=True)

# Convert categorical variables to numeric (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# Split the data into features and target
X = df.drop('Attrition_Yes', axis=1)
y = df['Attrition_Yes']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Model evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
