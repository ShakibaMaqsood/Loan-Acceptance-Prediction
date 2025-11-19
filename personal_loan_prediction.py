import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("bank.csv")
print("\nDataset Shape:", df.shape)
print(df.head())

# Rename target column to match task
df.rename(columns={"deposit": "loan_acceptance"}, inplace=True)
df["loan_acceptance"] = df["loan_acceptance"].map({"yes": 1, "no": 0})

# ----------------------------
# 2. Basic Data Exploration
# ----------------------------
print("\nDataset Description:\n", df.describe())
print("\nJob Distribution:\n", df['job'].value_counts())
print("\nMarital Status Distribution:\n", df['marital'].value_counts())
print("\nEducation Distribution:\n", df['education'].value_counts())

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=20)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.close()

# Marital Status Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['marital'])
plt.title("Marital Status Distribution")
plt.savefig("marital_distribution.png")
plt.close()

# ----------------------------
# 3. Insight Analysis (EDA)
# ----------------------------
# Create age groups for analysis only
df['age_group'] = pd.cut(df['age'], bins=[18,30,40,50,60,100],
                         labels=['18-30','31-40','41-50','51-60','60+'])

print("\nLoan Acceptance by Age Group:")
print(df.groupby('age_group')['loan_acceptance'].mean())

print("\nLoan Acceptance by Job:")
print(df.groupby('job')['loan_acceptance'].mean())

print("\nLoan Acceptance by Marital Status:")
print(df.groupby('marital')['loan_acceptance'].mean())

# ----------------------------
# 4. Encode Categorical Features
# ----------------------------
encode_cols = ['job', 'marital', 'education', 'default', 'housing',
               'loan', 'contact', 'month', 'poutcome']

le = LabelEncoder()
for col in encode_cols:
    df[col] = le.fit_transform(df[col])

# ----------------------------
# 5. Prepare Data for Modeling
# ----------------------------
# Drop age_group for ML model
X = df.drop(['loan_acceptance', 'age_group'], axis=1)
y = df['loan_acceptance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ----------------------------
# 6. Train Models
# ----------------------------
# Logistic Regression
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

# ----------------------------
# 7. Evaluation
# ----------------------------
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

print("\nDecision Tree Accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, log_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Logistic Regression")
plt.savefig("confusion_matrix_logistic.png")
plt.close()

# ----------------------------
# 8. Feature Importance (Decision Tree)
# ----------------------------
importances = tree_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(9,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Decision Tree)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nFeature importance saved as: feature_importance.png")
