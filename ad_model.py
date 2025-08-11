import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# ----- Expanded sample dataset -----
data = {
    'Age': [25, 34, 22, 45, 30, 40, 28, 50, 23, 35, 33, 29, 41, 38, 27],
    'Location': ['Urban', 'Suburban', 'Urban', 'Rural', 'Urban', 'Suburban', 'Rural', 'Urban', 'Urban', 'Rural', 'Urban', 'Suburban', 'Urban', 'Rural', 'Suburban'],
    'Interests': ['Sports', 'Fashion', 'Music', 'Tech', 'Fashion', 'Sports', 'Tech', 'Music', 'Fashion', 'Sports', 'Tech', 'Fashion', 'Music', 'Sports', 'Tech'],
    'ShoppingFrequency': [5, 3, 7, 2, 4, 6, 1, 3, 5, 2, 4, 5, 3, 6, 2],  # times/month
    'IncomeLevel': ['Low', 'High', 'Medium', 'Medium', 'High', 'Low', 'Low', 'High', 'Medium', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low'],
    'DeviceType': ['Mobile', 'Desktop', 'Mobile', 'Tablet', 'Desktop', 'Mobile', 'Tablet', 'Desktop', 'Mobile', 'Tablet', 'Mobile', 'Desktop', 'Tablet', 'Mobile', 'Desktop'],
    # Target ad category clicked (multi-class)
    'AdCategory': ['Sportswear', 'Clothing', 'Music Gear', 'Gadgets', 'Clothing', 'Sportswear', 'Gadgets', 'Music Gear', 'Clothing', 'Sportswear', 'Gadgets', 'Clothing', 'Music Gear', 'Sportswear', 'Gadgets']
}

df = pd.DataFrame(data)

# ----- Features and target -----
X = df.drop('AdCategory', axis=1)
y = df['AdCategory']

# ----- Preprocessing -----
categorical_features = ['Location', 'Interests', 'IncomeLevel', 'DeviceType']
numeric_features = ['Age', 'ShoppingFrequency']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# ----- Model pipeline -----
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ----- Train-test split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- Train model -----
model.fit(X_train, y_train)

# ----- Predictions and evaluation -----
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ----- Visualization 1: Demographic distributions -----
plt.figure(figsize=(12, 6))

plt.subplot(2,3,1)
sns.histplot(df['Age'], bins=8, kde=True)
plt.title("Age Distribution")

plt.subplot(2,3,2)
sns.countplot(x='Location', data=df)
plt.title("Location Distribution")

plt.subplot(2,3,3)
sns.countplot(x='Interests', data=df)
plt.title("Interests Distribution")

plt.subplot(2,3,4)
sns.countplot(x='IncomeLevel', data=df)
plt.title("Income Level Distribution")

plt.subplot(2,3,5)
sns.countplot(x='DeviceType', data=df)
plt.title("Device Type Distribution")

plt.tight_layout()
plt.show()

# ----- Visualization 2: Feature importance -----
cat_features = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_features)
all_features = np.concatenate([cat_features, numeric_features])

importances = model.named_steps['classifier'].feature_importances_

feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ----- Visualization 3: Confusion matrix -----
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ----- Visualization 4: Distribution of predicted Ad categories -----
pred_counts = pd.Series(y_pred).value_counts()

plt.figure(figsize=(6,4))
sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="viridis")
plt.title("Predicted Ad Category Distribution")
plt.ylabel("Count")
plt.xlabel("Ad Category")
plt.show()

from sklearn.metrics import auc, roc_curve
# ----- Visualization 5: ROC Curve (One-vs-Rest) -----
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=model.classes_)
y_score = model.predict_proba(X_test)

plt.figure(figsize=(8,6))
for i, class_label in enumerate(model.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc='lower right')
plt.show()

# ----- Example prediction -----
new_user = pd.DataFrame({
    'Age': [29],
    'Location': ['Urban'],
    'Interests': ['Fashion'],
    'ShoppingFrequency': [4],
    'IncomeLevel': ['Medium'],
    'DeviceType': ['Mobile']
})
predicted_ad = model.predict(new_user)[0]
print(f"Recommended Ad Category for new user: {predicted_ad}")
