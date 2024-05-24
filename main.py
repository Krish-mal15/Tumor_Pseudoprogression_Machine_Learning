import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC

warnings.filterwarnings("ignore", category=UserWarning)

data_path = 'Pseudo-Immune-Progression-Data.xlsx'

data = pd.read_excel(data_path)
data.drop('GeneID', axis=1, inplace=True)

progression_status = data.iloc[-1].values
data = data.iloc[:-1]

data = data.T

labels = []

# Progression = 0  Pseudo-progression = 1

for i in progression_status:
    if i == 'Progress':
        labels.append(0)
    elif i == 'Pseudo':
        labels.append(1)

print("Shape of data:", data.shape)
print("Length of labels:", len(labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = LR()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Logistic Regression Accuracy:", accuracy)

# ROC Evaluation

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Model Coefficients Logistic Regression

coefficients = model.coef_[0]
gene_names = data.columns

coefficients_df = pd.DataFrame({'Gene': gene_names, 'Coefficient': coefficients})

coefficients_df['Abs_Coefficient'] = coefficients_df['Coefficient'].abs()
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)

print("Top influential genes:")
print(coefficients_df.head(10))

# Random Forest Classifier (Lower ROC, but Higher Accuracy)

model = RFC()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Random Forest Accuracy:", accuracy)

