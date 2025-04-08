import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.metrics import *
from collections import Counter

df = pd.read_csv(r"C:\Users\LAHARI\OneDrive\Desktop\MajorProject\Ransomware.csv", sep='|')

df.head()

print(df.shape[0])

df.describe()
df.isnull().sum()
df.legitimate.value_counts()
df.info()

plt.pie(df.legitimate.value_counts().values.tolist(), labels=['Safe', 'Ransomware'], autopct='%.2f%%')
plt.legend()
plt.title(f"Distribution of Labelled Data, total - {len(df)}")
plt.show()

# Create correlation matrix
corr_matrix = df.drop(['Name', 'md5', 'legitimate'], axis=1).corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(to_drop)

df.drop(to_drop, axis=1, inplace=True)
sns.heatmap(df.drop(['Name', 'md5', 'legitimate'], axis=1).corr())

def iv_woe(data, target, bins=10, show_woe=False):
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)
        if show_woe:
            print(d)
    return newDF, woeDF

df.legitimate.dtypes
iv, woe = iv_woe(df.drop(['Name', 'md5'], axis=1), 'legitimate')
iv.sort_values(by='IV', ascending=False)

thresh = 1
res = len(iv) - len(iv[iv['IV'] > thresh])
print(res)

features = iv.sort_values(by='IV', ascending=False)['Variable'][:res].values.tolist()
print(features, '\n')
print('Total number of features-\n', len(features))

X = df[features]
y = df['legitimate']
randomseed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=randomseed)

print(X_test.shape[0] + X_train.shape[0])
print('Training labels shape:', y_train.shape)
print('Test labels shape:', y_test.shape)
print('Training features shape:', X_train.shape)
print('Test features shape:', X_test.shape)

rf = RandomForestClassifier(random_state=randomseed)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
pred_proba = rf.predict_proba(X_test)
pred_proba = np.array([prob[1] for prob in pred_proba])

cm = confusion_matrix(y_test, pred)
cm

classes = ['Safe', 'Malware']
cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
cmd.plot()
plt.show()

TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
fpr = FP / (FP + TN)
auc = roc_auc_score(y_test, pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"AUC Score: {auc:.4f}")
