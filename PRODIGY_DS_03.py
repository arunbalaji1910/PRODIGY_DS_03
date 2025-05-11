import pandas as pd

df = pd.read_csv("bank-additional-full.csv", sep=';')  
df.head()

df_encoded = pd.get_dummies(df.drop(columns=['y']), drop_first=True)

df_encoded['target'] = df['y'].map({'yes': 1, 'no': 0})
from sklearn.model_selection import train_test_split

X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4, random_state=42) 

clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

print("\nBank Marketing Dataset - Decision Tree Classifier Completed by Arun Balaji! ")
