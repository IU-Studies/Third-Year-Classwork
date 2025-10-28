import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

data = pd.read_csv(r'C:\Users\IU\Downloads\SocialNetworkAds.csv')
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

sns.countplot(x='Purchased', data=data)
plt.title("Purchase Distribution")
plt.show()

sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=data, palette='coolwarm')
plt.title("Age vs Estimated Salary (Colored by Purchase)")
plt.show()

X = data[['Age', 'EstimatedSalary']].values
y = data['Purchased'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

cm_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)

print("=== Logistic Regression Results ===")
print("Confusion Matrix:\n", cm_log)
print(f"Accuracy: {acc_log:.2f}")
print(f"Precision: {prec_log:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_log))

ConfusionMatrixDisplay(cm_log, display_labels=['Not Purchased', 'Purchased']).plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)

print("\n=== k-NN Results ===")
print("Confusion Matrix:\n", cm_knn)
print(f"Accuracy: {acc_knn:.2f}")
print(f"Precision: {prec_knn:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

ConfusionMatrixDisplay(cm_knn, display_labels=['Not Purchased', 'Purchased']).plot()
plt.title("Confusion Matrix - k-NN")
plt.show()

def plot_decision_boundary(X_set, y_set, model, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('lightcoral', 'lightgreen'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title(title)
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.legend()
    plt.show()

plot_decision_boundary(X_train, y_train, log_reg, "Decision Boundary - Logistic Regression (Training set)")
plot_decision_boundary(X_test, y_test, log_reg, "Decision Boundary - Logistic Regression (Test set)")
plot_decision_boundary(X_train, y_train, knn, "Decision Boundary - k-NN (Training set)")
plot_decision_boundary(X_test, y_test, knn, "Decision Boundary - k-NN (Test set)")

print("\n=== Model Performance Comparison ===")
print(f"Logistic Regression -> Accuracy: {acc_log:.2f}, Precision: {prec_log:.2f}")
print(f"k-NN               -> Accuracy: {acc_knn:.2f}, Precision: {prec_knn:.2f}")

print("\nConclusion:")
print("Both Logistic Regression and k-NN models were trained and evaluated.")
print("Performance varies slightly depending on data distribution and feature scaling.")
print("Logistic Regression performs well on linearly separable data, while k-NN adapts to complex decision boundaries.")
