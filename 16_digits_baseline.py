import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data      # the same images, but each one flattened into 64 numbers in a row
y = digits.target
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("Accuracy:", f"{accuracy_score(y_test, pred):.2%}")

ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
plt.title("Random Forest on digit pixels")
plt.show()