import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns

X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = [0]*50 + [1]*50
# y = [0]*50 + [1]*50

# Visualization: Show distance from correct boundary (zero)
m = LogisticRegression(fit_intercept=False)
m.fit(X, y)
print(m.coef_)
print(m.intercept_)
y_proba = m.predict_proba(X)
plt.plot(X.reshape(1, -1), y_proba)
plt.show()