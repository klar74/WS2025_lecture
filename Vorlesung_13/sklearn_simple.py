# Ganz einfache Nutzung von sklearn
from sklearn.linear_model import LinearRegression
import numpy as np

# Beispiel-Daten
# WICHTIG: Doppelte eckige Klammern bei X!
# Äußere []: Liste der Datenpunkte (Samples)
# Innere []: Features pro Datenpunkt (hier nur 1 Feature)
# Sklearn braucht 2D-Arrays: Shape (n_samples, n_features) = (5, 1)
X = np.array([[1], [2], [3], [4], [5]])  # Feature - Shape: (5, 1)
y = np.array([2, 4, 6, 8, 10])           # Zielvariable - Shape: (5,)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Modell erstellen und trainieren
model = LinearRegression()
model.fit(X, y)

# Vorhersage
# Wieder doppelte Klammern: [[6]] bedeutet "1 Sample mit 1 Feature"
# Ohne doppelte Klammern würde sklearn einen Fehler werfen (probiere das aus!).
vorhersage = model.predict([[6]])  # Input muss 2D sein: (1, 1)
print(f"Vorhersage für x=6: {vorhersage[0]:.1f}")