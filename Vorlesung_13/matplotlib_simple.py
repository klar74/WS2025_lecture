# Erstellen von Plots mit matplotlib
import matplotlib.pyplot as plt

# Beispiel-Daten
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Liniendiagramm
plt.plot(x, y, marker='o')
plt.title("Einfache Lineare Beziehung")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()