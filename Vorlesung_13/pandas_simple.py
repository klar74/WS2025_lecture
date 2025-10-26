# Import von pandas und einfache Funktionen
import pandas as pd

# DataFrame erstellen
daten = {
    "Name": ["Max", "Anna", "Tom"],
    "Alter": [25, 22, 28],
    "Note": [1.7, 2.3, 1.0]
}
df = pd.DataFrame(daten)

# DataFrame anzeigen
print(df)

# Statistische Zusammenfassung
print(df.describe())

# Durchschnitt berechnen
print("Durchschnittsalter:", df["Alter"].mean())