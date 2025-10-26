# Definition einer Begrüßungsfunktion
def begruessung(name):
    print(f"Hallo, {name}! Willkommen bei Data Science.")

# Definition einer Funktion zur Berechnung des Quadrats einer Zahl
def quadrat(x):
    return x * x

# Beispielaufrufe der Funktionen
begruessung("Max")
zahl = 7
print(f"Das Quadrat von {zahl} ist {quadrat(zahl)}.")