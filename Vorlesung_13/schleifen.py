# Countdown mit Schleife
print("Countdown:")
for i in range(5, 0, -1):
    print(f"{i}...")
print("Start!")

# Über eine Liste iterieren
fruechte = ["Apfel", "Banane", "Orange", "Traube"]
print("Meine Lieblings-Früchte:")
for frucht in fruechte:
    print(f"- {frucht}")

# Zahlen-Beispiel für Data Science
temperaturen = [18.5, 22.1, 19.8, 25.3, 21.7]
print("Temperaturen:")
for i, temp in enumerate(temperaturen):
    print(f"Tag {i+1}: {temp}°C")