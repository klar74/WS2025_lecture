# Definition einer Liste von Noten
noten = [1.7, 2.3, 1.0, 3.7, 2.0]

print("Alle Noten:")
for note in noten:
    print(note)

durchschnitt = sum(noten) / len(noten)
print(f"Durchschnittsnote: {durchschnitt:.2f}")

# Ein Element an die Liste anhängen
noten.append(2.7)
print("Noten nach Hinzufügen:", noten)