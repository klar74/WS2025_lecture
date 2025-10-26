# Definition eines Dictionaries für einen Studenten
# Ein Dictionary speichert Schlüssel-Wert-Paare.
# Geschweifte Klammern {} werden verwendet.
student = {
    "name": "Max Mustermann",
    "alter": 25,
    "studiengang": "Data Science",
    "note": 1.7
}

print(f"Name: {student['name']}")
print(f"Note: {student['note']}")

student["matrikelnummer"] = 123456
print(f"Matrikelnummer: {student['matrikelnummer']}")

for key in student:
    print(f"{key}: {student[key]}")