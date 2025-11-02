# ROC-Kurve Berechnung - Übungsaufgabe

## Aufgabe: ROC-Kurve manuell erstellen

**Gegeben sind 10 Qualitätsprüfungen mit folgenden Ergebnissen:**

### Daten

| Sample | Wahres Label | Defektwahrscheinlichkeit |
|--------|--------------|-------------------------|
| 1      | 0 (OK)       | 0.2                     |
| 2      | 0 (OK)       | 0.3                     |
| 3      | 0 (OK)       | 0.1                     |
| 4      | 0 (OK)       | 0.6                     |
| 5      | 0 (OK)       | 0.7                     |
| 6      | 1 (NOK)      | 0.8                     |
| 7      | 1 (NOK)      | 0.9                     |
| 8      | 1 (NOK)      | 0.4                     |
| 9      | 1 (NOK)      | 0.85                    |
| 10     | 1 (NOK)      | 0.75                    |

**Erklärung:**
- **Label 0**: Teil ist in Ordnung (OK)
- **Label 1**: Teil ist defekt (NOK)
- **Defektwahrscheinlichkeit**: Vom Modell vorhergesagte Wahrscheinlichkeit für einen Defekt

---

## Aufgaben

### 1. Konfusionsmatrix bei Schwelle 0.5
Bestimmen Sie für eine Klassifikationsschwelle von 0.5:
- **True Positives (TP)**: Korrekt als defekt erkannt
- **True Negatives (TN)**: Korrekt als OK erkannt  
- **False Positives (FP)**: Fälschlicherweise als defekt erkannt
- **False Negatives (FN)**: Übersehene Defekte

### 2. Metriken berechnen
Berechnen Sie:
- **TPR (True Positive Rate)** = TP / (TP + FN)
- **FPR (False Positive Rate)** = FP / (FP + TN)

### 3. ROC-Kurve erstellen
- Wählen Sie mindestens 5 verschiedene Schwellenwerte
- Berechnen Sie für jeden Schwellenwert TPR und FPR
- Zeichnen Sie die ROC-Kurve (FPR auf x-Achse, TPR auf y-Achse)

### 4. AUC schätzen
Schätzen Sie die **AUC (Area Under Curve)** basierend auf Ihrer ROC-Kurve.

---

## Lösungsansatz

### Schwellenwerte vorschlagen:
- 0.0 (alle als defekt klassifiziert)
- 0.3, 0.5, 0.7, 0.9
- 1.0 (alle als OK klassifiziert)

### Beispiel-Rechnung für Schwelle 0.5:

**Klassifikation:** Defekt wenn Wahrscheinlichkeit ≥ 0.5

**Vorhersagen:**
- Sample 1-3: OK (richtig) → TN = 3
- Sample 4-5: NOK (falsch) → FP = 2  
- Sample 6-7, 9-10: NOK (richtig) → TP = 4
- Sample 8: OK (falsch) → FN = 1

**Metriken:**
- TPR = 4/(4+1) = 0.8
- FPR = 2/(2+3) = 0.4

---

## Erwartetes Ergebnis

Die ROC-Kurve sollte **oberhalb der Diagonalen** verlaufen, da das Modell besser als zufällige Klassifikation ist.

**AUC-Interpretation:**
- AUC = 0.5: Zufällige Klassifikation
- AUC = 1.0: Perfekte Klassifikation
- AUC > 0.7: Gute Klassifikation

---

## Diskussionsfragen

1. **Was passiert bei sehr niedrigen Schwellenwerten (z.B. 0.1)?**
2. **Was passiert bei sehr hohen Schwellenwerten (z.B. 0.9)?**
3. **Wie würde sich die ROC-Kurve ändern, wenn alle Defektwahrscheinlichkeiten um 0.1 erhöht würden?**
4. **Wann wäre eine hohe FPR akzeptabel, wann problematisch?**