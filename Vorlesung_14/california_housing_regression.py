"""
California Housing - Lineare Regression (Python Script Version)

Dieses Skript führt eine komplette Machine Learning Pipeline mit dem California Housing Datensatz durch.
Basierend auf dem Jupyter Notebook california_housing_regression.ipynb

Autor: Auto-generiert aus Notebook
"""

# Schritt 1: Bibliotheken importieren
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Matplotlib für Skript-Modus konfigurieren (keine GUI)
plt.ioff()  # Interactive mode off
plt.style.use('default')

# Plot-Verzeichnis erstellen
plot_dir = "california_housing_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Verzeichnis '{plot_dir}' erstellt.")
else:
    print(f"Verzeichnis '{plot_dir}' existiert bereits.")

print("Alle Bibliotheken erfolgreich importiert!")

# Schritt 2: Daten laden
print("\n" + "="*60)
print("SCHRITT 2: DATEN LADEN")
print("="*60)
print("Lade California Housing Datensatz...")
california = fetch_california_housing()

# Als pandas DataFrame für bessere Handhabung
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='HouseValue')

print(f"Datensatz geladen: {X.shape[0]} Regionen, {X.shape[1]} Features")
print(f"Features: {list(california.feature_names)}")

# Schritt 3: Daten verstehen
print("\n" + "="*60)
print("SCHRITT 3: DATEN VERSTEHEN")
print("="*60)
print("Erste 5 Datensätze:")
print(X.head())

print(f"\nHauspreise - Erste 5 Werte:")
print(y.head().values)

print(f"\nPreis-Statistiken:")
print(f"   Durchschnitt: ${y.mean():.2f} (x100k) = ${y.mean()*100:.0f}k")
print(f"   Minimum: ${y.min():.2f} (x100k) = ${y.min()*100:.0f}k")  
print(f"   Maximum: ${y.max():.2f} (x100k) = ${y.max()*100:.0f}k")

# Schritt 3b: Dataset-Eigenart verstehen - Preiskappung bei $500k
print("\n" + "-"*40)
print("DATASET-EIGENART: PREISKAPPUNG")
print("-"*40)
print("WICHTIGER HINWEIS - Dataset-Eigenart:")
print(f"   Maximum im Dataset: ${y.max():.5f} (x100k) = ${y.max()*100:.0f}k")

# Korrekte Analyse: 5.0 UND 5.00001 sind beide gekappte Werte
capped_at_500k = np.sum(y >= 5.0)

print(f"   Werte bei genau $500.000k: {capped_at_500k}")
print(f"   Werte >= $490k und < $500k: {np.sum((y >= 4.9) & (y < 5.0))} von {len(y)} ({np.sum((y >= 4.9) & (y < 5.0))/len(y)*100:.1f}%)")

# Visualisierung des Kappungseffekts
plt.figure(figsize=(10, 6))
plt.hist(y, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(5.00001, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Kappung bei $500k (992 Häuser)')
plt.xlabel('Hauswerte (x100k $)')
plt.ylabel('Häufigkeit')
plt.title('Verteilung der Hauswerte mit Kappung bei $500.000')
plt.legend()
plt.grid(True, alpha=0.3)
plot_path = os.path.join(plot_dir, 'plot_01_preisverteilung_original.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot gespeichert: {plot_path}")

print(f"\nQUELLEN & HINTERGRUND:")
print(f"   Original-Studie: Pace & Barry (1997), Statistics and Probability Letters")
print(f"   Datenquelle: US Census 1990, California Block Groups")
print(f"   Preprocessing: Häuser >$500k wurden auf $500k 'gekappt'")

# Schritt 4: Train/Test Split
print("\n" + "="*60)
print("SCHRITT 4: TRAIN/TEST SPLIT")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% für Test
    random_state=42     # Für reproduzierbare Ergebnisse
)

print(f"Training: {X_train.shape[0]} Regionen")
print(f"Test: {X_test.shape[0]} Regionen")
print(f"Verhältnis: {X_train.shape[0]/X_test.shape[0]:.1f}:1 (Train:Test)")

# Schritt 4b: Feature-Skalierung (Optional)
print("\n" + "-"*40)
print("SCHRITT 4B: FEATURE-SKALIERUNG (OPTIONAL)")
print("-"*40)
# StandardScaler normalisiert Features auf Mittelwert=0, Standardabweichung=1
# Das kann bei linearer Regression helfen, wenn Features sehr unterschiedliche Skalen haben

USE_SCALING = True  # Setze auf False, um Skalierung zu deaktivieren

if USE_SCALING:
    print("Skaliere Features mit StandardScaler...")
    scaler = StandardScaler()
    
    # Scaler nur auf Trainingsdaten fitten (wichtig für ehrliche Evaluierung!)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Zurück zu DataFrame für bessere Lesbarkeit
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print("Feature-Skalierung abgeschlossen!")
    print(f"   Beispiel - Durchschnitt nach Skalierung: {X_train.mean().round(3).values}")
    print(f"   Beispiel - Std nach Skalierung: {X_train.std().round(3).values}")
else:
    print("Feature-Skalierung übersprungen (USE_SCALING = False)")
    print("   Hinweis: Lineare Regression funktioniert oft auch ohne Skalierung gut")
    print("   Bei großen Unterschieden in Feature-Skalen kann Skalierung helfen.")
    print("   Bei Interpretation der Koeffizienten ist Skalierung wichtig.")

# Schritt 5: Modell trainieren
print("\n" + "="*60)
print("SCHRITT 5: MODELL TRAINIEREN")
print("="*60)
model = LinearRegression()

print("Trainiere das Modell...")
model.fit(X_train, y_train)

print(f"Modell trainiert!")
print(f"Basis-Hauspreis: ${model.intercept_:.2f} (x100k)")

# Die wichtigsten Features zeigen
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Koeffizient': model.coef_
}).sort_values('Koeffizient', key=abs, ascending=False)

print(f"\nTop 5 wichtigste Features:")
for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Koeffizient']:.3f}")

# Schritt 6: Vorhersagen und Bewertung
print("\n" + "="*60)
print("SCHRITT 6: VORHERSAGEN UND BEWERTUNG")
print("="*60)
y_pred = model.predict(X_test)

# Performance-Metriken berechnen
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Modell-Performance:")
print(f"   MSE:  {mse:.2f} x(100k$)^2 Durchschnittsfehler")
print(f"   RMSE: {rmse:.2f} (x100k$) = ${rmse*100:.0f}k Durchschnittsfehler")  
print(f"   R-Quadrat-Score: {r2:.3f} = {r2*100:.1f}% der Variation erklärt")

# Beispiel-Vorhersagen zeigen
print(f"\nBeispiel-Vorhersagen:")
for i in range(5):
    actual = y_test.iloc[i] * 100  # In Einheiten 100k umrechnen
    predicted = y_pred[i] * 100
    error = abs(actual - predicted)
    print(f"   Tatsächlich: ${actual:.0f}k, Vorhersage: ${predicted:.0f}k, Fehler: ${error:.0f}k")

# Schritt 7: Ergebnisse visualisieren
print("\n" + "="*60)
print("SCHRITT 7: ERGEBNISSE VISUALISIEREN")
print("="*60)
plt.figure(figsize=(12, 4))

# Grafik 1: Vorhersage vs. Realität
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideale Linie (y = ŷ)')
plt.xlabel('Tatsächliche Preise (x100k $)')
plt.ylabel('Vorhergesagte Preise (x100k $)')
plt.title('Vorhersage vs. Realität')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafik 2: Fehler-Verteilung
plt.subplot(1, 2, 2)
errors = (y_test - y_pred) * 100  # In 100k Dollar
plt.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
plt.xlabel('Vorhersagefehler (100k $)')
plt.ylabel('Häufigkeit')
plt.title('Verteilung der Vorhersagefehler')
plt.grid(True, alpha=0.3)
plt.axvline(0, color='red', linestyle='--', alpha=0.8, label='Kein Fehler')
plt.legend()

plt.tight_layout()
plot_path = os.path.join(plot_dir, 'plot_02_original_modell_performance.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot gespeichert: {plot_path}")

print(f"Durchschnittlicher Absolut-Fehler: ${np.mean(np.abs(errors)):.0f}k")

# Schritt 8: Residuenplot - Analyse der Vorhersagefehler
print("\n" + "="*60)
print("SCHRITT 8: RESIDUENANALYSE")
print("="*60)
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, s=30, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Ideale Linie (Residuen = 0)')
plt.xlabel('Vorhergesagte Preise (x100k $)')
plt.ylabel('Residuen (x100k $)')
plt.title('Residuen vs. Vorhersagewerte')
plt.legend()
plt.grid(True, alpha=0.3)
plot_path = os.path.join(plot_dir, 'plot_03_residuen_original.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot gespeichert: {plot_path}")

print("RESIDUENANALYSE:")
print(f"   Mittelwert der Residuen: {np.mean(residuals):.6f} (sollte ca. 0 sein)")
print(f"   Standardabweichung: {np.std(residuals):.4f}")
print(f"   Min/Max Residuen: {np.min(residuals):.2f} / {np.max(residuals):.2f}")
print("\nINTERPRETATION:")
print("   Gute Modelle: Residuen zufällig um 0 verteilt")
print("   Probleme: Systematische Muster in den Residuen")

# ==========================================
# VERGLEICHSANALYSE: BEREINIGTER DATENSATZ
# ==========================================

print("\n" + "="*60)
print("VERGLEICHSANALYSE: BEREINIGTER DATENSATZ")
print("="*60)

# Schritt 9: Bereinigten Datensatz erstellen
print("\n" + "-"*40)
print("SCHRITT 9: BEREINIGTEN DATENSATZ ERSTELLEN")
print("-"*40)
print("Erstelle bereinigten Datensatz...")

# Originaldaten sichern
X_original = X.copy()
y_original = y.copy()

# Filter: Entferne alle Werte >= 5.0 (entspricht $500k)
clean_mask = y < 5.0
X_clean = X[clean_mask].copy()
y_clean = y[clean_mask].copy()

print(f"DATENSATZ-VERGLEICH:")
print(f"   Original: {len(y_original)} Regionen")
print(f"   Bereinigt: {len(y_clean)} Regionen")
print(f"   Entfernt: {len(y_original) - len(y_clean)} gekappte Häuser ({(len(y_original) - len(y_clean))/len(y_original)*100:.1f}%)")

print(f"\nPREIS-STATISTIKEN BEREINIGT:")
print(f"   Durchschnitt: ${y_clean.mean():.2f} (x100k) = ${y_clean.mean()*100:.0f}k")
print(f"   Minimum: ${y_clean.min():.2f} (x100k) = ${y_clean.min()*100:.0f}k")  
print(f"   Maximum: ${y_clean.max():.2f} (x100k) = ${y_clean.max()*100:.0f}k")

# Schritt 9b: Feature-Skalierung für bereinigten Datensatz (Optional)
print("\n" + "-"*40)
print("SCHRITT 9B: FEATURE-SKALIERUNG FÜR BEREINIGTEN DATENSATZ")
print("-"*40)
# StandardScaler für den bereinigten Datensatz - unabhängig vom Original-Datensatz

USE_SCALING_CLEAN = True  # Setze auf False, um Skalierung für bereinigten Datensatz zu deaktivieren

if USE_SCALING_CLEAN:
    print("Skaliere Features des bereinigten Datensatzes mit StandardScaler...")
    scaler_clean = StandardScaler()
    
    # Features vor dem Train/Test Split skalieren
    X_clean_scaled = scaler_clean.fit_transform(X_clean)
    
    # Zurück zu DataFrame für bessere Lesbarkeit
    X_clean = pd.DataFrame(X_clean_scaled, columns=X_clean.columns, index=X_clean.index)
    
    print("Feature-Skalierung (bereinigt) abgeschlossen!")
    print(f"   Durchschnitt nach Skalierung: {X_clean.mean().round(3).values}")
    print(f"   Std nach Skalierung: {X_clean.std().round(3).values}")
    
    print("\nWICHTIG: Bereinigter Datensatz wird separat skaliert!")
    print("   Original-Datensatz und bereinigter Datensatz haben unterschiedliche Skalierung")
else:
    print("Feature-Skalierung für bereinigten Datensatz übersprungen (USE_SCALING_CLEAN = False)")
    print("   Hinweis: Für Vergleichbarkeit oft besser, beide Datensätze gleich zu behandeln")

# Schritt 10: Train/Test Split für bereinigten Datensatz
print("\n" + "-"*40)
print("SCHRITT 10: TRAIN/TEST SPLIT FÜR BEREINIGTEN DATENSATZ")
print("-"*40)
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_clean, 
    test_size=0.2,      # 20% für Test
    random_state=42     # Für reproduzierbare Ergebnisse
)

print(f"Training (bereinigt): {X_train_clean.shape[0]} Regionen")
print(f"Test (bereinigt): {X_test_clean.shape[0]} Regionen")
print(f"Verhältnis: {X_train_clean.shape[0]/X_test_clean.shape[0]:.1f}:1 (Train:Test)")

# Visualisierung: Vergleich der Preisverteilungen
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(y_original, bins=50, alpha=0.7, edgecolor='black', color='red', label='Original')
plt.axvline(5.0, color='red', linestyle='--', linewidth=2, label='Kappung bei $500k')
plt.xlabel('Hauswerte (x100k $)')
plt.ylabel('Häufigkeit')
plt.title('Original Datensatz (mit Kappung)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_clean, bins=50, alpha=0.7, edgecolor='black', color='green', label='Bereinigt')
plt.xlabel('Hauswerte (x100k $)')
plt.ylabel('Häufigkeit')
plt.title('Bereinigter Datensatz (ohne Kappung)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(plot_dir, 'plot_04_preisverteilung_vergleich.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot gespeichert: {plot_path}")

# Schritt 11: Modell auf bereinigten Daten trainieren
print("\n" + "-"*40)
print("SCHRITT 11: MODELL AUF BEREINIGTEN DATEN TRAINIEREN")
print("-"*40)
model_clean = LinearRegression()

print("Trainiere das Modell auf bereinigten Daten...")
model_clean.fit(X_train_clean, y_train_clean)

print(f"Bereinigtes Modell trainiert!")
print(f"Basis-Hauspreis (bereinigt): ${model_clean.intercept_:.2f} (x100k)")

# Die wichtigsten Features zeigen
feature_importance_clean = pd.DataFrame({
    'Feature': X_clean.columns,
    'Koeffizient_Original': model.coef_,
    'Koeffizient_Bereinigt': model_clean.coef_
}).sort_values('Koeffizient_Bereinigt', key=abs, ascending=False)

print(f"\nVergleich der Feature-Wichtigkeit:")
print(f"{'Feature':<12} {'Original':<10} {'Bereinigt':<10} {'Differenz':<10}")
print("-" * 50)
for _, row in feature_importance_clean.head().iterrows():
    diff = row['Koeffizient_Bereinigt'] - row['Koeffizient_Original']
    print(f"{row['Feature']:<12} {row['Koeffizient_Original']:<10.3f} {row['Koeffizient_Bereinigt']:<10.3f} {diff:<10.3f}")

# Schritt 12: Vorhersagen und Bewertung (bereinigter Datensatz)
print("\n" + "-"*40)
print("SCHRITT 12: VORHERSAGEN UND BEWERTUNG (BEREINIGTER DATENSATZ)")
print("-"*40)
y_pred_clean = model_clean.predict(X_test_clean)

# Performance-Metriken berechnen
mse_clean = mean_squared_error(y_test_clean, y_pred_clean)
rmse_clean = np.sqrt(mse_clean)
r2_clean = r2_score(y_test_clean, y_pred_clean)

print(f"PERFORMANCE-VERGLEICH:")
print(f"                    Original     Bereinigt    Verbesserung")
print(f"   RMSE:           ${rmse:.2f}        ${rmse_clean:.2f}       {((rmse-rmse_clean)/rmse*100):+.1f}%")
print(f"   R-Quadrat-Score:  {r2:.3f}        {r2_clean:.3f}       {((r2_clean-r2)/r2*100):+.1f}%")

# Beispiel-Vorhersagen zeigen
print(f"\nBeispiel-Vorhersagen (bereinigter Datensatz):")
for i in range(5):
    actual = y_test_clean.iloc[i] * 100  # In normale Tausend Dollar umrechnen
    predicted = y_pred_clean[i] * 100
    error = abs(actual - predicted)
    print(f"   Tatsächlich: ${actual:.0f}k, Vorhersage: ${predicted:.0f}k, Fehler: ${error:.0f}k")

# Schritt 13: Visualisierung bereinigter Datensatz
print("\n" + "-"*40)
print("SCHRITT 13: VISUALISIERUNG BEREINIGTER DATENSATZ")
print("-"*40)
plt.figure(figsize=(15, 4))

# Grafik 1: Vorhersage vs. Realität (Original)
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, s=30, color='red', label='Original')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)
plt.xlabel('Tatsächliche Preise (×100k $)')
plt.ylabel('Vorhergesagte Preise (×100k $)')
plt.title(f'Original Datensatz\nR² = {r2:.3f}')
plt.grid(True, alpha=0.3)

# Grafik 2: Vorhersage vs. Realität (Bereinigt)
plt.subplot(1, 3, 2)
plt.scatter(y_test_clean, y_pred_clean, alpha=0.6, s=30, color='green', label='Bereinigt')
plt.plot([y_test_clean.min(), y_test_clean.max()], [y_test_clean.min(), y_test_clean.max()], 'b--', lw=2)
plt.xlabel('Tatsächliche Preise (×100k $)')
plt.ylabel('Vorhergesagte Preise (×100k $)')
plt.title(f'Bereinigter Datensatz\nR² = {r2_clean:.3f}')
plt.grid(True, alpha=0.3)

# Grafik 3: Fehler-Verteilungen im Vergleich
plt.subplot(1, 3, 3)
errors_original = (y_test - y_pred) * 100
errors_clean = (y_test_clean - y_pred_clean) * 100
plt.hist(errors_original, bins=30, alpha=0.5, color='red', label=f'Original (σ={np.std(errors_original):.0f}k)', density=True)
plt.hist(errors_clean, bins=30, alpha=0.5, color='green', label=f'Bereinigt (σ={np.std(errors_clean):.0f}k)', density=True)
plt.xlabel('Vorhersagefehler (Tausend $)')
plt.ylabel('Dichte')
plt.title('Fehlerverteilung im Vergleich')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(0, color='black', linestyle='--', alpha=0.8)

plt.tight_layout()
plot_path = os.path.join(plot_dir, 'plot_05_modell_vergleich.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot gespeichert: {plot_path}")

print(f"Durchschnittlicher Absolut-Fehler:")
print(f"   Original: ${np.mean(np.abs(errors_original)):.0f}k")
print(f"   Bereinigt: ${np.mean(np.abs(errors_clean)):.0f}k")

# Schritt 14: Residuenplot-Vergleich (Original vs. Bereinigt)
print("\n" + "-"*40)
print("SCHRITT 14: RESIDUENPLOT-VERGLEICH")
print("-"*40)
plt.figure(figsize=(18, 8))

# Residuen berechnen
residuals_original = y_test - y_pred
residuals_clean = y_test_clean - y_pred_clean

# Obere Reihe: Scatter-Plots
plt.subplot(2, 2, 1)
plt.scatter(y_pred, residuals_original, alpha=0.6, s=30, color='red')
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Vorhergesagte Preise (x100k $)')
plt.ylabel('Residuen (x100k $)')
plt.title('Residuen Original Datensatz (Scatter)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.scatter(y_pred_clean, residuals_clean, alpha=0.6, s=30, color='green')
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Vorhergesagte Preise (x100k $)')
plt.ylabel('Residuen (x100k $)')
plt.title('Residuen Bereinigter Datensatz (Scatter)')
plt.grid(True, alpha=0.3)

# Untere Reihe: Density-Plots (2D Hexbin-Plots)
plt.subplot(2, 2, 3)
plt.hexbin(y_pred, residuals_original, gridsize=30, cmap='Reds', alpha=0.8)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Vorhergesagte Preise (x100k $)')
plt.ylabel('Residuen (x100k $)')
plt.title('Residuen Original Datensatz (Density)')
plt.colorbar(label='Anzahl Punkte')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.hexbin(y_pred_clean, residuals_clean, gridsize=30, cmap='Greens', alpha=0.8)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Vorhergesagte Preise (x100k $)')
plt.ylabel('Residuen (x100k $)')
plt.title('Residuen Bereinigter Datensatz (Density)')
plt.colorbar(label='Anzahl Punkte')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(plot_dir, 'plot_06_residuen_vergleich.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot gespeichert: {plot_path}")

print("RESIDUENANALYSE VERGLEICH:")
print(f"                         Original        Bereinigt       Verbesserung")
print(f"   Mittelwert:          {np.mean(residuals_original):8.6f}     {np.mean(residuals_clean):8.6f}     {((np.mean(residuals_clean)-np.mean(residuals_original))/abs(np.mean(residuals_original))*100):+.1f}%")
print(f"   Standardabweichung:   {np.std(residuals_original):7.4f}      {np.std(residuals_clean):7.4f}      {((np.std(residuals_clean)-np.std(residuals_original))/np.std(residuals_original)*100):+.1f}%")
print(f"   Min Residuen:        {np.min(residuals_original):8.2f}     {np.min(residuals_clean):8.2f}")
print(f"   Max Residuen:        {np.max(residuals_clean):8.2f}     {np.max(residuals_clean):8.2f}")

# ==========================================
# ZUSAMMENFASSUNG
# ==========================================

print("\n" + "="*60)
print("ZUSAMMENFASSUNG")
print("="*60)

print("Was wir erreicht haben:")
print("- Komplette ML-Pipeline von Daten bis Vorhersage")

print("\nWas wir gelernt haben:")
print("- MedInc (Einkommen) und Lage (Latitude/Longitude) sind wichtige Faktoren für Hauspreise")
print("- Lineare Regression ist ein guter Startpunkt für Preisvorhersagen")
print("- Train/Test Split ist essentiell für ehrliche Modell-Bewertung")

print("\nWichtige Dataset-Eigenart:")
print("- Preiskappung bei $500k: 992 Häuser (4.8%) wurden ursprünglich bei $500k gekappt")
print("- Quellen: Pace & Barry (1997), 'Sparse Spatial Autoregressions', Statistics and Probability Letters")
print("- Effekt auf Vorhersagen: Künstliche Häufung bei der Kappungsgrenze verfälscht Vorhersagen")

print("\nMögliche nächste Schritte:")
print("- Probiert andere Algorithmen aus (Random Forest, XGBoost)")
print("- Feature Engineering: Neue Features aus bestehenden ableiten")

print("\n" + "="*60)
print("SKRIPT ERFOLGREICH ABGESCHLOSSEN!")
print("="*60)
print(f"Plots wurden gespeichert im Verzeichnis '{plot_dir}':")
print(f"- {os.path.join(plot_dir, 'plot_01_preisverteilung_original.png')}")
print(f"- {os.path.join(plot_dir, 'plot_02_original_modell_performance.png')}")
print(f"- {os.path.join(plot_dir, 'plot_03_residuen_original.png')}")
print(f"- {os.path.join(plot_dir, 'plot_04_preisverteilung_vergleich.png')}")
print(f"- {os.path.join(plot_dir, 'plot_05_modell_vergleich.png')}")
print(f"- {os.path.join(plot_dir, 'plot_06_residuen_vergleich.png')}")