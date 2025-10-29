"""
Decision Trees - Iris Flower Classification
Korrespondierendes Python-Skript zum Notebook decision_trees_iris.ipynb

Speichert alle Plots als PNG-Dateien ohne Anzeige (non-interactive backend)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')

# Plot-Verzeichnis erstellen
PLOT_DIR = "iris_decision_tree_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Für schönere Plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def save_plot(filename, title=None):
    """Speichert Plot als PNG und schließt die Figure"""
    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')
    print(f"Plot gespeichert: {filename}")
    plt.close()

def load_and_explore_data():
    """Lädt und exploriert den Iris-Datensatz"""
    print("=" * 60)
    print("🌺 SCHRITT 1: IRIS-DATEN LADEN UND VERSTEHEN")
    print("=" * 60)
    
    # Iris-Datensatz laden
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # DataFrame mit Klassen-Namen erstellen
    df = X.copy()
    df['Blumenart'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("🎉 Iris-Datensatz erfolgreich geladen!")
    print(f"📊 Daten: {len(df)} Blumen, {len(df.columns)-1} Eigenschaften")
    print(f"🌸 Arten: {', '.join(iris.target_names)}")
    
    # Basis-Statistiken
    print("\n📈 Grundlegende Statistiken:")
    print(df.describe())
    
    # Klassen-Verteilung
    print(f"\n🎯 Verteilung der Blumenarten:")
    class_counts = df['Blumenart'].value_counts()
    for art, anzahl in class_counts.items():
        print(f"   {art}: {anzahl} Blumen")
    
    # Korrelations-Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Korrelation zwischen Blüten-Eigenschaften')
    save_plot("01_correlation_heatmap.png")
    
    # Pairplot - Beziehungen zwischen Features
    plt.figure(figsize=(12, 10))
    pair_plot = sns.pairplot(df, hue='Blumenart', diag_kind='hist', 
                            plot_kws={'alpha': 0.7})
    pair_plot.fig.suptitle('Paarweise Beziehungen aller Eigenschaften', 
                          y=1.02, fontsize=16)
    save_plot("02_pairplot_features.png")
    
    # Detaillierte Scatter-Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Kelchblatt: Länge vs. Breite
    sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                   hue='Blumenart', ax=axes[0], s=80)
    axes[0].set_title('Kelchblatt: Länge vs. Breite')
    
    # Blütenblatt: Länge vs. Breite  
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                   hue='Blumenart', ax=axes[1], s=80)
    axes[1].set_title('Blütenblatt: Länge vs. Breite')
    
    # Kelch- vs. Blütenblatt-Länge
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                   hue='Blumenart', ax=axes[2], s=80)
    axes[2].set_title('Kelch- vs. Blütenblatt-Länge')
    
    plt.tight_layout()
    save_plot("03_detailed_scatterplots.png", "Detaillierte Feature-Analysen")
    
    return iris, X, y, df

def train_first_tree(iris, X, y):
    """Trainiert den ersten Entscheidungsbaum"""
    print("\n" + "=" * 60)
    print("🌳 SCHRITT 2: ERSTEN ENTSCHEIDUNGSBAUM TRAINIEREN")
    print("=" * 60)
    
    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"🚂 Training: {len(X_train)} Blumen")
    print(f"🧪 Test: {len(X_test)} Blumen")
    print(f"⚖️ Verhältnis: {len(X_train)/len(X_test):.1f}:1 (Train:Test)")
    
    # Unbegrenzten Baum erstellen
    tree_unlimited = DecisionTreeClassifier(random_state=42, criterion='gini')
    tree_unlimited.fit(X_train, y_train)
    
    # Vorhersagen und Bewertung
    y_pred = tree_unlimited.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✨ Training abgeschlossen!")
    print(f"🎯 Genauigkeit auf Testdaten: {accuracy:.1%}")
    print(f"🌿 Baumtiefe: {tree_unlimited.get_depth()} Ebenen")
    print(f"🍃 Anzahl Blätter: {tree_unlimited.get_n_leaves()}")
    
    return X_train, X_test, y_train, y_test, tree_unlimited

def visualize_tree(tree, iris, filename_suffix=""):
    """Visualisiert einen Entscheidungsbaum"""
    print(f"\n👀 Visualisiere Entscheidungsbaum{filename_suffix}...")
    
    plt.figure(figsize=(20, 12))
    plot_tree(tree, feature_names=iris.feature_names, 
              class_names=iris.target_names, filled=True, rounded=True, fontsize=12)
    title = f'Entscheidungsbaum{filename_suffix} (Tiefe: {tree.get_depth()}, Blätter: {tree.get_n_leaves()})'
    save_plot(f"04_decision_tree{filename_suffix.lower().replace(' ', '_')}.png", title)

def analyze_feature_importance(tree, iris):
    """Analysiert Feature-Wichtigkeit"""
    print("\n🔍 FEATURE-WICHTIGKEIT ANALYSE")
    print("-" * 40)
    
    importances = tree.feature_importances_
    feature_names = iris.feature_names
    
    # Sortierte Wichtigkeiten
    indices = np.argsort(importances)[::-1]
    
    print("🏆 Wichtigkeit der Eigenschaften:")
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    # Visualisierung
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], color='skyblue', alpha=0.8)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Wichtigkeit')
    plt.title('Feature-Wichtigkeit im Entscheidungsbaum')
    save_plot("05_feature_importance.png")

def create_confusion_matrix(y_test, y_pred, iris):
    """Erstellt und visualisiert Confusion Matrix"""
    print("\n📊 CONFUSION MATRIX")
    print("-" * 30)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('Vorhergesagte Klasse')
    plt.title('Confusion Matrix - Entscheidungsbaum')
    save_plot("06_confusion_matrix.png")
    
    # Klassifikationsbericht
    print("📋 Detaillierter Klassifikationsbericht:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

def experiment_with_depths(X_train, X_test, y_train, y_test):
    """Experimentiert mit verschiedenen Baumtiefen"""
    print("\n" + "=" * 60)
    print("🛡️ SCHRITT 5: OVERFITTING VERMEIDEN - PARAMETER TESTEN")
    print("=" * 60)
    
    depths = range(1, 11)
    train_accuracies = []
    test_accuracies = []
    
    print("🧪 Teste verschiedene Baumtiefen...")
    
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        
        train_acc = tree.score(X_train, y_train)
        test_acc = tree.score(X_test, y_test)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"   Tiefe {depth}: Training {train_acc:.1%}, Test {test_acc:.1%}")
    
    # Visualisierung
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accuracies, 'o-', label='Training-Genauigkeit', 
             color='green', linewidth=2, markersize=8)
    plt.plot(depths, test_accuracies, 's-', label='Test-Genauigkeit', 
             color='red', linewidth=2, markersize=8)
    plt.xlabel('Maximale Baumtiefe')
    plt.ylabel('Genauigkeit')
    plt.title('Training vs. Test Genauigkeit bei verschiedenen Baumtiefen')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("07_depth_comparison.png")
    
    best_depth = depths[np.argmax(test_accuracies)]
    best_test_acc = max(test_accuracies)
    print(f"\n🏆 Beste Tiefe: {best_depth} (Test-Genauigkeit: {best_test_acc:.1%})")
    
    return best_depth

def experiment_with_min_samples(X_train, X_test, y_train, y_test, best_depth):
    """Experimentiert mit min_samples_leaf Parameter"""
    print("\n🧪 Teste verschiedene minimale Blattgrößen...")
    
    min_samples_values = [1, 2, 5, 10, 15, 20]
    test_accs_samples = []
    
    for min_samples in min_samples_values:
        tree = DecisionTreeClassifier(
            max_depth=best_depth, 
            min_samples_leaf=min_samples, 
            random_state=42
        )
        tree.fit(X_train, y_train)
        test_acc = tree.score(X_test, y_test)
        test_accs_samples.append(test_acc)
        print(f"   min_samples_leaf {min_samples}: Test {test_acc:.1%}")
    
    # Visualisierung
    plt.figure(figsize=(10, 6))
    plt.plot(min_samples_values, test_accs_samples, 'o-', 
             color='purple', linewidth=2, markersize=8)
    plt.xlabel('Minimale Anzahl Samples pro Blatt')
    plt.ylabel('Test-Genauigkeit')
    plt.title('Einfluss von min_samples_leaf auf Test-Genauigkeit')
    plt.grid(True, alpha=0.3)
    save_plot("08_min_samples_leaf_comparison.png")
    
    best_min_samples = min_samples_values[np.argmax(test_accs_samples)]
    best_acc = max(test_accs_samples)
    print(f"🏆 Beste min_samples_leaf: {best_min_samples} (Test-Genauigkeit: {best_acc:.1%})")
    
    return best_min_samples

def create_optimal_tree(X_train, X_test, y_train, y_test, best_depth, best_min_samples, iris):
    """Erstellt optimalen Baum mit besten Parametern"""
    print(f"\n🎯 Erstelle optimalen Baum (depth={best_depth}, min_samples_leaf={best_min_samples})...")
    
    optimal_tree = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_leaf=best_min_samples,
        random_state=42
    )
    optimal_tree.fit(X_train, y_train)
    
    # Bewertung
    train_acc = optimal_tree.score(X_train, y_train)
    test_acc = optimal_tree.score(X_test, y_test)
    
    print(f"🏆 OPTIMALER BAUM:")
    print(f"   🎯 Training-Genauigkeit: {train_acc:.1%}")
    print(f"   🎯 Test-Genauigkeit: {test_acc:.1%}")
    print(f"   🌿 Baumtiefe: {optimal_tree.get_depth()}")
    print(f"   🍃 Anzahl Blätter: {optimal_tree.get_n_leaves()}")
    
    # Visualisierung des optimalen Baums
    visualize_tree(optimal_tree, iris, " (Optimiert)")
    
    # Feature-Wichtigkeit des optimalen Baums
    analyze_feature_importance(optimal_tree, iris)
    
    # Confusion Matrix für optimalen Baum
    y_pred_optimal = optimal_tree.predict(X_test)
    create_confusion_matrix(y_test, y_pred_optimal, iris)
    
    return optimal_tree

def demonstrate_prediction_path(tree, iris, X_test):
    """Demonstriert Vorhersage-Pfad für ein Beispiel"""
    print("\n" + "=" * 60)
    print("🎯 VORHERSAGE-PFAD DEMONSTRATION")  
    print("=" * 60)
    
    # Beispiel-Blume auswählen
    example_idx = 0
    example_flower = X_test.iloc[example_idx:example_idx+1]
    
    print(f"🌸 Beispiel-Blume (Index {example_idx}):")
    for feature_name, value in zip(iris.feature_names, example_flower.iloc[0]):
        print(f"   {feature_name}: {value:.2f} cm")
    
    # Vorhersage
    prediction = tree.predict(example_flower)[0]
    probability = tree.predict_proba(example_flower)[0]
    
    print(f"\n🔮 Vorhersage: {iris.target_names[prediction]}")
    print(f"🎲 Wahrscheinlichkeiten:")
    for i, prob in enumerate(probability):
        print(f"   {iris.target_names[i]}: {prob:.1%}")

def main():
    """Hauptfunktion - führt alle Schritte aus"""
    print("🌺 IRIS ENTSCHEIDUNGSBAUM ANALYSE")
    print("=" * 60)
    print(f"Alle Plots werden im Verzeichnis '{PLOT_DIR}' gespeichert.\n")
    
    # Schritt 1: Daten laden und explorieren
    iris, X, y, df = load_and_explore_data()
    
    # Schritt 2: Ersten Baum trainieren
    X_train, X_test, y_train, y_test, tree_unlimited = train_first_tree(iris, X, y)
    
    # Schritt 3: Baum visualisieren
    visualize_tree(tree_unlimited, iris)
    
    # Schritt 4: Feature-Wichtigkeit und Confusion Matrix
    analyze_feature_importance(tree_unlimited, iris)
    y_pred = tree_unlimited.predict(X_test)
    create_confusion_matrix(y_test, y_pred, iris)
    
    # Schritt 5: Parameter-Optimierung
    best_depth = experiment_with_depths(X_train, X_test, y_train, y_test)
    best_min_samples = experiment_with_min_samples(X_train, X_test, y_train, y_test, best_depth)
    
    # Schritt 6: Optimalen Baum erstellen
    optimal_tree = create_optimal_tree(X_train, X_test, y_train, y_test, 
                                     best_depth, best_min_samples, iris)
    
    # Schritt 7: Vorhersage-Demonstration
    demonstrate_prediction_path(optimal_tree, iris, X_test)
    
    print(f"\n🎉 ANALYSE ABGESCHLOSSEN!")
    print(f"📁 Alle {len([f for f in os.listdir(PLOT_DIR) if f.endswith('.png')])} Plots wurden als PNG-Dateien in '{PLOT_DIR}' gespeichert.")
    print("\n🎓 ZUSAMMENFASSUNG:")
    print("   ✅ Iris-Datensatz analysiert und visualisiert")
    print("   ✅ Verschiedene Entscheidungsbäume trainiert")
    print("   ✅ Parameter optimiert gegen Overfitting")
    print("   ✅ Feature-Wichtigkeit analysiert")
    print("   ✅ Vorhersage-Qualität bewertet")

if __name__ == "__main__":
    main()