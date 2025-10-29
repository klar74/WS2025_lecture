"""
Decision Tree Classifier Beispiele und Visualisierungen

Quelle: Extrahiert aus kap4.ipynb - Kapitel 4: Supervised Learning
Notebook aus: https://github.com/DJCordhose/buch-machine-learning-notebooks
Bearbeitet für DHBW Vorlesung "Grundlagen Data Science und Künstliche Intelligenz"
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
import os

# Setze Random State für Reproduzierbarkeit
np.random.seed(52)
CMAP = 'bwr'

# Pfad für Plots
PLOT_DIR = "decision_tree_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def create_sample_data():
    """Erstelle Beispieldaten für Decision Tree Demonstration"""
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=5.)
    return X, y

def plot_data(X, y, title="Beispieldaten", filename=None):
    """Visualisiere die Datenpunkte"""
    fig = plt.figure(1, figsize=(9, 6))
    plt.scatter(X[:, 0], X[:, 1], s=80, c=y, cmap=CMAP)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {filename}")
    plt.close()

def plot_decision_boundary(X, y, tree, title="Decision Tree Classifier", filename=None):
    """Visualisiere Decision Boundary des Klassifikators"""
    fig = plt.figure(1, figsize=(9, 6))
    ax = plt.gca()
    
    ax.scatter(X[:, 0], X[:, 1], c=y, s=80, cmap=CMAP, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.7,
                          levels=np.arange(n_classes + 1) - 0.5,
                          cmap=CMAP,
                          zorder=1)
    
    ax.set(xlim=xlim, ylim=ylim)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {filename}")
    plt.close()

def demonstrate_tree_depths():
    """Demonstriere verschiedene Tree Depths"""
    X, y = create_sample_data()
    
    print("=== Decision Tree Classifier mit verschiedenen Tiefen ===\n")
    
    # Grundlegende Visualisierung der Daten
    plot_data(X, y, "Originaldaten für Decision Tree", "01_originaldaten.png")
    
    # Standard Decision Tree (unbeschränkte Tiefe)
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    print(f"Standard Decision Tree Score: {tree.score(X, y):.4f}")
    plot_decision_boundary(X, y, tree, "Standard Decision Tree (unbeschränkte Tiefe)", "02_standard_tree.png")
    
    # Wichtige Eigenschaften des gefitteten Baums
    print("\n--- Eigenschaften des Standard Decision Trees ---")
    print(f"Maximale Tiefe: {tree.get_depth()}")
    print(f"Anzahl Blätter: {tree.get_n_leaves()}")
    print(f"Anzahl Features: {tree.n_features_in_}")
    print(f"Anzahl Klassen: {tree.n_classes_}")
    print(f"Feature-Namen: {tree.feature_names_in_ if hasattr(tree, 'feature_names_in_') else 'Keine Namen definiert'}")
    
    # Feature Importance
    print(f"\nFeature Importance:")
    for i, importance in enumerate(tree.feature_importances_):
        print(f"  Feature {i}: {importance:.4f}")
    
    # Tree-Struktur Details
    print(f"\nTree-Struktur Details:")
    print(f"  Gesamtanzahl Knoten: {tree.tree_.node_count}")
    print(f"  Anzahl interne Knoten: {tree.tree_.node_count - tree.get_n_leaves()}")
    print(f"  Anzahl Samples pro Klasse: {dict(zip(tree.classes_, [sum(y == c) for c in tree.classes_]))}")

    # Verschiedene max_depth Werte testen
    depths = [1, 2, 3, 4, 5]
    
    for depth in depths:
        print(f"\n--- Max Depth: {depth} ---")
        tree_depth = DecisionTreeClassifier(max_depth=depth)
        tree_depth.fit(X, y)
        score = tree_depth.score(X, y)
        print(f"Accuracy: {score:.4f}")
        plot_decision_boundary(X, y, tree_depth, f"Decision Tree (max_depth={depth})", f"03_tree_depth_{depth}.png")

def demonstrate_regularization():
    """Demonstriere Regularisierungstechniken gegen Overfitting"""
    X, y = create_sample_data()
    
    print("\n=== Regularisierung gegen Overfitting ===\n")
    
    # 1. min_samples_leaf Parameter
    print("--- Regularisierung mit min_samples_leaf ---")
    tree_reg1 = DecisionTreeClassifier(max_depth=5,
                                      min_samples_leaf=3,
                                      min_samples_split=2)
    tree_reg1.fit(X, y)
    print(f"Score (min_samples_leaf=3): {tree_reg1.score(X, y):.4f}")
    plot_decision_boundary(X, y, tree_reg1, "Decision Tree (min_samples_leaf=3)", "04_reg_min_samples_leaf.png")
    
    # 2. min_samples_split Parameter
    print("--- Regularisierung mit min_samples_split ---")
    tree_reg2 = DecisionTreeClassifier(max_depth=5,
                                      min_samples_leaf=1,
                                      min_samples_split=10)
    tree_reg2.fit(X, y)
    print(f"Score (min_samples_split=10): {tree_reg2.score(X, y):.4f}")
    plot_decision_boundary(X, y, tree_reg2, "Decision Tree (min_samples_split=10)", "05_reg_min_samples_split.png")
    
    # 3. max_leaf_nodes Parameter
    print("--- Regularisierung mit max_leaf_nodes ---")
    tree_reg3 = DecisionTreeClassifier(max_depth=5,
                                      min_samples_leaf=1,
                                      min_samples_split=2,
                                      max_leaf_nodes=8)
    tree_reg3.fit(X, y)
    print(f"Score (max_leaf_nodes=8): {tree_reg3.score(X, y):.4f}")
    plot_decision_boundary(X, y, tree_reg3, "Decision Tree (max_leaf_nodes=8)", "06_reg_max_leaf_nodes.png")

def demonstrate_split_criteria():
    """Demonstriere verschiedene Split-Kriterien"""
    X, y = create_sample_data()
    
    print("\n=== Split-Kriterien: Gini vs. Entropy ===\n")
    
    # Gini Impurity (Standard)
    tree_gini = DecisionTreeClassifier(max_depth=5, criterion='gini')
    tree_gini.fit(X, y)
    print(f"Gini Impurity Score: {tree_gini.score(X, y):.4f}")
    plot_decision_boundary(X, y, tree_gini, "Decision Tree (Gini Impurity)", "07_gini_criterion.png")
    
    # Entropy
    tree_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy')
    tree_entropy.fit(X, y)
    print(f"Entropy Score: {tree_entropy.score(X, y):.4f}")
    plot_decision_boundary(X, y, tree_entropy, "Decision Tree (Entropy)", "08_entropy_criterion.png")

def create_tree_structure_visualization():
    """Erstelle eine manuelle Visualisierung der Tree-Struktur"""
    fig = plt.figure(1, figsize=(9, 6))
    ax = plt.gca()

    def textbox(ax, x, y, t, size=10, fc='w', ec='k', bstyle='round4', **kwargs):
        ax.text(x, y, t, ha='center', va='center', size=size,
                bbox=dict(boxstyle=bstyle, pad=0.5, ec=ec, fc=fc), **kwargs)

    slev0 = 12
    slev1 = 10
    slev2 = 8
    slev3 = 6

    # Level Labels
    textbox(ax, 1.2, 0.9, "Level 1", slev1, alpha=0.99, color='k', fc='orange', bstyle='larrow')
    textbox(ax, 1.2, 0.6, "Level 2", slev1, alpha=0.99, color='k', fc='orange', bstyle='larrow')
    textbox(ax, 1.2, 0.3, "Level 3", slev1, alpha=0.99, color='k', fc='orange', bstyle='larrow')

    # Decision Nodes
    textbox(ax, 0.5, 0.9, "x > 2 ?", slev0)
    textbox(ax, 0.3, 0.6, "y > 0 ?", slev1)
    textbox(ax, 0.7, 0.6, "y > -8 ?", slev1)
    textbox(ax, 0.12, 0.3, "x > 4 ?", slev2)
    textbox(ax, 0.62, 0.3, "y > -6 ?", slev2)
    textbox(ax, 0.88, 0.3, "y > -6 ?", slev2)

    # True/False Labels
    textbox(ax, 0.4, 0.75, "true", slev0, alpha=0.99, color='g')
    textbox(ax, 0.6, 0.75, "false", slev0, alpha=0.99, color='m')
    textbox(ax, 0.21, 0.45, "true", slev1, alpha=0.99, color='g')
    textbox(ax, 0.34, 0.45, "false", slev1, alpha=0.99, color='m')
    textbox(ax, 0.66, 0.45, "true", slev1, alpha=0.99, color='g')
    textbox(ax, 0.79, 0.45, "false", slev1, alpha=0.99, color='m')

    # Leaf Nodes
    textbox(ax, .0, 0., "class1", slev2, alpha=0.99, color='w', fc='b', bstyle='square')
    textbox(ax, .20, 0., "class2", slev2, alpha=0.99, color='w', fc='r', bstyle='square')
    textbox(ax, 0.38, 0.3, "class1", slev2, alpha=0.99, color='w', fc='b', bstyle='square')
    textbox(ax, .56, 0., "class1", slev2, alpha=0.99, color='w', fc='b', bstyle='square')
    textbox(ax, .68, 0., "class2", slev2, alpha=0.99, color='w', fc='r', bstyle='square')
    textbox(ax, .8, 0., "class2", slev2, alpha=0.99, color='w', fc='r', bstyle='square')
    textbox(ax, 1., 0., "class1", slev2, alpha=0.99, color='w', fc='b', bstyle='square')

    # Verbindungslinien
    ax.plot([0.3, 0.5, 0.7], [0.6, 0.9, 0.6], '-k')
    ax.plot([0.12, 0.3, 0.38], [0.3, 0.6, 0.3], '-k')
    ax.plot([0.62, 0.7, 0.88], [0.3, 0.6, 0.3], '-k')
    ax.plot([0.0, 0.12, 0.20], [0.0, 0.3, 0.0], '-k')
    ax.plot([0.56, 0.62, 0.68], [0.0, 0.3, 0.0], '-k')
    ax.plot([0.8, 0.88, 1.0], [0.0, 0.3, 0.0], '-k')
    
    ax.axis([0, 1, 0, 1])
    ax.axis('off')
    plt.title("Decision Tree Struktur")
    
    plt.savefig(os.path.join(PLOT_DIR, "09_tree_structure.png"), dpi=300, bbox_inches='tight')
    print("Plot gespeichert: 09_tree_structure.png")
    plt.close()

def main():
    """Hauptfunktion - führe alle Demonstrationen aus"""
    print("Decision Tree Classifier - Umfassende Demonstration")
    print("=" * 55)
    print(f"Alle Plots werden im Verzeichnis '{PLOT_DIR}' gespeichert.\n")
    
    # 1. Grundlegende Tree Depths
    demonstrate_tree_depths()
    
    # 2. Regularisierungstechniken
    demonstrate_regularization()
    
    # 3. Split-Kriterien
    demonstrate_split_criteria()
    
    # 4. Tree-Struktur Visualisierung
    print("\n=== Decision Tree Struktur (Konzeptionell) ===")
    create_tree_structure_visualization()
    
    print(f"\nDemo abgeschlossen! Alle Plots wurden als PNG-Dateien in '{PLOT_DIR}' gespeichert.")

if __name__ == "__main__":
    main()