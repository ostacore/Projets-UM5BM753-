# Dictionnaire de données — Breast Cancer Wisconsin (Diagnostic)

Fichier : `brutes/10_Breast_Cancer_Wisconsin_Diagnostic.csv` — 569 observations, 32 colonnes.

| Colonne | Description |
|---|---|
| `id` | Identifiant unique du patient |
| `diagnosis` | Diagnostic : `M` = malin (malignant), `B` = bénin (benign) — variable cible |

Les 30 colonnes restantes sont des caractéristiques calculées à partir de l'image numérisée d'une cytoponction (FNA) de masse mammaire, décrivant le noyau cellulaire. Pour chacune des 10 caractéristiques de base, trois statistiques sont fournies sur l'ensemble des noyaux de l'image : `_mean` (moyenne), `_se` (erreur standard), `_worst` (moyenne des 3 valeurs les plus élevées).

Caractéristiques de base :
- `radius` : rayon moyen (moyenne des distances du centre aux points du périmètre)
- `texture` : écart-type des niveaux de gris
- `perimeter`
- `area`
- `smoothness` : variation locale de la longueur des rayons
- `compactness` : périmètre² / aire − 1.0
- `concavity` : sévérité des portions concaves du contour
- `concave_points` : nombre de portions concaves du contour
- `symmetry`
- `fractal_dimension` : approximation de la « côte-line » − 1

Exemple : `radius_mean`, `radius_se`, `radius_worst`.

Aucun nettoyage n'a été appliqué à ce fichier (dossier `brutes/`). Toute version nettoyée/transformée doit être écrite dans `traitees/` avec une note décrivant les transformations effectuées.
