# Dictionnaire de données — Breast Cancer Wisconsin (Diagnostic)

Fichier : `raw/10_Breast_Cancer_Wisconsin_Diagnostic.csv` — 569 observations, 32 colonnes.

| Colonne       | Description                                                                      |
| ------------- | -------------------------------------------------------------------------------- |
| `id`        | Identifiant unique du patient                                                    |
| `diagnosis` | Diagnostic :`M` = malin (malignant), `B` = bénin (benign) — variable cible |

Les 30 colonnes restantes sont des caractéristiques calculées à partir de l'image numérisée d'une cytoponction (FNA) de masse mammaire, décrivant le noyau cellulaire. Pour chacune des 10 caractéristiques de base, trois statistiques sont fournies sur l'ensemble des noyaux de l'image : 

- `_mean` (moyenne arithmétique de la caractéristique mesurée sur l'ensemble des noyaux cellulaires analysés dans l'image),
- `_se` (erreur standard, qui quantifie la variabilité ou la dispersion des mesures au sein de l'échantillon cellulaire),
- `_worst` (valeur "maximale" ou pire cas, calculée comme la moyenne des 3 valeurs les plus élevées, ciblant les cellules les plus atypiques).

Caractéristiques de base :

- `radius` : rayon moyen (moyenne des distances du centre géométrique du noyau aux points du périmètre)
- `texture` : écart-type des niveaux de gris, , mesurant les variations de luminosité et l'aspect granuleux de la surface cellulaire
- `perimeter` : longueur totale du contour extérieur qui enveloppe le noyau cellulaire
- `area` : surface totale à deux dimensions occupée par le noyau de la cellule
- `smoothness` : variation locale de la longueur des rayons, évaluant la régularité ou le caractère rugueux du bord
- `compactness` : périmètre² / aire − 1.0, traduisant la compacité et la rondeur de la forme
- `concavity` : sévérité et profondeur des portions concaves/creuses ou rentrantes observées le long du contour
- `concave_points` : nombre de portions concaves du contour
- `symmetry` : degré de correspondance symétrique de la forme du noyau par rapport à ses axes médians
- `fractal_dimension` : approximation de la complexité géométrique et de l'irrégularité structurelle du contour

Exemple : `radius_mean`, `radius_se`, `radius_worst`.

Aucun nettoyage n'a été appliqué à ce fichier (dossier `raw/`). Toute version nettoyée/transformée doit être écrite dans `processed/` avec une note décrivant les transformations effectuées.
