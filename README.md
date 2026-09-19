# Projet UM5BM753 — Supervised and Unsupervised Learning for Biological Data

## Présentation
Projet réalisé dans le cadre de l'UE Supervised and Unsupervised Learning for Biological Data (M2 AIDA). Consignes détaillées : voir `02_Projects/00_docs/`.

## Sources
- `02_data/raw/10_Breast_Cancer_Wisconsin_Diagnostic.csv` : jeu de données Breast Cancer Wisconsin (Diagnostic), non modifié depuis sa source d'origine.

## Structure du dossier
Structure alignée sur la méthodologie AIDA du cours ("Working Effectively on an AIDA Project", diapo 7) :
- `01_scripts/` : fonctions Python réutilisables, importées par les notebooks
- `02_data/` : fichiers de données
  - `raw/` : fichiers sources inchangés
  - `processed/` : fichiers nettoyés prêts à l'emploi
- `03_notebooks/` : les 6 notebooks du rapport final, numérotés selon l'ordre d'exécution (01_Data_QC → 06_Zoom_Spectral_Clustering)
- `04_results/` : tableaux exportés (`.csv`)
- `05_figures/` : graphiques exportés
- `06_report/` : notes de synthèse, présentations, rapport final

## Marche à suivre
1. Créer/activer l'environnement virtuel du projet (`requirements.txt`).
2. Exécuter les notebooks de `03_notebooks/` dans l'ordre numéroté.
3. Les productions sont écrites dans `04_results/` (tableaux) et `05_figures/` (graphiques).

## Notes
- Les brouillons de travail (`00_Final_Report_.ipynb` historique, `01_Milestone1.ipynb`, et les drafts des coéquipiers) sont rangés dans `autres/`, hors de `03_notebooks/` qui ne contient que les notebooks destinés à master.
- `[TODO]` lister ici les principaux fichiers produits par chaque notebook (tables dans `04_results/`, figures dans `05_figures/`), pour remplacer l'ancien `run_summary.json` (retiré : il datait de `01_Milestone1.ipynb` et n'était plus synchronisé avec le pipeline actuel).
