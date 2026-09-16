# Projet UM5BM753 — Supervised and Unsupervised Learning for Biological Data

## Présentation
Projet réalisé dans le cadre de l'UE Supervised and Unsupervised Learning for Biological Data (M2 AIDA). Consignes détaillées : voir `02_Projects/00_docs/`.

## Sources
- `02_data/raw/10_Breast_Cancer_Wisconsin_Diagnostic.csv` : jeu de données Breast Cancer Wisconsin (Diagnostic), non modifié depuis sa source d'origine.

## Structure du dossier
Structure alignée sur la méthodologie AIDA du cours ("Working Effectively on an AIDA Project", diapo 7) :
- `01_scripts/` : fonctions Python réutilisables, importées par les notebooks (à peupler à partir du prochain notebook soumis — `01_Milestone1.ipynb` reste un brouillon autonome)
- `02_data/` : fichiers de données
  - `raw/` : fichiers sources inchangés
  - `processed/` : fichiers nettoyés prêts à l'emploi
  - `data_dictionary.md` : description des colonnes
- `03_notebooks/` : notebooks Jupyter, numérotés selon l'ordre d'exécution
- `04_results/` : tableaux exportés (`tables/`), `run_summary.json`
- `05_figures/` : graphiques exportés
- `06_report/` : notes de synthèse, présentations, rapport final

## Marche à suivre
1. Créer/activer l'environnement virtuel du projet (`requirements.txt`).
2. Exécuter les notebooks de `03_notebooks/` dans l'ordre numéroté.
3. Les productions sont écrites dans `04_results/` (tableaux) et `05_figures/` (graphiques).

## Notes
- `01_Milestone1.ipynb` et les autres notebooks de `03_notebooks/` sont des brouillons de travail. Le notebook final soumis au professeur suivra la structure recommandée par la méthodologie AIDA : fonctions réutilisables extraites vers `01_scripts/`, notebook réduit à l'histoire scientifique (Context → Gap → Question → Data → Approach → Results → Conclusion).
