# GeoRaster Tools — Plugin QGIS

**Auteur** : LOMPO Ibrahim (Burkina Faso, Afrique de l'Ouest)  
**Email** : ibrahimlompo33@gmail.com  
**Licence** : GNU GPL v2  
**Version** : 1.1.1  
**QGIS minimum** : 3.16  

---

## Description

Plugin QGIS de traitement raster complet développé par LOMPO Ibrahim.  
Regroupe en une seule interface 8 outils couvrant la chaîne complète d'analyse raster :
vectorisation, indices spectraux, analyse terrain, calcul raster sécurisé,
statistiques zonales exactes, traitements courants, et classification ML supervisée/non supervisée.

---

## Fonctionnalités

### 🗺 Vectorisation
- Polygonisation via `gdal.Polygonize`
- Courbes de niveau via `gdal.ContourGenerate`

### 📡 Indices spectraux
| Indice | Formule |
|--------|---------|
| NDVI | (NIR − R) / (NIR + R) |
| NDWI | (Vert − NIR) / (Vert + NIR) |
| NDBI | (MIR − NIR) / (MIR + NIR) |
| EVI  | 2.5 × (NIR − R) / (NIR + 6R − 7.5B + 1) |
| SAVI | 1.5 × (NIR − R) / (NIR + R + L) — L paramétrable |

### ⛰ Analyse de terrain (DEM/MNT)
- Pente, Orientation, Ombrage (Hillshade), TPI, TRI
- Correction automatique du facteur d'échelle pour les CRS géographiques (degrés)

### 🔢 Calcul raster sécurisé
- Formules personnalisées via `A[b]` (bande b)
- Évaluation par AST whitelist — aucune exécution de code arbitraire
- Fonctions `np.*` autorisées : `where`, `clip`, `sqrt`, `log`, `exp`, `abs`, etc.

### 📊 Statistiques zonales
- Globales ou par polygone
- Masque géométrique exact (pas d'approximation par bounding box)
- Min, Max, Moyenne, Std, Médiane, P5/P95, comptage pixels

### 🔄 Traitements courants
- Reprojection (EPSG validé)
- Rééchantillonnage (résolution personnalisée, Bilinear)
- Découpage par couche vecteur
- Mosaïque multi-fichiers
- Raster → Points (vectorisation NumPy, haute performance)

### 🤖 Classification non supervisée — K-Means
- Normalisation StandardScaler automatique
- Score de silhouette optionnel
- Sélection de bandes personnalisée

### 🌲 Classification supervisée — Random Forest
- Split train/test stratifié paramétrable
- Métriques : Overall Accuracy, Kappa, matrice de confusion, rapport par classe
- Export de l'importance des variables (feature importances)
- Encodage reproductible des classes (LabelEncoder trié)

---

## Installation

### Depuis le dépôt QGIS
`Plugins → Gérer et installer → Rechercher "GeoRaster Tools"`

### Manuellement
1. Télécharger le `.zip`
2. `Plugins → Gérer et installer → Installer depuis un ZIP`

### Dépendances optionnelles (classification ML)
```bash
pip install scikit-learn
```

---

## Structure du projet
```
raster_vectorizer/
├── __init__.py             # Entry point QGIS
├── raster_vectorizer.py    # Code principal (8 onglets)
├── metadata.txt            # Métadonnées plugin
├── icon.png                # Icône 64×64 px
├── LICENSE.txt             # GNU GPL v2
├── README.md               # Ce fichier
├── CHANGELOG.md            # Historique des versions
└── requirements.txt        # Dépendances Python
```

---

## Notes techniques

- Tous les traitements lourds sont exécutés hors du thread UI via `QgsTask`
- Sorties GeoTIFF compressées LZW + tuilées (TILED=YES)
- Logging dans le panneau QGIS : `Vue → Panneaux → Messages` (tag `GeoRasterTools`)

---

## Licence

Ce plugin est distribué sous licence **GNU General Public License v2**.  
Voir `LICENSE.txt` ou [https://www.gnu.org/licenses/old-licenses/gpl-2.0.html](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)

---

*Développé avec ❤ au Burkina Faso par LOMPO Ibrahim — v1.1.1*
# GeoRaster-Tools
