# Changelog — GeoRaster Tools

Toutes les modifications notables sont documentées dans ce fichier.
Format : [Semantic Versioning](https://semver.org/lang/fr/)

---

## [1.1.1] — 2026-05-04

### Corrections de bugs (audit)

- **[CRIT]** RF KFold : scorer `"cohen_kappa"` remplacé par `make_scorer(cohen_kappa_score)` — l'ancienne chaîne n'existe pas dans `sklearn.metrics.SCORERS` et levait une `ValueError` à l'exécution
- **[CRIT]** Registre fichiers temporaires : `_tmp()` alimente désormais `_TMP_REGISTRY` (module-level) — `unload()` nettoie réellement les fichiers orphelins au lieu d'itérer une liste vide
- **[MAJ]** `_BandProxy.__getitem__` : `isinstance(b, numbers.Integral)` remplace `isinstance(b, int)` — accepte `np.int32`/`np.int64`/`np.intp` dans les formules de calcul
- **[MAJ]** Découpage : `cutlineSRS` et `dstNodata=-9999` ajoutés à `gdal.Warp` — évite un clip silencieusement incorrect lorsque le CRS du vecteur diffère du raster
- **[MAJ]** EVI : `safe_denom` introduit avant `np.where` — supprime la division réelle par zéro et les `RuntimeWarning: divide by zero` sur grands rasters
- **[MAJ]** Stats zonales : avertissement UI + log si `nodata=None` en mode zones (pixels hors emprise potentiellement inclus)
- **[MIN]** `_safe_eval` AST : validation que tout accès attribut porte sur `np.*` exclusivement (`node.value.id == "np"`) — sandbox plus stricte
- **[MIN]** Reprojection : `dstNodata=-9999` ajouté à `gdal.Warp` — supprime le remplissage silencieux à 0 hors emprise

---

## [1.1.0] — 2026-05-03

### Sécurité
- `eval()` remplacé par `_safe_eval()` avec validation AST whitelist (29 nœuds autorisés)
- Aucune exécution de code arbitraire possible via l'interface de calcul

### Performances
- Raster → Points : vectorisation NumPy (suppression boucle Python O(n²) → O(n))
- Transactions OGR groupées pour la création des points

### Corrections algorithmiques
- Stats zonales : masque géométrique exact par `gdal.RasterizeLayer` (suppression approx. MBR)
- K-Means : normalisation `StandardScaler` avant clustering (bandes à grande amplitude)
- Random Forest : split train/test stratifié + métriques OA/Kappa/matrice de confusion
- Random Forest : `LabelEncoder` trié → encodage reproductible entre sessions
- EVI : masque dénominateur ≈ 0 au lieu de `clip` aveugle
- `_nodata_mask()` unifié — `None`-safe, `isclose` float-safe

### Nouvelles fonctionnalités
- RF : export `feature_importances_` avec bar chart ASCII
- RF : fenêtre de résultats scrollable
- K-Means : score de silhouette optionnel (subsample 50k pixels)
- SAVI : paramètre L exposé en interface (0 = sol nu, 1 = couverture dense)
- Hillshade : détection automatique CRS géographique + facteur d'échelle paramétrable
- EVI : avertissement réflectances normalisées [0,1]
- Logging `QgsMessageLog` (tag `GeoRasterTools`) sur toutes les opérations

### Architecture
- Threading : `QgsTask` sur tous les onglets — UI non bloquée
- `_validate_bands()` systématique avant tout `GetRasterBand()`
- `_epsg_to_wkt()` avec validation `try/except` + message explicite
- VRT mosaïque : `FlushCache()` + nettoyage fichier temporaire
- GeoTIFF sortie : `COMPRESS=LZW + TILED=YES + BIGTIFF=IF_SAFER`
- RF `n_jobs` limité à `cpu_count()//2` (non-saturant)
- Registre `_tmp_files` + nettoyage dans `unload()`

---

## [1.0.0] — 2026-04-05

### Initial
- Vectorisation (polygones + courbes de niveau)
- Indices spectraux (NDVI, NDWI, NDBI, EVI, SAVI)
- Analyse terrain (pente, aspect, hillshade, TPI, TRI)
- Calcul raster personnalisé
- Statistiques raster
- Traitements (reprojection, rééchantillonnage, découpage, mosaïque, raster→points)
- Classification non supervisée K-Means
- Classification supervisée Random Forest
