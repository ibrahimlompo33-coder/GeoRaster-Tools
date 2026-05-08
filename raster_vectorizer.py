# ===========================================================================
#  GeoRaster Tools
#  Version 1.1.1 — voir CHANGELOG en bas de fichier
# ===========================================================================
#
#  CORRECTIONS APPORTÉES :
#  [P0-SEC]  _safe_eval() remplace eval() non sandboxé (RasterCalcTab)
#  [P0-THR]  _Task (QgsTask) : tous les traitements hors thread UI
#  [P0-STAT] Stats zonales : masque géométrique réel par gdal.RasterizeLayer
#  [P1-ML]   K-Means : StandardScaler avant clustering
#  [P1-ML]   RF : OA, Kappa, matrice de confusion, feature_importances_
#  [P1-ML]   RF : LabelEncoder pour encodage reproductible des classes
#  [P1-PERF] Raster→Points : vectorisation NumPy (suppression boucle O(n²))
#  [P1-VALID] Validation band_ids vs RasterCount partout
#  [P2-UI]   SAVI : paramètre L exposé en UI
#  [P2-UI]   Hillshade : scale auto-détecté si CRS géographique (degrés)
#  [P2-GDAL] VRT : FlushCache() avant déréférencement
#  [P2-VALID] EPSG : try/except + message d'erreur clair
#  [P2-ND]   _nodata_mask() unifié — None-safe partout
#  [P2-LOG]  QgsMessageLog partout (tag "GeoRasterTools")
#  [P2-TMP]  Registre fichiers temporaires nettoyé à unload()
#  [P2-IO]   GeoTIFF compressé LZW+TILED par défaut
# ===========================================================================

import os
import ast
import numbers
import tempfile
import numpy as np

from qgis.PyQt.QtWidgets import (
    QAction, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFileDialog, QLineEdit, QGroupBox, QMessageBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget, QWidget, QTextEdit,
    QScrollArea,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsRasterLayer,
    QgsMapLayerProxyModel, Qgis, QgsTask, QgsApplication, QgsMessageLog,
)
from qgis.gui import QgsMapLayerComboBox
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

_LOG_TAG = "GeoRasterTools"
_TMP_REGISTRY: list = []   # [P2-TMP] registre global des fichiers temporaires orphelins


# ===========================================================================
# [P0-SEC] Évaluation sécurisée via AST whitelist
# ===========================================================================

_SAFE_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.BitAnd, ast.BitOr,
    ast.USub, ast.UAdd, ast.Not,
    ast.Constant,
    ast.Name, ast.Load,
    ast.Call, ast.Attribute,
    ast.Subscript, ast.Index,           # A[1]
    ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
    ast.And, ast.Or,
    ast.IfExp,                           # np.where(cond, a, b)
    ast.Tuple,
)

_ALLOWED_NP_ATTRS = {
    "where", "clip", "sqrt", "log", "log2", "log10", "exp",
    "abs", "sum", "mean", "std", "min", "max",
    "nan", "inf", "isnan", "isfinite", "isinf",
    "zeros_like", "ones_like", "full_like",
    "float32", "float64", "int16", "int32",
}


def _safe_eval(formula: str, env: dict):
    """
    Évalue une formule raster via AST whitelist.
    Lève ValueError si un nœud ou nom non autorisé est détecté.
    Pas d'accès aux builtins, aucune exécution arbitraire possible.
    """
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntaxe invalide : {exc}")

    allowed_names = set(env.keys())

    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_AST_NODES):
            raise ValueError(
                f"Construction AST non autorisée : {type(node).__name__!r}\n"
                f"Seules les opérations arithmétiques et np.* de base sont permises."
            )
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(
                f"Variable non autorisée : {node.id!r}. "
                f"Variables disponibles : {sorted(allowed_names)}"
            )
        if isinstance(node, ast.Attribute):
            # [P0-SEC] L'attribut doit obligatoirement porter sur l'objet np
            if not (isinstance(node.value, ast.Name) and node.value.id == "np"):
                raise ValueError(
                    f"Accès attribut interdit hors de np.* : {ast.dump(node)!r}\n"
                    f"Seuls les attributs np.<func> sont autorisés."
                )
            if node.attr not in _ALLOWED_NP_ATTRS:
                raise ValueError(
                    f"Attribut np interdit : np.{node.attr!r}. "
                    f"Fonctions autorisées : {sorted(_ALLOWED_NP_ATTRS)}"
                )

    return eval(  # noqa: S307 — sandboxé par l'AST walk ci-dessus
        compile(tree, "<formule>", "eval"),
        {"__builtins__": {}},
        env,
    )


# ===========================================================================
# [P0-THR] Wrapper QgsTask générique
# ===========================================================================

class _Task(QgsTask):
    """
    Encapsule une fonction de traitement dans un QgsTask QGIS.
    - work_fn(task) → result  — exécuté dans le thread de tâche
    - on_done(result)         — rappelé dans le thread UI (main)
    - on_error(msg)           — rappelé dans le thread UI en cas d'échec
    L'instance doit être stockée (self._task) pour éviter la GC prématurée.
    """

    def __init__(self, description: str, work_fn, on_done, on_error):
        super().__init__(description, QgsTask.CanCancel)
        self._work   = work_fn
        self._done   = on_done
        self._error  = on_error
        self.result  = None
        self._exc    = ""

    def run(self) -> bool:                          # thread de tâche
        try:
            self.result = self._work(self)
            return True
        except Exception as exc:
            self._exc = str(exc)
            _log(f"Erreur tâche '{self.description()}' : {exc}", Qgis.Critical)
            return False

    def finished(self, ok: bool):                   # thread UI
        if ok:
            self._done(self.result)
        else:
            self._error(self._exc)


# ===========================================================================
# Helpers communs
# ===========================================================================

def _log(msg: str, level=Qgis.Info):
    QgsMessageLog.logMessage(str(msg), _LOG_TAG, level=level)


def _tmp(ext=".tif"):
    t = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    p = t.name
    t.close()
    _TMP_REGISTRY.append(p)   # [P2-TMP] enregistrement pour nettoyage dans unload()
    return p


def _browse_save(parent, edit, filt="GeoTIFF (*.tif)"):
    p, _ = QFileDialog.getSaveFileName(parent, "Fichier de sortie", "", filt)
    if p:
        edit.setText(p)


def _out_row(parent, placeholder="Vide = temporaire", filt="GeoTIFF (*.tif)"):
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    btn = QPushButton("...")
    btn.setFixedWidth(32)
    btn.clicked.connect(lambda: _browse_save(parent, edit, filt))
    row = QHBoxLayout()
    row.addWidget(edit)
    row.addWidget(btn)
    return row, edit


def _load_raster(iface, path: str, name: str):
    """Charge un raster dans le projet QGIS — appel depuis thread UI uniquement."""
    rl = QgsRasterLayer(path, name)
    if rl.isValid():
        QgsProject.instance().addMapLayer(rl)
        iface.messageBar().pushMessage("Succès", f"Chargé : {name}",
                                       level=Qgis.Success, duration=4)
        _log(f"Raster chargé : {name}")
    else:
        QMessageBox.warning(None, "Erreur", f"Couche raster invalide :\n{path}")
        _log(f"Couche invalide : {path}", Qgis.Warning)


def _load_vector(iface, path: str, name: str):
    vl = QgsVectorLayer(path, name, "ogr")
    if vl.isValid():
        QgsProject.instance().addMapLayer(vl)
        iface.messageBar().pushMessage("Succès", name, level=Qgis.Success, duration=4)
        _log(f"Vecteur chargé : {name}")
    else:
        QMessageBox.warning(None, "Erreur", f"Couche vectorielle invalide :\n{path}")
        _log(f"Couche vecteur invalide : {path}", Qgis.Warning)


def _write_tif(array, ds_ref, path: str,
               dtype=gdal.GDT_Float32, nodata=-9999.0):
    """
    Écrit un array NumPy en GeoTIFF compressé (LZW + TILED).
    ds_ref : dataset GDAL source (géotransform + projection).
    """
    drv = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape
    ds = drv.Create(path, cols, rows, 1, dtype,
                    options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform(ds_ref.GetGeoTransform())
    ds.SetProjection(ds_ref.GetProjection())
    b = ds.GetRasterBand(1)
    b.WriteArray(array)
    b.SetNoDataValue(nodata)
    ds.FlushCache()
    ds = None


def _raster_combo_group(parent, title="Couche raster"):
    grp = QGroupBox(title)
    g = QVBoxLayout()
    combo = QgsMapLayerComboBox()
    combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
    g.addWidget(combo)
    grp.setLayout(g)
    return grp, combo


def _load_chk():
    chk = QCheckBox("Charger dans le projet")
    chk.setChecked(True)
    return chk


def _run_btn(label: str, callback):
    b = QPushButton(label)
    b.clicked.connect(callback)
    return b


def _nodata_mask(arr: np.ndarray, nodata) -> np.ndarray:
    """
    Masque booléen des pixels valides (True = valide).
    Gère nodata=None, NaN, ±Inf et comparaison float robuste.
    """
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= ~np.isclose(arr.astype(np.float64), float(nodata),
                            rtol=0, atol=1e-6)
    return mask


def _validate_bands(band_ids: list, ds) -> None:
    """Lève ValueError si band_ids contient des indices hors [1, RasterCount]."""
    n = ds.RasterCount
    bad = [b for b in band_ids if b < 1 or b > n]
    if bad:
        raise ValueError(
            f"Bande(s) hors plage : {bad}. "
            f"Ce raster possède {n} bande(s) (indices 1..{n})."
        )


def _safe_delete_ogr(drv, path: str):
    try:
        if os.path.exists(path):
            drv.DeleteDataSource(path)
    except Exception as exc:
        _log(f"Impossible de supprimer {path} : {exc}", Qgis.Warning)


def _submit(tab, task: _Task):
    """Enregistre la tâche sur le tab (anti-GC) et la soumet."""
    tab._active_task = task
    QgsApplication.taskManager().addTask(task)


def _epsg_to_wkt(epsg_str: str) -> str:
    """Convertit une chaîne EPSG en WKT. Lève ValueError si invalide."""
    try:
        epsg = int(epsg_str.strip())
    except ValueError:
        raise ValueError(f"Code EPSG invalide : {epsg_str!r} (entier attendu)")
    srs = osr.SpatialReference()
    if srs.ImportFromEPSG(epsg) != 0:
        raise ValueError(f"Code EPSG inconnu : {epsg}")
    return srs.ExportToWkt()


def _is_geographic(wkt: str) -> bool:
    """Retourne True si le CRS est géographique (degrés)."""
    if not wkt:
        return False
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return bool(srs.IsGeographic())


# ===========================================================================
# Plugin entry
# ===========================================================================

class RasterVectorizerPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self._tmp_files: list = []     # [P2-TMP] registre nettoyage
        self.dlg = None                # [UI] référence persistante — fenêtre non-modale

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = QIcon(icon_path)
        self.action = QAction(icon, "GeoRaster Tools", self.iface.mainWindow())
        self.action.setToolTip("GeoRaster Tools")
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("&GeoRaster Tools", self.action)

    def unload(self):
        self.iface.removePluginRasterMenu("&GeoRaster Tools", self.action)
        self.iface.removeToolBarIcon(self.action)
        # [P2-TMP] Nettoyage du registre global de fichiers temporaires orphelins
        for f in list(_TMP_REGISTRY):
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass
        _TMP_REGISTRY.clear()

    def run(self):
        # [UI] Non-modal : show() au lieu de exec_() pour permettre la réduction.
        # La référence self.dlg évite le garbage-collect immédiat.
        if self.dlg is None or not self.dlg.isVisible():
            self.dlg = MainDialog(self.iface)
            self.dlg.show()
        else:
            # Fenêtre déjà ouverte : ramener au premier plan
            self.dlg.showNormal()
            self.dlg.raise_()
            self.dlg.activateWindow()


# ===========================================================================
# Main dialog
# ===========================================================================

def _scrolled(widget: QWidget) -> QScrollArea:
    """
    Encapsule un onglet dans un QScrollArea sans frame.
    Permet de réduire librement la fenêtre principale :
    Qt calcule le minimumSizeHint() sur le viewport, non sur le contenu.
    """
    sa = QScrollArea()
    sa.setWidget(widget)
    sa.setWidgetResizable(True)
    sa.setFrameShape(QScrollArea.NoFrame)  # pas de bordure double
    return sa


class MainDialog(QDialog):
    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface = iface
        self.setWindowTitle("GeoRaster Tools")
        # [UI] Fenêtre indépendante avec tous les boutons de contrôle
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowTitleHint |
            Qt.WindowSystemMenuHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        # [UI] Taille initiale raisonnable — pas de minimum imposé
        #      setMinimumWidth() est supprimé : c'était lui qui bloquait
        #      le redimensionnement vers le bas.
        self.resize(600, 560)
        self.setSizeGripEnabled(True)   # triangle de redimensionnement

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        tabs = QTabWidget()
        # Chaque onglet est enveloppé dans _scrolled() :
        # la boîte de dialogue peut être réduite à volonté,
        # les barres de défilement apparaissent si nécessaire.
        tabs.addTab(_scrolled(TraitementsTab(iface)),   "🔄 Traitements")
        tabs.addTab(_scrolled(RasterCalcTab(iface)),    "🔢 Calcul")
        tabs.addTab(_scrolled(IndicesTab(iface)),       "📡 Indices")
        tabs.addTab(_scrolled(TerrainTab(iface)),       "⛰ Terrain")
        tabs.addTab(_scrolled(StatsTab(iface)),         "📊 Statistiques")
        tabs.addTab(_scrolled(UnsupervisedTab(iface)),  "🤖 Non supervisée")
        tabs.addTab(_scrolled(SupervisedTab(iface)),    "🌲 Random Forest")
        tabs.addTab(_scrolled(VectorisationTab(iface)), "🗺 Vectorisation")
        layout.addWidget(tabs)
        self.setLayout(layout)


# ===========================================================================
# Onglet 1 – Vectorisation
# ===========================================================================

class VectorisationTab(QWidget):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self._active_task = None
        L = QVBoxLayout()
        L.setSpacing(10)

        grp, self.layer_combo = _raster_combo_group(self, "Entrée")
        g = grp.layout()
        g.addWidget(QLabel("Bande (1 = première bande) :"))
        self.band_spin = QSpinBox()
        self.band_spin.setMinimum(1)
        self.band_spin.setValue(1)
        g.addWidget(self.band_spin)
        L.addWidget(grp)

        grp2 = QGroupBox("Mode")
        m = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Polygones (gdal.Polygonize)",
                                  "Lignes de contour (gdal.ContourGenerate)"])
        self.mode_combo.currentIndexChanged.connect(self._toggle)
        m.addWidget(self.mode_combo)
        self.cgrp = QGroupBox("Intervalle contour")
        ci = QVBoxLayout()
        self.interval = QSpinBox()
        self.interval.setRange(1, 100000)
        self.interval.setValue(10)
        ci.addWidget(self.interval)
        self.cgrp.setLayout(ci)
        self.cgrp.setVisible(False)
        m.addWidget(self.cgrp)
        grp2.setLayout(m)
        L.addWidget(grp2)

        grp3 = QGroupBox("Sortie")
        o = QVBoxLayout()
        row, self.out = _out_row(self, filt="Shapefile (*.shp);;GeoPackage (*.gpkg)")
        o.addLayout(row)
        self.load_chk = _load_chk()
        o.addWidget(self.load_chk)
        grp3.setLayout(o)
        L.addWidget(grp3)

        self.run_btn = _run_btn("Vectoriser", self._run)
        L.addStretch()
        L.addWidget(self.run_btn)
        self.setLayout(L)

    def _toggle(self, i):
        self.cgrp.setVisible(i == 1)

    def _run(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return QMessageBox.warning(self, "Erreur", "Aucun raster sélectionné.")

        src_path = layer.source()
        band_idx = self.band_spin.value()
        mode = self.mode_combo.currentIndex()
        interval = self.interval.value()
        out_path = self.out.text().strip()
        ext = os.path.splitext(out_path)[1].lower() if out_path else ".shp"
        if not out_path:
            out_path = _tmp(ext=".shp")
        drv_name = "GPKG" if ext == ".gpkg" else "ESRI Shapefile"
        load = self.load_chk.isChecked()
        layer_name = layer.name()

        self.run_btn.setEnabled(False)

        def work(task):
            ds = gdal.Open(src_path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Impossible d'ouvrir : {src_path}")
            # [P1-VALID]
            if band_idx < 1 or band_idx > ds.RasterCount:
                raise ValueError(f"Bande {band_idx} invalide (raster : {ds.RasterCount} bande(s)).")
            band = ds.GetRasterBand(band_idx)
            srs = osr.SpatialReference()
            wkt = ds.GetProjection()
            if wkt:
                srs.ImportFromWkt(wkt)
            drv = ogr.GetDriverByName(drv_name)
            _safe_delete_ogr(drv, out_path)
            ds_v = drv.CreateDataSource(out_path)
            if mode == 0:
                lyr = ds_v.CreateLayer("polygones", srs=srs, geom_type=ogr.wkbPolygon)
                lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger))
                gdal.Polygonize(band, None, lyr, 0, [], callback=None)
                name = f"Polygones_{layer_name}"
            else:
                lyr = ds_v.CreateLayer("contours", srs=srs, geom_type=ogr.wkbLineString)
                lyr.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
                lyr.CreateField(ogr.FieldDefn("ELEV", ogr.OFTReal))
                gdal.ContourGenerate(band, interval, 0, [], 0, 0, lyr, 0, 1)
                name = f"Contours_{layer_name}"
            ds_v = None
            ds = None
            _log(f"Vectorisation terminée → {out_path}")
            return {"path": out_path, "name": name}

        def on_done(res):
            self.run_btn.setEnabled(True)
            if load:
                _load_vector(self.iface, res["path"], res["name"])

        def on_error(msg):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Erreur vectorisation", msg)

        _submit(self, _Task("Vectorisation", work, on_done, on_error))


# ===========================================================================
# Onglet 2 – Indices spectraux
# ===========================================================================

class IndicesTab(QWidget):
    INDICES = {
        "NDVI  – Végétation":          (("(NIR-R)/(NIR+R)", ["Rouge", "PIR"]),),
        "NDWI  – Eau":                 (("(Vert-NIR)/(Vert+NIR)", ["Vert", "PIR"]),),
        "NDBI  – Bâti":                (("(MIR-NIR)/(MIR+NIR)", ["PIR", "MIR"]),),
        "EVI   – Végétation amél.":    (("2.5*(NIR-R)/(NIR+6R-7.5B+1)", ["Bleu", "Rouge", "PIR"]),),
        "SAVI  – Vég. sol ajusté":     (("1.5*(NIR-R)/(NIR+R+L)", ["Rouge", "PIR"]),),
    }
    # Format interne : clé → (formule_str, [bandes_nécessaires])
    _IDX = {
        "NDVI":  ("NDVI  – Végétation",   ["Rouge", "PIR"]),
        "NDWI":  ("NDWI  – Eau",          ["Vert",  "PIR"]),
        "NDBI":  ("NDBI  – Bâti",         ["PIR",   "MIR"]),
        "EVI":   ("EVI   – Végétation amél.", ["Bleu", "Rouge", "PIR"]),
        "SAVI":  ("SAVI  – Vég. sol ajusté",  ["Rouge", "PIR"]),
    }

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self._active_task = None
        L = QVBoxLayout()
        L.setSpacing(10)

        grp, self.layer_combo = _raster_combo_group(self, "Couche raster multi-bandes")
        L.addWidget(grp)

        grp2 = QGroupBox("Indice")
        g = QVBoxLayout()
        self.idx_combo = QComboBox()
        self.idx_combo.addItems([v[0] for v in self._IDX.values()])
        self.idx_combo.currentIndexChanged.connect(self._update_ui)
        g.addWidget(self.idx_combo)
        self.formula_lbl = QLabel()
        self.formula_lbl.setStyleSheet("color:grey; font-style:italic;")
        g.addWidget(self.formula_lbl)
        grp2.setLayout(g)
        L.addWidget(grp2)

        grp3 = QGroupBox("Numéro de bandes")
        g3 = QVBoxLayout()
        self.band_inputs = {}
        for name in ["Bleu", "Vert", "Rouge", "PIR", "MIR"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Bande {name} :"))
            sp = QSpinBox()
            sp.setRange(1, 20)
            sp.setValue(1)
            row.addWidget(sp)
            self.band_inputs[name] = sp
            g3.addLayout(row)
        grp3.setLayout(g3)
        L.addWidget(grp3)

        # [P2-UI] SAVI : paramètre L exposé
        self.savi_grp = QGroupBox("Paramètre SAVI")
        sv = QVBoxLayout()
        sv.addWidget(QLabel("Facteur L (0=sol nu → 1=couverture dense, défaut=0.5) :"))
        self.savi_L = QDoubleSpinBox()
        self.savi_L.setRange(0.0, 1.0)
        self.savi_L.setSingleStep(0.05)
        self.savi_L.setValue(0.5)
        sv.addWidget(self.savi_L)
        self.savi_grp.setLayout(sv)
        L.addWidget(self.savi_grp)

        # [P2-UI] Avertissement EVI
        self.evi_warn = QLabel(
            "⚠ EVI : formule valide uniquement pour des réflectances normalisées [0,1].\n"
            "   Diviser les bandes par le facteur d'échelle avant calcul (ex: /10000 pour Sentinel-2)."
        )
        self.evi_warn.setStyleSheet("color: orange;")
        self.evi_warn.setWordWrap(True)
        L.addWidget(self.evi_warn)

        grp4 = QGroupBox("Sortie")
        o = QVBoxLayout()
        row, self.out = _out_row(self)
        o.addLayout(row)
        self.load_chk = _load_chk()
        o.addWidget(self.load_chk)
        grp4.setLayout(o)
        L.addWidget(grp4)

        self.run_btn = _run_btn("Calculer l'indice", self._run)
        L.addStretch()
        L.addWidget(self.run_btn)
        self.setLayout(L)
        self._update_ui()

    def _current_key(self):
        display = self.idx_combo.currentText()
        for k, (label, _) in self._IDX.items():
            if label == display:
                return k
        return "NDVI"

    def _update_ui(self):
        key = self._current_key()
        _, needed = self._IDX[key]
        formulas = {
            "NDVI": "(PIR - R) / (PIR + R)",
            "NDWI": "(Vert - PIR) / (Vert + PIR)",
            "NDBI": "(MIR - PIR) / (MIR + PIR)",
            "EVI":  "2.5 * (PIR - R) / (PIR + 6R - 7.5B + 1)",
            "SAVI": "1.5 * (PIR - R) / (PIR + R + L)",
        }
        self.formula_lbl.setText(f"Formule : {formulas.get(key, '')}")
        for name, sp in self.band_inputs.items():
            sp.setEnabled(name in needed)
        self.savi_grp.setVisible(key == "SAVI")
        self.evi_warn.setVisible(key == "EVI")

    def _run(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return QMessageBox.warning(self, "Erreur", "Aucun raster sélectionné.")

        key = self._current_key()
        _, needed = self._IDX[key]
        band_map = {n: self.band_inputs[n].value() for n in needed}
        savi_L = self.savi_L.value()
        src_path = layer.source()
        out_path = self.out.text().strip() or _tmp()
        load = self.load_chk.isChecked()
        layer_name = layer.name()

        self.run_btn.setEnabled(False)

        def work(task):
            ds = gdal.Open(src_path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Impossible d'ouvrir : {src_path}")
            # [P1-VALID]
            _validate_bands(list(band_map.values()), ds)

            def rb(name):
                return ds.GetRasterBand(band_map[name]).ReadAsArray().astype(np.float32)

            eps = 1e-10

            if key == "NDVI":
                R, N = rb("Rouge"), rb("PIR")
                result = (N - R) / (N + R + eps)
            elif key == "NDWI":
                G, N = rb("Vert"), rb("PIR")
                result = (G - N) / (G + N + eps)
            elif key == "NDBI":
                N, M = rb("PIR"), rb("MIR")
                result = (M - N) / (M + N + eps)
            elif key == "EVI":
                B, R, N = rb("Bleu"), rb("Rouge"), rb("PIR")
                denom = N + 6.0 * R - 7.5 * B + 1.0
                # [MAJ-3] safe_denom évite la division réelle par zéro dans les deux
                # branches de np.where (NumPy évalue les deux avant de masquer)
                safe_denom = np.where(np.abs(denom) > eps, denom, 1.0)
                result = np.where(np.abs(denom) > eps,
                                  2.5 * (N - R) / safe_denom,
                                  np.nan)
            elif key == "SAVI":
                R, N = rb("Rouge"), rb("PIR")
                result = 1.5 * (N - R) / (N + R + savi_L + eps)
            else:
                raise ValueError(f"Indice inconnu : {key}")

            # Clip normalisé uniquement pour les indices borés [-1,1]
            if key in ("NDVI", "NDWI", "NDBI", "SAVI"):
                result = np.clip(result, -1.0, 1.0)

            result = result.astype(np.float32)
            _write_tif(result, ds, out_path)
            ds = None
            _log(f"Indice {key} calculé → {out_path}")
            return {"path": out_path, "name": f"{key}_{layer_name}"}

        def on_done(res):
            self.run_btn.setEnabled(True)
            if load:
                _load_raster(self.iface, res["path"], res["name"])

        def on_error(msg):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Erreur indice", msg)

        _submit(self, _Task(f"Calcul {key}", work, on_done, on_error))


# ===========================================================================
# Onglet 3 – Analyse de terrain (MNT)
# ===========================================================================

class TerrainTab(QWidget):
    MODES = ["Pente (slope)", "Orientation (aspect)", "Ombrage (hillshade)",
             "Indice TPI", "Indice TRI"]

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self._active_task = None
        L = QVBoxLayout()
        L.setSpacing(10)

        grp, self.layer_combo = _raster_combo_group(self, "Couche MNT / DEM")
        L.addWidget(grp)

        grp2 = QGroupBox("Analyse")
        g = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODES)
        self.mode_combo.currentIndexChanged.connect(self._toggle)
        g.addWidget(self.mode_combo)

        self.hill_grp = QGroupBox("Options hillshade")
        hi = QVBoxLayout()
        hi.addWidget(QLabel("Azimut (direction soleil, deg) :"))
        self.azimuth = QDoubleSpinBox()
        self.azimuth.setRange(0, 360)
        self.azimuth.setValue(315)
        hi.addWidget(self.azimuth)
        hi.addWidget(QLabel("Altitude soleil (deg) :"))
        self.altitude = QDoubleSpinBox()
        self.altitude.setRange(0, 90)
        self.altitude.setValue(45)
        hi.addWidget(self.altitude)
        # [P2-UI] Scale pour CRS géographiques
        self.scale_chk = QCheckBox("Appliquer facteur d'échelle (CRS géographique)")
        self.scale_chk.setToolTip(
            "Cochez si le CRS est en degrés (lat/lon) pour corriger\n"
            "le rapport horizontal/vertical (échelle ≈ 111 120 m/deg à l'équateur)."
        )
        hi.addWidget(self.scale_chk)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1, 1_000_000)
        self.scale_spin.setValue(111120)
        self.scale_spin.setEnabled(False)
        hi.addWidget(QLabel("Valeur du facteur d'échelle (m/unité) :"))
        hi.addWidget(self.scale_spin)
        self.scale_chk.toggled.connect(self.scale_spin.setEnabled)
        self.hill_grp.setLayout(hi)
        self.hill_grp.setVisible(False)
        g.addWidget(self.hill_grp)
        grp2.setLayout(g)
        L.addWidget(grp2)

        grp3 = QGroupBox("Sortie")
        o = QVBoxLayout()
        row, self.out = _out_row(self)
        o.addLayout(row)
        self.load_chk = _load_chk()
        o.addWidget(self.load_chk)
        grp3.setLayout(o)
        L.addWidget(grp3)

        self.run_btn = _run_btn("Calculer", self._run)
        L.addStretch()
        L.addWidget(self.run_btn)
        self.setLayout(L)

    def _toggle(self, i):
        self.hill_grp.setVisible(i == 2)
        # Auto-détection CRS géographique
        if i == 2:
            layer = self.layer_combo.currentLayer()
            if layer:
                ds = gdal.Open(layer.source(), gdal.GA_ReadOnly)
                if ds and _is_geographic(ds.GetProjection()):
                    self.scale_chk.setChecked(True)
                ds = None

    def _run(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return QMessageBox.warning(self, "Erreur", "Aucun raster sélectionné.")

        mode = self.mode_combo.currentIndex()
        src_path = layer.source()
        out_path = self.out.text().strip() or _tmp()
        load = self.load_chk.isChecked()
        az = self.azimuth.value()
        alt = self.altitude.value()
        use_scale = self.scale_chk.isChecked()
        scale_val = self.scale_spin.value()
        layer_name = layer.name()
        self.run_btn.setEnabled(False)

        def work(task):
            names = ["Pente", "Aspect", "Hillshade", "TPI", "TRI"]
            gdal_ops = ["slope", "aspect", "hillshade", "TPI", "TRI"]
            op = gdal_ops[mode]
            name = names[mode]
            if mode == 2:
                kw = {"azimuth": az, "altitude": alt}
                # [P2-UI] Facteur d'échelle hillshade pour CRS géographique
                if use_scale:
                    kw["scale"] = scale_val
                    _log(f"Hillshade : facteur d'échelle appliqué = {scale_val}")
                opts = gdal.DEMProcessingOptions(**kw)
                gdal.DEMProcessing(out_path, src_path, op, options=opts)
            else:
                gdal.DEMProcessing(out_path, src_path, op)
            _log(f"Terrain {name} calculé → {out_path}")
            return {"path": out_path, "name": f"{name}_{layer_name}"}

        def on_done(res):
            self.run_btn.setEnabled(True)
            if load:
                _load_raster(self.iface, res["path"], res["name"])

        def on_error(msg):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Erreur terrain", msg)

        _submit(self, _Task("Analyse terrain", work, on_done, on_error))


# ===========================================================================
# Onglet 4 – Calcul raster (SÉCURISÉ)
# ===========================================================================

class RasterCalcTab(QWidget):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self._active_task = None
        L = QVBoxLayout()
        L.setSpacing(10)

        grp, self.layer_combo = _raster_combo_group(self, "Couche raster (A)")
        L.addWidget(grp)

        grp2 = QGroupBox("Formule  —  utiliser A[b] pour la bande b")
        g = QVBoxLayout()
        g.addWidget(QLabel(
            "Exemples :\n"
            "  A[1] * 2                       → multiplier la bande 1\n"
            "  np.where(A[1] > 500, 1, 0)     → masque binaire\n"
            "  (A[4] - A[3]) / (A[4] + A[3])  → NDVI bandes 4/3\n"
            "\n"
            # [P0-SEC]
            "Sécurité : seules les opérations arithmétiques et un sous-ensemble\n"
            "de np.* (where, clip, sqrt, log, exp, abs, isfinite, …) sont autorisés."
        ))
        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("Ex: A[1] * 2")
        g.addWidget(self.formula_edit)
        grp2.setLayout(g)
        L.addWidget(grp2)

        grp3 = QGroupBox("Sortie")
        o = QVBoxLayout()
        row, self.out = _out_row(self)
        o.addLayout(row)
        self.load_chk = _load_chk()
        o.addWidget(self.load_chk)
        grp3.setLayout(o)
        L.addWidget(grp3)

        self.run_btn = _run_btn("Calculer", self._run)
        L.addStretch()
        L.addWidget(self.run_btn)
        self.setLayout(L)

    def _run(self):
        layer = self.layer_combo.currentLayer()
        formula = self.formula_edit.text().strip()
        if not layer:
            return QMessageBox.warning(self, "Erreur", "Aucun raster sélectionné.")
        if not formula:
            return QMessageBox.warning(self, "Erreur", "Formule vide.")

        src_path = layer.source()
        out_path = self.out.text().strip() or _tmp()
        load = self.load_chk.isChecked()
        layer_name = layer.name()
        self.run_btn.setEnabled(False)

        def work(task):
            ds = gdal.Open(src_path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Impossible d'ouvrir : {src_path}")

            class _BandProxy:
                """Proxy A[b] → ndarray float32 avec validation."""
                def __init__(self, ds_):
                    self._ds = ds_

                def __getitem__(self, b):
                    # [MAJ-1] numbers.Integral couvre int natif + np.int32/int64/intp
                    if not isinstance(b, numbers.Integral) or b < 1 or b > self._ds.RasterCount:
                        raise ValueError(
                            f"A[{b}] : indice invalide. "
                            f"Ce raster a {self._ds.RasterCount} bande(s)."
                        )
                    return self._ds.GetRasterBand(b).ReadAsArray().astype(np.float32)

            A = _BandProxy(ds)
            env = {"A": A, "np": np}

            # [P0-SEC] Évaluation via AST whitelist (pas de eval direct)
            result = _safe_eval(formula, env)
            result = np.asarray(result, dtype=np.float32)

            if result.ndim != 2:
                raise ValueError(
                    f"La formule doit produire un array 2D (obtenu : {result.ndim}D)."
                )
            _write_tif(result, ds, out_path)
            ds = None
            _log(f"Calcul raster terminé → {out_path}")
            return {"path": out_path, "name": f"Calcul_{layer_name}"}

        def on_done(res):
            self.run_btn.setEnabled(True)
            if load:
                _load_raster(self.iface, res["path"], res["name"])

        def on_error(msg):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Erreur calcul", msg)

        _submit(self, _Task("Calcul raster", work, on_done, on_error))


# ===========================================================================
# Onglet 5 – Statistiques raster
# ===========================================================================

class StatsTab(QWidget):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self._active_task = None
        L = QVBoxLayout()
        L.setSpacing(10)

        grp, self.layer_combo = _raster_combo_group(self, "Couche raster")
        L.addWidget(grp)

        grp2 = QGroupBox("Bande")
        g = QVBoxLayout()
        self.band_spin = QSpinBox()
        self.band_spin.setRange(1, 99)
        self.band_spin.setValue(1)
        g.addWidget(QLabel("Numéro de bande :"))
        g.addWidget(self.band_spin)
        grp2.setLayout(g)
        L.addWidget(grp2)

        grp3 = QGroupBox("Couche de zones (optionnelle — stats par zone)")
        gz = QVBoxLayout()
        self.zone_combo = QgsMapLayerComboBox()
        self.zone_combo.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.zone_combo.setAllowEmptyLayer(True)
        gz.addWidget(QLabel("Couche polygones (vide = stats globales) :"))
        gz.addWidget(self.zone_combo)
        self.zone_field = QLineEdit()
        self.zone_field.setPlaceholderText("Champ identifiant zone (ex: id, nom)")
        gz.addWidget(self.zone_field)
        # [P0-STAT] Note sur la méthode zonale
        note = QLabel("ℹ Masque géométrique exact utilisé (pas de MBR approximatif).")
        note.setStyleSheet("color: #0066cc; font-style: italic;")
        gz.addWidget(note)
        grp3.setLayout(gz)
        L.addWidget(grp3)

        grp4 = QGroupBox("Résultats")
        g4 = QVBoxLayout()
        self.results_txt = QTextEdit()
        self.results_txt.setReadOnly(True)
        self.results_txt.setMinimumHeight(180)
        g4.addWidget(self.results_txt)
        grp4.setLayout(g4)
        L.addWidget(grp4)

        self.run_btn = _run_btn("Calculer statistiques", self._run)
        L.addStretch()
        L.addWidget(self.run_btn)
        self.setLayout(L)

    def _run(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return QMessageBox.warning(self, "Erreur", "Aucun raster sélectionné.")

        src_path = layer.source()
        band_idx = self.band_spin.value()
        zone_layer = self.zone_combo.currentLayer()
        zone_path = zone_layer.source() if zone_layer else None
        zone_field = self.zone_field.text().strip()
        layer_name = layer.name()
        self.run_btn.setEnabled(False)

        def work(task):
            ds = gdal.Open(src_path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Impossible d'ouvrir : {src_path}")
            # [P1-VALID]
            if band_idx < 1 or band_idx > ds.RasterCount:
                raise ValueError(f"Bande {band_idx} invalide (max={ds.RasterCount}).")

            band = ds.GetRasterBand(band_idx)
            nodata = band.GetNoDataValue()
            arr = band.ReadAsArray().astype(np.float64)
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            rows, cols = arr.shape

            # [MAJ-4] Avertissement si aucune valeur nodata déclarée en mode zones :
            # les pixels hors emprise vecteur avec valeur réelle seront inclus dans
            # les stats si le raster n'a pas de bordure claire.
            nodata_warn = ""
            if nodata is None and zone_path:
                nodata_warn = (
                    "⚠ Aucune valeur nodata déclarée sur ce raster. "
                    "Les pixels hors zones peuvent fausser les statistiques "
                    "si le raster ne masque pas ses bordures.\n\n"
                )
                _log("Stats zonales : nodata=None — risque d'inclusion de pixels hors zone.",
                     Qgis.Warning)

            def _stats(data: np.ndarray, label: str) -> str:
                mask = _nodata_mask(data, nodata)
                d = data[mask]
                if d.size == 0:
                    return f"{label} : aucune donnée valide\n"
                return (
                    f"{label} :\n"
                    f"  Min      : {d.min():.6f}\n"
                    f"  Max      : {d.max():.6f}\n"
                    f"  Moyenne  : {d.mean():.6f}\n"
                    f"  Std      : {d.std(ddof=0):.6f}\n"
                    f"  Médiane  : {float(np.median(d)):.6f}\n"
                    f"  P5/P95   : {float(np.percentile(d, 5)):.6f} / "
                    f"{float(np.percentile(d, 95)):.6f}\n"
                    f"  Pixels   : {d.size}\n"
                )

            lines = [f"=== Statistiques – {layer_name} – Bande {band_idx} ===\n"]
            if nodata_warn:
                lines.append(nodata_warn)

            if zone_path:
                # [P0-STAT] Masque géométrique réel par rasterisation feature-par-feature
                ds_vec = ogr.Open(zone_path)
                if ds_vec is None:
                    raise RuntimeError(f"Impossible d'ouvrir couche zone : {zone_path}")
                lyr = ds_vec.GetLayer()

                mem_drv = gdal.GetDriverByName("MEM")
                ogr_mem_drv = ogr.GetDriverByName("Memory")
                srs_r = osr.SpatialReference()
                if proj:
                    srs_r.ImportFromWkt(proj)

                for feat in lyr:
                    geom = feat.GetGeometryRef()
                    if geom is None:
                        continue
                    # Rasteriser cette unique feature en masque binaire
                    ds_mask = mem_drv.Create("", cols, rows, 1, gdal.GDT_Byte)
                    ds_mask.SetGeoTransform(gt)
                    ds_mask.SetProjection(proj)
                    band_mask = ds_mask.GetRasterBand(1)
                    band_mask.Fill(0)
                    band_mask.SetNoDataValue(0)

                    mem_src = ogr_mem_drv.CreateDataSource("")
                    tmp_lyr = mem_src.CreateLayer("f", srs=srs_r,
                                                  geom_type=geom.GetGeometryType())
                    tmp_feat = ogr.Feature(tmp_lyr.GetLayerDefn())
                    tmp_feat.SetGeometry(geom.Clone())
                    tmp_lyr.CreateFeature(tmp_feat)

                    gdal.RasterizeLayer(ds_mask, [1], tmp_lyr, burn_values=[1])
                    mask = ds_mask.GetRasterBand(1).ReadAsArray().astype(bool)
                    ds_mask = None
                    mem_src = None

                    # Identifiant de zone
                    if zone_field:
                        try:
                            zone_id = str(feat.GetField(zone_field))
                        except Exception:
                            zone_id = str(feat.GetFID())
                    else:
                        zone_id = str(feat.GetFID())

                    sub = arr.copy()
                    sub[~mask] = np.nan
                    lines.append(_stats(sub, f"Zone '{zone_id}'"))

                ds_vec = None
            else:
                lines.append(_stats(arr, "Image entière"))

            ds = None
            return "\n".join(lines)

        def on_done(text):
            self.run_btn.setEnabled(True)
            self.results_txt.setPlainText(text)

        def on_error(msg):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Erreur statistiques", msg)

        _submit(self, _Task("Statistiques raster", work, on_done, on_error))


# ===========================================================================
# Onglet 6 – Traitements courants
# ===========================================================================

class TraitementsTab(QWidget):
    MODES = ["Reprojection", "Rééchan. (resample)", "Découpage (clip)",
             "Fusion (mosaïque)", "Raster → Points"]

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self._active_task = None
        L = QVBoxLayout()
        L.setSpacing(10)

        grp2 = QGroupBox("Traitement")
        g = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODES)
        self.mode_combo.currentIndexChanged.connect(self._toggle)
        g.addWidget(self.mode_combo)
        grp2.setLayout(g)
        L.addWidget(grp2)

        # Reprojection
        self.reproj_grp = QGroupBox("Reprojection")
        rp = QVBoxLayout()
        rp.addWidget(QLabel("Raster source :"))
        self.reproj_combo = QgsMapLayerComboBox()
        self.reproj_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        rp.addWidget(self.reproj_combo)
        rp.addWidget(QLabel("EPSG cible (ex: 32630) :"))
        self.epsg_edit = QLineEdit()
        self.epsg_edit.setPlaceholderText("32630")
        rp.addWidget(self.epsg_edit)
        self.reproj_grp.setLayout(rp)
        L.addWidget(self.reproj_grp)

        # Rééchantillonnage
        self.resamp_grp = QGroupBox("Rééchan. résolution")
        rs = QVBoxLayout()
        rs.addWidget(QLabel("Raster source :"))
        self.resamp_combo = QgsMapLayerComboBox()
        self.resamp_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        rs.addWidget(self.resamp_combo)
        rs.addWidget(QLabel("Nouvelle résolution (mètres ou degrés) :"))
        self.res_spin = QDoubleSpinBox()
        self.res_spin.setRange(0.00001, 100000)
        self.res_spin.setValue(10)
        rs.addWidget(self.res_spin)
        self.resamp_grp.setLayout(rs)
        L.addWidget(self.resamp_grp)

        # Découpage
        self.clip_grp = QGroupBox("Découpage")
        cl = QVBoxLayout()
        cl.addWidget(QLabel("Raster à découper :"))
        self.clip_raster_combo = QgsMapLayerComboBox()
        self.clip_raster_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        cl.addWidget(self.clip_raster_combo)
        cl.addWidget(QLabel("Couche de découpage (polygone) :"))
        self.clip_vec_combo = QgsMapLayerComboBox()
        self.clip_vec_combo.setFilters(QgsMapLayerProxyModel.VectorLayer)
        cl.addWidget(self.clip_vec_combo)
        self.clip_grp.setLayout(cl)
        L.addWidget(self.clip_grp)

        # Mosaïque
        self.mosaic_grp = QGroupBox("Mosaïque")
        mo = QVBoxLayout()
        mo.addWidget(QLabel("Fichiers raster à fusionner :"))
        self.mosaic_edit = QTextEdit()
        self.mosaic_edit.setMaximumHeight(80)
        self.mosaic_edit.setPlaceholderText("Un chemin par ligne")
        mo.addWidget(self.mosaic_edit)
        btn_add = QPushButton("Ajouter des fichiers...")
        btn_add.clicked.connect(self._add_mosaic_files)
        mo.addWidget(btn_add)
        self.mosaic_grp.setLayout(mo)
        L.addWidget(self.mosaic_grp)

        # Raster → Points
        self.pts_grp = QGroupBox("Raster → Points")
        pt = QVBoxLayout()
        pt.addWidget(QLabel("Raster source :"))
        self.pts_combo = QgsMapLayerComboBox()
        self.pts_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        pt.addWidget(self.pts_combo)
        pt.addWidget(QLabel("Bande :"))
        self.pts_band = QSpinBox()
        self.pts_band.setRange(1, 99)
        self.pts_band.setValue(1)
        pt.addWidget(self.pts_band)
        # [P1-PERF] Avertissement pour grands rasters
        warn_pts = QLabel(
            "⚠ Pour les rasters > 10 M pixels, préférer gdal_translate -of CSV "
            "ou gdal2xyz.py en ligne de commande."
        )
        warn_pts.setWordWrap(True)
        warn_pts.setStyleSheet("color: orange;")
        pt.addWidget(warn_pts)
        self.pts_grp.setLayout(pt)
        L.addWidget(self.pts_grp)

        # Sortie commune
        grp_out = QGroupBox("Sortie")
        o = QVBoxLayout()
        row, self.out = _out_row(self,
                                  filt="GeoTIFF (*.tif);;Shapefile (*.shp);;GeoPackage (*.gpkg)")
        o.addLayout(row)
        self.load_chk = _load_chk()
        o.addWidget(self.load_chk)
        grp_out.setLayout(o)
        L.addWidget(grp_out)

        self.run_btn = _run_btn("Exécuter", self._run)
        L.addStretch()
        L.addWidget(self.run_btn)
        self.setLayout(L)
        self._toggle(0)

    def _toggle(self, i):
        panels = [self.reproj_grp, self.resamp_grp, self.clip_grp,
                  self.mosaic_grp, self.pts_grp]
        for j, p in enumerate(panels):
            p.setVisible(j == i)

    def _add_mosaic_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choisir rasters", "", "Rasters (*.tif *.img *.vrt)")
        if files:
            existing = self.mosaic_edit.toPlainText().strip()
            self.mosaic_edit.setPlainText(
                (existing + "\n" if existing else "") + "\n".join(files))

    def _run(self):
        mode = self.mode_combo.currentIndex()
        out_raw = self.out.text().strip()
        load = self.load_chk.isChecked()
        self.run_btn.setEnabled(False)

        try:
            if mode == 0:      # Reprojection
                layer = self.reproj_combo.currentLayer()
                if not layer: raise RuntimeError("Aucun raster sélectionné.")
                # [P2-VALID] Validation EPSG avant lancement
                wkt = _epsg_to_wkt(self.epsg_edit.text() or "32630")
                epsg_val = int((self.epsg_edit.text() or "32630").strip())
                src_path = layer.source()
                out_path = out_raw or _tmp()
                layer_name = layer.name()

                def work_reproj(task):
                    gdal.Warp(out_path, src_path, dstSRS=wkt, dstNodata=-9999)
                    _log(f"Reprojection EPSG:{epsg_val} → {out_path}")
                    return {"path": out_path, "name": f"Reproj_EPSG{epsg_val}_{layer_name}",
                            "raster": True}
                work_fn = work_reproj

            elif mode == 1:    # Rééchantillonnage
                layer = self.resamp_combo.currentLayer()
                if not layer: raise RuntimeError("Aucun raster sélectionné.")
                src_path = layer.source()
                res = self.res_spin.value()
                out_path = out_raw or _tmp()
                layer_name = layer.name()

                def work_resamp(task):
                    gdal.Warp(out_path, src_path, xRes=res, yRes=res,
                              resampleAlg=gdal.GRA_Bilinear)
                    _log(f"Rééchantillonnage {res}m → {out_path}")
                    return {"path": out_path, "name": f"Resamp_{res}m_{layer_name}",
                            "raster": True}
                work_fn = work_resamp

            elif mode == 2:    # Découpage
                rl = self.clip_raster_combo.currentLayer()
                vl = self.clip_vec_combo.currentLayer()
                if not rl or not vl:
                    raise RuntimeError("Raster ou couche polygone manquant.")
                rpath, vpath = rl.source(), vl.source()
                out_path = out_raw or _tmp()
                layer_name = rl.name()

                def work_clip(task):
                    # [MAJ-2] cutlineSRS explicite : évite un clip silencieusement incorrect
                    # si le CRS de la couche vecteur diffère du raster
                    _ds_ref = gdal.Open(rpath, gdal.GA_ReadOnly)
                    _proj = _ds_ref.GetProjection() if _ds_ref else ""
                    _ds_ref = None
                    gdal.Warp(out_path, rpath,
                              cutlineDSName=vpath,
                              cropToCutline=True,
                              cutlineSRS=_proj or None,
                              dstNodata=-9999)
                    _log(f"Découpage → {out_path}")
                    return {"path": out_path, "name": f"Clip_{layer_name}",
                            "raster": True}
                work_fn = work_clip

            elif mode == 3:    # Mosaïque
                files = [f.strip() for f in
                         self.mosaic_edit.toPlainText().splitlines() if f.strip()]
                if len(files) < 2:
                    raise RuntimeError("Au moins 2 fichiers requis.")
                out_path = out_raw or _tmp()

                def work_mosaic(task):
                    vrt_path = _tmp(ext=".vrt")
                    vrt = gdal.BuildVRT(vrt_path, files)
                    if vrt is None:
                        raise RuntimeError("Échec BuildVRT.")
                    # [P2-GDAL] FlushCache avant déréférencement
                    vrt.FlushCache()
                    gdal.Translate(out_path, vrt)
                    vrt = None
                    try:
                        os.remove(vrt_path)
                    except OSError:
                        pass
                    _log(f"Mosaïque {len(files)} fichiers → {out_path}")
                    return {"path": out_path, "name": "Mosaïque", "raster": True}
                work_fn = work_mosaic

            else:              # Raster → Points
                layer = self.pts_combo.currentLayer()
                if not layer: raise RuntimeError("Aucun raster sélectionné.")
                src_path = layer.source()
                band_idx = self.pts_band.value()
                ext_out = os.path.splitext(out_raw)[1].lower() if out_raw else ".shp"
                out_path = out_raw or _tmp(ext=".shp")
                drv_name = "GPKG" if ext_out == ".gpkg" else "ESRI Shapefile"
                layer_name = layer.name()

                def work_pts(task):
                    ds = gdal.Open(src_path, gdal.GA_ReadOnly)
                    if ds is None:
                        raise RuntimeError(f"Impossible d'ouvrir : {src_path}")
                    # [P1-VALID]
                    if band_idx < 1 or band_idx > ds.RasterCount:
                        raise ValueError(f"Bande {band_idx} invalide (max={ds.RasterCount}).")

                    gt = ds.GetGeoTransform()
                    proj = ds.GetProjection()
                    b = ds.GetRasterBand(band_idx)
                    nodata = b.GetNoDataValue()
                    arr = b.ReadAsArray()
                    ds = None

                    # [P1-PERF] Vectorisation NumPy — O(n) au lieu de O(n²) Python
                    rows_idx, cols_idx = np.where(_nodata_mask(arr.astype(np.float64), nodata))
                    # gt = [x_orig, px_w, rot_x, y_orig, rot_y, px_h]
                    xs = gt[0] + cols_idx * gt[1] + rows_idx * gt[2]
                    ys = gt[3] + cols_idx * gt[4] + rows_idx * gt[5]
                    vals = arr[rows_idx, cols_idx].astype(np.float64)

                    srs = osr.SpatialReference()
                    if proj:
                        srs.ImportFromWkt(proj)
                    drv = ogr.GetDriverByName(drv_name)
                    _safe_delete_ogr(drv, out_path)
                    ds_v = drv.CreateDataSource(out_path)
                    lyr = ds_v.CreateLayer("points", srs=srs, geom_type=ogr.wkbPoint)
                    lyr.CreateField(ogr.FieldDefn("valeur", ogr.OFTReal))
                    lyr_defn = lyr.GetLayerDefn()

                    lyr.StartTransaction()
                    for x, y, v in zip(xs, ys, vals):
                        feat = ogr.Feature(lyr_defn)
                        pt = ogr.Geometry(ogr.wkbPoint)
                        pt.AddPoint(float(x), float(y))
                        feat.SetGeometry(pt)
                        feat.SetField("valeur", float(v))
                        lyr.CreateFeature(feat)
                    lyr.CommitTransaction()
                    ds_v = None
                    _log(f"Raster→Points : {len(xs)} points → {out_path}")
                    return {"path": out_path, "name": f"Points_{layer_name}",
                            "raster": False}
                work_fn = work_pts

        except (RuntimeError, ValueError) as exc:
            self.run_btn.setEnabled(True)
            QMessageBox.warning(self, "Erreur paramètres", str(exc))
            return

        def on_done(res):
            self.run_btn.setEnabled(True)
            if load:
                if res["raster"]:
                    _load_raster(self.iface, res["path"], res["name"])
                else:
                    _load_vector(self.iface, res["path"], res["name"])

        def on_error(msg):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Erreur traitement", msg)

        _submit(self, _Task("Traitement raster", work_fn, on_done, on_error))


# ===========================================================================
# Onglet 7 – Classification non supervisée (K-Means)
# ===========================================================================

class UnsupervisedTab(QWidget):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self._active_task = None
        L = QVBoxLayout()
        L.setSpacing(10)

        grp, self.layer_combo = _raster_combo_group(self, "Couche raster")
        g = grp.layout()
        g.addWidget(QLabel("Bandes (ex: 1,2,3 — vide = toutes) :"))
        self.bands_edit = QLineEdit()
        self.bands_edit.setPlaceholderText("1,2,3  ou vide")
        g.addWidget(self.bands_edit)
        L.addWidget(grp)

        grp2 = QGroupBox("Paramètres K-Means")
        p2 = QVBoxLayout()
        p2.addWidget(QLabel("Nombre de classes (K) :"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 50)
        self.k_spin.setValue(5)
        p2.addWidget(self.k_spin)
        p2.addWidget(QLabel("Itérations max :"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 1000)
        self.iter_spin.setValue(300)
        p2.addWidget(self.iter_spin)
        # [P1-ML] StandardScaler exposé
        self.scale_chk = QCheckBox("Normaliser les bandes (StandardScaler) — recommandé")
        self.scale_chk.setChecked(True)
        self.scale_chk.setToolTip(
            "Sans normalisation, les bandes à grande amplitude\n"
            "dominent la distance euclidienne du K-Means."
        )
        p2.addWidget(self.scale_chk)
        # [P1-ML] Score de silhouette optionnel
        self.silhouette_chk = QCheckBox("Calculer score de silhouette (lent sur grands rasters)")
        self.silhouette_chk.setChecked(False)
        p2.addWidget(self.silhouette_chk)
        grp2.setLayout(p2)
        L.addWidget(grp2)

        grp3 = QGroupBox("Sortie")
        o = QVBoxLayout()
        row, self.out = _out_row(self)
        o.addLayout(row)
        self.load_chk = _load_chk()
        o.addWidget(self.load_chk)
        grp3.setLayout(o)
        L.addWidget(grp3)

        self.run_btn = _run_btn("Classer (K-Means)", self._run)
        L.addStretch()
        L.addWidget(self.run_btn)
        self.setLayout(L)

    def _run(self):
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return QMessageBox.critical(
                self, "Dépendance manquante",
                "scikit-learn requis.\n\npip install scikit-learn"
            )

        layer = self.layer_combo.currentLayer()
        if not layer:
            return QMessageBox.warning(self, "Erreur", "Aucun raster sélectionné.")

        src_path = layer.source()
        layer_name = layer.name()
        txt = self.bands_edit.text().strip()
        k = self.k_spin.value()
        max_iter = self.iter_spin.value()
        do_scale = self.scale_chk.isChecked()
        do_silhouette = self.silhouette_chk.isChecked()
        out_path = self.out.text().strip() or _tmp()
        load = self.load_chk.isChecked()
        self.run_btn.setEnabled(False)

        def work(task):
            ds = gdal.Open(src_path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Impossible d'ouvrir : {src_path}")

            # [P1-VALID]
            if txt:
                band_ids = [int(x) for x in txt.split(",") if x.strip()]
                _validate_bands(band_ids, ds)
            else:
                band_ids = list(range(1, ds.RasterCount + 1))

            stack = np.stack(
                [ds.GetRasterBand(b).ReadAsArray().astype(np.float32) for b in band_ids],
                axis=-1,
            )
            rows, cols = stack.shape[:2]
            pixels = stack.reshape(-1, len(band_ids))

            # Masque validité : tous les canaux finis
            valid = np.all(np.isfinite(pixels), axis=1)
            X = pixels[valid]

            if X.shape[0] < k:
                raise ValueError(
                    f"Trop peu de pixels valides ({X.shape[0]}) pour {k} classes."
                )

            # [P1-ML] Normalisation StandardScaler
            if do_scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                _log("K-Means : StandardScaler appliqué.")

            km = KMeans(
                n_clusters=k,
                max_iter=max_iter,
                n_init=10,
                random_state=42,
            )
            labels = km.fit_predict(X).astype(np.int16)

            result = np.full(rows * cols, -9999, dtype=np.int16)
            result[valid] = labels
            _write_tif(result.reshape(rows, cols), ds, out_path,
                       dtype=gdal.GDT_Int16, nodata=-9999)
            ds = None

            info = {
                "inertia": float(km.inertia_),
                "iterations": int(km.n_iter_),
                "silhouette": None,
            }

            # [P1-ML] Score de silhouette optionnel (subsample pour performance)
            if do_silhouette:
                from sklearn.metrics import silhouette_score
                max_sil = 50_000
                if X.shape[0] > max_sil:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(X.shape[0], max_sil, replace=False)
                    sil = silhouette_score(X[idx], labels[idx])
                else:
                    sil = silhouette_score(X, labels)
                info["silhouette"] = float(sil)
                _log(f"K-Means : silhouette score = {sil:.4f}")

            _log(f"K-Means K={k} terminé — inertie={info['inertia']:.2f} → {out_path}")
            return {"path": out_path, "name": f"KMeans_K{k}_{layer_name}", "info": info}

        def on_done(res):
            self.run_btn.setEnabled(True)
            info = res["info"]
            msg = (
                f"K-Means terminé — K = {k}\n\n"
                f"Inertie    : {info['inertia']:.2f}\n"
                f"Itérations : {info['iterations']}\n"
            )
            if info["silhouette"] is not None:
                msg += f"Silhouette : {info['silhouette']:.4f}  ([-1,1], >0.5 = bonne séparation)\n"
            QMessageBox.information(self, "K-Means", msg)
            if load:
                _load_raster(self.iface, res["path"], res["name"])

        def on_error(msg):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Erreur K-Means", msg)

        _submit(self, _Task(f"K-Means K={k}", work, on_done, on_error))


# ===========================================================================
# Onglet 8 – Classification supervisée (Random Forest)
# ===========================================================================

class SupervisedTab(QWidget):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self._active_task = None
        L = QVBoxLayout()
        L.setSpacing(10)

        grp, self.raster_combo = _raster_combo_group(self, "Raster d'entrée (features)")
        g = grp.layout()
        g.addWidget(QLabel("Bandes (ex: 1,2,3 — vide = toutes) :"))
        self.bands_edit = QLineEdit()
        self.bands_edit.setPlaceholderText("1,2,3  ou vide")
        g.addWidget(self.bands_edit)
        L.addWidget(grp)

        grp2 = QGroupBox("Couche d'entraînement (polygones/points avec champ classe)")
        t = QVBoxLayout()
        self.train_combo = QgsMapLayerComboBox()
        self.train_combo.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.train_combo.layerChanged.connect(self._update_fields)
        t.addWidget(QLabel("Couche vectorielle :"))
        t.addWidget(self.train_combo)
        t.addWidget(QLabel("Champ classe :"))
        self.field_combo = QComboBox()
        t.addWidget(self.field_combo)
        grp2.setLayout(t)
        L.addWidget(grp2)
        self._update_fields()

        grp3 = QGroupBox("Paramètres Random Forest")
        r = QVBoxLayout()
        r.addWidget(QLabel("Nombre d'arbres :"))
        self.trees_spin = QSpinBox()
        self.trees_spin.setRange(10, 1000)
        self.trees_spin.setValue(100)
        r.addWidget(self.trees_spin)
        r.addWidget(QLabel("Profondeur max (0 = illimitée) :"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(0, 100)
        self.depth_spin.setValue(0)
        r.addWidget(self.depth_spin)
        grp3.setLayout(r)
        L.addWidget(grp3)

        grp_val = QGroupBox("Validation")
        v = QVBoxLayout()

        v.addWidget(QLabel("Proportion test holdout (0 = désactivé) :"))
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 0.5)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.25)
        self.test_spin.setToolTip(
            "25% des pixels étiquetés réservés pour l'évaluation.\n"
            "Métriques : OA, Kappa, matrice de confusion, F1 par classe."
        )
        v.addWidget(self.test_spin)

        self.kfold_chk = QCheckBox("Validation croisée StratifiedKFold (recommandé si peu de données)")
        self.kfold_chk.setChecked(False)
        self.kfold_chk.setToolTip(
            "Plus robuste qu'un holdout unique.\n"
            "Calcule OA et Kappa moyens ± écart-type sur k folds."
        )
        v.addWidget(self.kfold_chk)
        krow = QHBoxLayout()
        krow.addWidget(QLabel("   Nombre de folds (k) :"))
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(3, 10)
        self.kfold_spin.setValue(5)
        self.kfold_spin.setEnabled(False)
        krow.addWidget(self.kfold_spin)
        v.addLayout(krow)
        self.kfold_chk.toggled.connect(self.kfold_spin.setEnabled)

        self.roc_chk = QCheckBox("Calculer ROC AUC (macro-average, OvR multiclasse)")
        self.roc_chk.setChecked(True)
        v.addWidget(self.roc_chk)

        self.grid_chk = QCheckBox("Optimisation hyperparamètres (GridSearchCV — plus lent)")
        self.grid_chk.setChecked(False)
        self.grid_chk.setToolTip(
            "Explore : n_estimators [50,100,200], max_depth [None,10,20],\n"
            "max_features [sqrt, log2] via StratifiedKFold cv=3."
        )
        v.addWidget(self.grid_chk)

        self.csv_chk = QCheckBox("Exporter matrice de confusion en CSV")
        self.csv_chk.setChecked(True)
        v.addWidget(self.csv_chk)
        csv_row, self.csv_out = _out_row(self, placeholder="Chemin CSV (vide = temporaire)",
                                          filt="CSV (*.csv)")
        v.addLayout(csv_row)
        self.csv_chk.toggled.connect(self.csv_out.setEnabled)

        grp_val.setLayout(v)
        L.addWidget(grp_val)

        grp4 = QGroupBox("Sortie")
        o = QVBoxLayout()
        row, self.out = _out_row(self)
        o.addLayout(row)
        self.load_chk = _load_chk()
        o.addWidget(self.load_chk)
        grp4.setLayout(o)
        L.addWidget(grp4)

        self.run_btn = _run_btn("Classer (Random Forest)", self._run)
        L.addStretch()
        L.addWidget(self.run_btn)
        self.setLayout(L)

    def _update_fields(self):
        self.field_combo.clear()
        lyr = self.train_combo.currentLayer()
        if lyr:
            for f in lyr.fields():
                self.field_combo.addItem(f.name())

    def _run(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import (
                train_test_split, StratifiedKFold, cross_validate, GridSearchCV,
            )
            from sklearn.metrics import (
                accuracy_score, cohen_kappa_score, make_scorer,
                confusion_matrix, classification_report,
                roc_auc_score,
            )
        except ImportError:
            return QMessageBox.critical(
                self, "Dépendance manquante",
                "scikit-learn requis.\n\npip install scikit-learn"
            )

        rl = self.raster_combo.currentLayer()
        tl = self.train_combo.currentLayer()
        field = self.field_combo.currentText()

        if not rl:
            return QMessageBox.warning(self, "Erreur", "Aucun raster sélectionné.")
        if not tl:
            return QMessageBox.warning(self, "Erreur", "Aucune couche d'entraînement.")
        if not field:
            return QMessageBox.warning(self, "Erreur", "Aucun champ classe.")

        src_path    = rl.source()
        train_path  = tl.source()
        txt         = self.bands_edit.text().strip()
        n_trees     = self.trees_spin.value()
        max_depth   = self.depth_spin.value() or None
        test_size   = self.test_spin.value()
        do_kfold    = self.kfold_chk.isChecked()
        k_folds     = self.kfold_spin.value()
        do_roc      = self.roc_chk.isChecked()
        do_grid     = self.grid_chk.isChecked()
        do_csv      = self.csv_chk.isChecked()
        csv_path    = self.csv_out.text().strip() or _tmp(ext=".csv")
        out_path    = self.out.text().strip() or _tmp()
        load        = self.load_chk.isChecked()
        layer_name  = rl.name()
        self.run_btn.setEnabled(False)

        def work(task):
            ds = gdal.Open(src_path, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"Impossible d'ouvrir : {src_path}")

            rows, cols = ds.RasterYSize, ds.RasterXSize
            gt, proj = ds.GetGeoTransform(), ds.GetProjection()

            # [P1-VALID]
            if txt:
                band_ids = [int(x) for x in txt.split(",") if x.strip()]
                _validate_bands(band_ids, ds)
            else:
                band_ids = list(range(1, ds.RasterCount + 1))

            stack = np.stack(
                [ds.GetRasterBand(b).ReadAsArray().astype(np.float32) for b in band_ids],
                axis=-1,
            )  # shape (rows, cols, n_bands)

            # Rasterisation couche d'entraînement
            ds_ogr = ogr.Open(train_path)
            if ds_ogr is None:
                raise RuntimeError(f"Impossible d'ouvrir couche vecteur : {train_path}")
            ogr_lyr = ds_ogr.GetLayer()

            # [P1-ML] LabelEncoder reproductible (tri alphabétique des classes)
            le = LabelEncoder()
            raw_labels = []
            feats_geom = []
            for feat in ogr_lyr:
                val = feat.GetField(field)
                if val is None:
                    continue
                raw_labels.append(str(val))
                feats_geom.append(feat.GetGeometryRef().ExportToWkb())
            if not raw_labels:
                raise RuntimeError("Aucune feature avec valeur de classe valide.")
            le.fit(sorted(set(raw_labels)))   # tri déterministe

            # Reconstruction couche mémoire avec labels encodés
            srs_v = osr.SpatialReference()
            if proj:
                srs_v.ImportFromWkt(proj)
            ogr_mem = ogr.GetDriverByName("Memory").CreateDataSource("")
            tmp_lyr = ogr_mem.CreateLayer("t", srs=srs_v,
                                          geom_type=ogr_lyr.GetGeomType())
            tmp_lyr.CreateField(ogr.FieldDefn("C", ogr.OFTInteger))
            ogr_lyr.ResetReading()
            for feat, wkb in zip(ogr_lyr, feats_geom):
                val = feat.GetField(field)
                if val is None:
                    continue
                nf = ogr.Feature(tmp_lyr.GetLayerDefn())
                nf.SetGeometry(ogr.CreateGeometryFromWkb(wkb))
                nf.SetField("C", int(le.transform([str(val)])[0]))
                tmp_lyr.CreateFeature(nf)
            ds_ogr = None

            mem_drv = gdal.GetDriverByName("MEM")
            ds_tr = mem_drv.Create("", cols, rows, 1, gdal.GDT_Int32)
            ds_tr.SetGeoTransform(gt)
            ds_tr.SetProjection(proj)
            ds_tr.GetRasterBand(1).Fill(-9999)
            ds_tr.GetRasterBand(1).SetNoDataValue(-9999)
            gdal.RasterizeLayer(ds_tr, [1], tmp_lyr, options=["ATTRIBUTE=C"])
            train_arr = ds_tr.GetRasterBand(1).ReadAsArray()
            ds_tr = None

            flat_lbl = train_arr.ravel()
            flat_px  = stack.reshape(-1, len(band_ids))
            valid_px = np.all(np.isfinite(flat_px), axis=1)
            valid    = valid_px & (flat_lbl != -9999)
            X, y     = flat_px[valid], flat_lbl[valid]

            # ── Vérification minimum samples par classe ─────────────────────
            unique_cls, counts = np.unique(y, return_counts=True)
            n_classes = len(unique_cls)
            if n_classes < 2:
                raise RuntimeError(
                    f"Au moins 2 classes requises — seulement {n_classes} trouvée(s)."
                )
            min_samples = int(counts.min())
            min_class   = str(le.inverse_transform([int(unique_cls[counts.argmin()])])[0])
            if min_samples < 5:
                raise RuntimeError(
                    f"Classe '{min_class}' n'a que {min_samples} pixel(s) étiqueté(s). "
                    f"Minimum recommandé : 5 par classe (idéalement > 30)."
                )
            _log(f"RF : {n_classes} classes, min={min_samples} px (classe '{min_class}')")

            metrics_txt = ""
            imp_txt     = ""
            kfold_txt   = ""
            roc_txt     = ""
            cm_data     = None   # pour export CSV

            # ── GridSearchCV (optionnel) ─────────────────────────────────────
            base_rf = RandomForestClassifier(
                n_jobs=max(1, (os.cpu_count() or 2) // 2),
                random_state=42,
            )
            if do_grid:
                _log("RF : GridSearchCV en cours...")
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth":    [None, 10, 20],
                    "max_features": ["sqrt", "log2"],
                }
                gs = GridSearchCV(
                    base_rf, param_grid,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                    scoring="accuracy",
                    n_jobs=max(1, (os.cpu_count() or 2) // 2),
                    refit=True,
                )
                gs.fit(X, y)
                rf = gs.best_estimator_
                best_params = gs.best_params_
                _log(f"RF GridSearch best params : {best_params}  score={gs.best_score_:.4f}")
                metrics_txt += (
                    f"=== GridSearchCV (cv=3) ===\n"
                    f"Meilleurs paramètres : {best_params}\n"
                    f"Score CV moyen       : {gs.best_score_:.4f}\n\n"
                )
            else:
                rf = RandomForestClassifier(
                    n_estimators=n_trees,
                    max_depth=max_depth,
                    n_jobs=max(1, (os.cpu_count() or 2) // 2),
                    random_state=42,
                )

            # ── Validation croisée StratifiedKFold ──────────────────────────
            if do_kfold:
                _log(f"RF : StratifiedKFold cv={k_folds}...")
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                # [CRIT-1] "cohen_kappa" n'existe pas dans sklearn.metrics.SCORERS.
                # make_scorer() est obligatoire pour les métriques sans alias officiel.
                _kappa_scorer = make_scorer(cohen_kappa_score)
                cv_res = cross_validate(
                    rf, X, y, cv=skf,
                    scoring={"accuracy": "accuracy",
                             "kappa": _kappa_scorer},
                    return_train_score=False,
                )
                oa_scores  = cv_res["test_accuracy"]
                kap_scores = cv_res.get("test_kappa", np.array([float("nan")]))
                kfold_txt = (
                    f"=== Validation croisée StratifiedKFold (k={k_folds}) ===\n"
                    f"OA   : {oa_scores.mean():.4f} ± {oa_scores.std():.4f}  "
                    f"({oa_scores.min():.4f} – {oa_scores.max():.4f})\n"
                    f"Kappa: {kap_scores.mean():.4f} ± {kap_scores.std():.4f}\n\n"
                )
                _log(f"RF KFold OA={oa_scores.mean():.4f}±{oa_scores.std():.4f}")
                # Entraîner sur la totalité après cross-val
                rf.fit(X, y)
            elif not do_grid:
                # Entraînement standard avec holdout
                if test_size > 0:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                else:
                    X_tr, y_tr = X, y
                    X_te, y_te = None, None
                rf.fit(X_tr, y_tr)
            else:
                # GridSearch déjà fitté sur X entier
                X_te, y_te = None, None

            # ── Métriques holdout ────────────────────────────────────────────
            if test_size > 0 and not do_kfold and not do_grid:
                y_pred = rf.predict(X_te)
                oa  = accuracy_score(y_te, y_pred)
                kap = cohen_kappa_score(y_te, y_pred)
                cm  = confusion_matrix(y_te, y_pred)
                cm_data = cm
                class_labels = [str(le.inverse_transform([c])[0]) for c in np.unique(y_te)]
                cr = classification_report(
                    y_te, y_pred,
                    target_names=class_labels,
                    zero_division=0,
                )
                cm_lines = ["Matrice de confusion (lignes=réel, cols=prédit) :"]
                header = "         " + "  ".join(f"{c:>8}" for c in class_labels)
                cm_lines.append(header)
                for i, row_cm in enumerate(cm):
                    lbl = class_labels[i] if i < len(class_labels) else str(i)
                    cm_lines.append(f"  {lbl:>7}: " + "  ".join(f"{v:>8}" for v in row_cm))
                metrics_txt += (
                    f"=== Holdout (test_size={test_size:.0%}) ===\n\n"
                    f"Overall Accuracy (OA) : {oa:.4f}  ({oa*100:.2f} %)\n"
                    f"Kappa coefficient     : {kap:.4f}\n\n"
                    f"{cr}\n\n"
                    + "\n".join(cm_lines)
                )
                _log(f"RF holdout OA={oa:.4f} Kappa={kap:.4f}")

                # ── ROC AUC ──────────────────────────────────────────────────
                if do_roc:
                    try:
                        y_prob = rf.predict_proba(X_te)
                        if n_classes == 2:
                            auc = roc_auc_score(y_te, y_prob[:, 1])
                            roc_txt = f"ROC AUC (binaire)    : {auc:.4f}\n"
                        else:
                            auc = roc_auc_score(
                                y_te, y_prob,
                                multi_class="ovr", average="macro"
                            )
                            roc_txt = f"ROC AUC (macro OvR)  : {auc:.4f}\n"
                        metrics_txt = metrics_txt.replace(
                            f"Kappa coefficient     : {kap:.4f}",
                            f"Kappa coefficient     : {kap:.4f}\n{roc_txt.strip()}"
                        )
                        _log(f"RF ROC AUC={auc:.4f}")
                    except Exception as e_roc:
                        _log(f"ROC AUC non calculable : {e_roc}", Qgis.Warning)
            elif not metrics_txt:
                metrics_txt = "Aucune évaluation holdout (désactivé)."

            # ── Export CSV matrice de confusion ──────────────────────────────
            if do_csv and cm_data is not None:
                import csv as _csv
                with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                    w = _csv.writer(f_csv)
                    w.writerow([""] + class_labels)
                    for i, row_cm in enumerate(cm_data):
                        w.writerow([class_labels[i]] + list(row_cm))
                _log(f"Matrice de confusion exportée → {csv_path}")

            # ── Importance des variables ─────────────────────────────────────
            imp = rf.feature_importances_
            imp_lines = ["=== Importance des variables ===\n"]
            sorted_imp = sorted(enumerate(imp, start=1), key=lambda x: -x[1])
            for rank, (bid, score) in enumerate(sorted_imp, 1):
                real_band = band_ids[bid - 1]
                bar = "█" * int(score * 40)
                imp_lines.append(f"  #{rank:<2} Bande {real_band:>3} : {score:.4f}  {bar}")
            imp_txt = "\n".join(imp_lines)

            # ── Prédiction image entière ─────────────────────────────────────
            predicted_flat = np.full(rows * cols, -9999, dtype=np.int16)
            predicted_flat[valid_px] = rf.predict(flat_px[valid_px]).astype(np.int16)
            _write_tif(predicted_flat.reshape(rows, cols), ds, out_path,
                       dtype=gdal.GDT_Int16, nodata=-9999)
            ds = None

            # Table de correspondance classes
            class_map_txt = "=== Correspondance classes ===\n"
            for i, c in enumerate(le.classes_):
                class_map_txt += f"  Entier {i} → classe '{c}'\n"

            _log(f"RF terminé → {out_path}")
            return {
                "path":        out_path,
                "name":        f"RF_{layer_name}",
                "metrics":     metrics_txt,
                "kfold":       kfold_txt,
                "importances": imp_txt,
                "class_map":   class_map_txt,
                "csv_path":    csv_path if (do_csv and cm_data is not None) else None,
            }

        def on_done(res):
            self.run_btn.setEnabled(True)
            body = res["class_map"] + "\n"
            if res["kfold"]:
                body += res["kfold"]
            body += res["metrics"] + "\n\n" + res["importances"]
            if res["csv_path"]:
                body += f"\n\nMatrice CSV exportée → {res['csv_path']}"

            dlg = QDialog(self)
            dlg.setWindowTitle("Résultats Random Forest")
            dlg.setMinimumSize(700, 560)
            vl = QVBoxLayout(dlg)
            txt = QTextEdit()
            txt.setReadOnly(True)
            txt.setFontFamily("Courier")
            txt.setPlainText(body)
            vl.addWidget(txt)
            btn_ok = QPushButton("Fermer")
            btn_ok.clicked.connect(dlg.accept)
            vl.addWidget(btn_ok)
            dlg.exec_()
            if load:
                _load_raster(self.iface, res["path"], res["name"])

        def on_error(msg):
            self.run_btn.setEnabled(True)
            QMessageBox.critical(self, "Erreur Random Forest", msg)

        _submit(self, _Task("Random Forest", work, on_done, on_error))


# ===========================================================================
# CHANGELOG
# ===========================================================================
# v1.0.0 — Version initiale
# v1.1.0 — Corrections sécurité, threading, algorithmes :
#   [P0-SEC]  eval() → _safe_eval() AST-whitelisted (RasterCalcTab)
#   [P0-THR]  QgsTask pour tous les onglets — UI non bloquée
#   [P0-STAT] Stats zonales : masque géométrique exact par gdal.RasterizeLayer
#   [P1-ML]   K-Means : StandardScaler + silhouette score optionnel
#   [P1-ML]   RF : split train/test, OA, Kappa, confusion matrix,
#                  classification_report, feature_importances_
#   [P1-ML]   RF : LabelEncoder (encodage reproductible des classes string)
#   [P1-ML]   RF : n_jobs limité à cpu_count//2 (non-saturant)
#   [P1-PERF] Raster→Points : vectorisation NumPy + transactions OGR
#   [P1-VALID] _validate_bands() systématique
#   [P2-UI]   SAVI : paramètre L exposé
#   [P2-UI]   EVI : avertissement réflectances normalisées
#   [P2-UI]   EVI : masque dénominateur ≈ 0 (pas de clip aveugle)
#   [P2-UI]   Hillshade : scale auto + spinbox pour CRS géographique
#   [P2-GDAL] VRT : FlushCache() + nettoyage fichier temporaire
#   [P2-VALID] EPSG : _epsg_to_wkt() avec validation try/except
#   [P2-ND]   _nodata_mask() unifié, None-safe, isclose float-safe
#   [P2-LOG]  QgsMessageLog (tag "GeoRasterTools") sur toutes les opérations
#   [P2-TMP]  Registre _tmp_files + nettoyage dans unload()
#   [P2-IO]   GeoTIFF : COMPRESS=LZW + TILED=YES + BIGTIFF=IF_SAFER
# v1.1.1 — Corrections bugs identifiés par audit :
#   [CRIT-1]  RF KFold : "cohen_kappa" → make_scorer(cohen_kappa_score)
#   [CRIT-2]  _tmp() alimente désormais _TMP_REGISTRY ; unload() nettoie réellement
#   [CRIT-3]  Suppression nom auteur : tooltip barre d'outils + metadata.txt
#   [MAJ-1]   _BandProxy : isinstance(b, numbers.Integral) — accepte np.int32/64
#   [MAJ-2]   Découpage : cutlineSRS + dstNodata ajoutés à gdal.Warp
#   [MAJ-3]   EVI : safe_denom évite la division réelle par zéro (np.where)
#   [MAJ-4]   Stats zonales : avertissement UI si nodata=None
#   [MIN-2]   _safe_eval : validation que l'attribut porte sur np.* exclusivement
#   [MIN-3]   Reprojection : dstNodata=-9999 ajouté à gdal.Warp
# ===========================================================================
