import io, pickle, joblib, pandas as pd, streamlit as st, gcsfs, json, numpy as np

from google.oauth2 import service_account
from st_files_connection import FilesConnection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

st.set_page_config(
    page_title = "Accidentes MX",
    layout = "wide")

st.header(body = "Accidentes MX")

st.subheader(
    body = """
    Propósito. Estimar la gravedad del accidente (CLASACC) para un registro dado —“Sólo daños”, “No fatal”, “Fatal”— y ofrecer una lectura rápida del desempeño del modelo en 2023.
    """)

@st.cache_resource
def get_fs():
    # Lee y parsea el JSON de tu bloque [gcs]
    creds_info = json.loads(st.secrets["gcs"]["credentials"])
    # Crea credenciales a partir del JSON
    creds = service_account.Credentials.from_service_account_info(creds_info)
    # IMPORTANTE: añade scopes de GCS
    creds = creds.with_scopes(["https://www.googleapis.com/auth/devstorage.read_only"])
    # Crea el filesystem con el project_id y las credenciales
    return gcsfs.GCSFileSystem(project=creds_info["project_id"], token=creds)

fs = get_fs()
BUCKET = st.secrets["gcs"]["bucket"]

# --- Utilidades de lectura (cache_data = cachea bytes/dfs) ---
@st.cache_data(ttl = 3600)
def read_parquet_gcs(path: str) -> pd.DataFrame:
    with fs.open(f"gs://{BUCKET}/{path}", "rb") as f:
        return pd.read_parquet(f)
    
@st.cache_resource
def load_joblib(path: str):
    with fs.open(f"gs://{BUCKET}/{path}", "rb") as f:
        return joblib.load(f)

@st.cache_resource
def load_pickle(path: str):
    with fs.open(f"gs://{BUCKET}/{path}", "rb") as f:
        return pickle.load(f)     


st.set_page_config(
    page_title = "Accidentes MX",
    layout = "wide"
    )

# --- Carga de archivos ---
with st.spinner("Cargando artefactos de clasificación..."):
    clf = load_joblib("app_artifacts/models/clasificacion_xgb.joblib")
    # kmeans = load_joblib("app_artifacts/models/kmeans_k8.joblib")
    target_classes = load_joblib("app_artifacts/models/target_classes.joblib")
    preds_df = read_parquet_gcs("app_artifacts/predictions/clasificacion_preds_2023.parquet")

# df_profiles = read_parquet_gcs("app_artifacts/clusters/cluster_profiles.parquet")
# df_assign = read_parquet_gcs("app_artifacts/clusters/assignments_2023.parquet")
# df_assign = read_parquet_gcs("app_artifacts/forecasts/forecast_hourly_next168.parquet")

st.success("Artefactos cargados desde GCS (cacheados)")

# ----------------------------
# Métricas globales (a partir de preds_df)
# Nota: estas predicciones son sobre TODO X (ojo con sobreoptimismo si incluiste train).
# ----------------------------
y_true = preds_df["CLASACC_true"].astype(str)
y_pred = preds_df["CLASACC_pred"].astype(str)

acc = accuracy_score(y_true, y_pred)
f1m = f1_score(y_true, y_pred, average="macro")
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("F1-macro", f"{f1m:.3f}")
c3.metric("Registros evaluados", f"{len(preds_df):,}")

st.subheader("Reporte por clase")
st.dataframe(
    report_df.round(3),
    use_container_width=True
)

# ----------------------------
# Matriz de confusión (normalizada por filas)
# ----------------------------
st.subheader("Matriz de confusión (normalizada por clase real)")
labels = list(sorted(y_true.unique()))
cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

st.dataframe(
    cm_df.style.format("{:.2f}").background_gradient(cmap="Blues"),
    use_container_width=True
)

# ----------------------------
# Confianza promedio por clase (probabilidad media de la clase verdadera)
# ----------------------------
st.subheader("Confianza promedio por clase (probabilidad media)")
proba_cols = [c for c in preds_df.columns if c.startswith("proba_")]
if proba_cols:
    # Normaliza nombres: "proba_Sólo daños" -> "Sólo daños"
    proba_named = preds_df[proba_cols].copy()
    proba_named.columns = [c.replace("proba_", "") for c in proba_named.columns]

    # Confianza media cuando la clase verdadera es c
    conf_by_true = (
        pd.concat([y_true.rename("true"), proba_named], axis=1)
        .groupby("true")[proba_named.columns]
        .mean()
        .stack()
        .rename("confianza")
        .reset_index()
        .rename(columns={"level_1": "clase"})
    )
    # Filtra la diagonal (confianza de la clase correcta)
    conf_diag = conf_by_true[conf_by_true["true"] == conf_by_true["clase"]][["true", "confianza"]]
    conf_diag = conf_diag.set_index("true").reindex(labels)

    st.bar_chart(conf_diag, height=220, use_container_width=True)
else:
    st.info("No se encontraron columnas de probabilidad en preds_df (proba_*).")

# ----------------------------
# Errores “más seguros”: cuando el modelo se equivocó con alta confianza
# ----------------------------
st.subheader("Errores de alta confianza (top 15)")
wrong = preds_df.loc[y_true != y_pred].copy()

def prob_of_pred(row):
    col = f"proba_{row['CLASACC_pred']}"
    return row[col] if col in row.index else np.nan

if not wrong.empty and proba_cols:
    wrong["confianza_pred"] = wrong.apply(prob_of_pred, axis=1)
    top_wrong = wrong.sort_values("confianza_pred", ascending=False).head(15)
    st.dataframe(
        top_wrong[["ROW_ID","CLASACC_true","CLASACC_pred","confianza_pred"]].round(3),
        use_container_width=True
    )
else:
    st.write("— No hay información suficiente para listar errores con confianza.")

# ----------------------------
# Scoring interactivo (formulario)
# ----------------------------
st.header("Scoring interactivo")

st.caption("Completa los campos que tengas. Las columnas faltantes se completan automáticamente (categóricas='NA', numéricas=0).")

def _get_expected_cols_from_pipeline(clf_pipeline):
    """Obtiene las columnas de entrada esperadas (cat y num) desde el ColumnTransformer ('prep')."""
    prep = clf_pipeline.named_steps["prep"]
    cat_cols, num_cols = [], []
    for name, trans, cols in prep.transformers_:
        if isinstance(trans, type(prep.transformers_[0][1])):  # no fiable, mejor por nombre
            pass
        if name == "cat":
            cat_cols = list(cols)
        elif name == "num":
            num_cols = list(cols)
    # Fallback: detectar OneHot por atributo
    if not cat_cols or not num_cols:
        for name, trans, cols in prep.transformers_:
            if hasattr(trans, "categories_") or hasattr(trans, "categories"):
                cat_cols = list(cols)
            else:
                num_cols = list(cols)
    return cat_cols, num_cols

def build_X_for_model(ui_dict: dict, clf_pipeline):
    """Construye un DataFrame con TODAS las columnas que el modelo espera."""
    cat_cols, num_cols = _get_expected_cols_from_pipeline(clf_pipeline)
    row = {}
    for c in cat_cols:
        row[c] = str(ui_dict.get(c, "NA"))
    for c in num_cols:
        val = ui_dict.get(c, 0)
        try:
            row[c] = float(val)
        except Exception:
            row[c] = 0.0
    return pd.DataFrame([row], columns=cat_cols + num_cols)

with st.form("form_scoring"):
    st.subheader("Características del escenario")
    # Categóricas (usa los nombres que entrenaste)
    colA, colB, colC, colD = st.columns(4)
    EDO     = colA.text_input("EDO", value="NA")
    TIPACCID= colB.text_input("TIPACCID", value="Colisión con vehículo automotor")
    CAUSAACCI = colC.text_input("CAUSAACCI", value="Conductor")
    CVEGEO5   = colD.text_input("CVEGEO5", value="090150001")

    # Numéricas principales
    col1, col2 = st.columns(2)
    HORA = col1.slider("HORA", 0, 23, 14)
    MES  = col2.slider("MES", 1, 12, 6)

    with st.expander("Vehículos (cuentas)"):
        cols = st.columns(8)
        AUTOMOVIL  = cols[0].number_input("AUTOMOVIL", 0, 9999, 1)
        CAMION     = cols[1].number_input("CAMION", 0, 9999, 0)
        CAMIONETA  = cols[2].number_input("CAMIONETA", 0, 9999, 0)
        MOTOCICLET = cols[3].number_input("MOTOCICLET", 0, 9999, 0)
        BICICLETA  = cols[4].number_input("BICICLETA", 0, 9999, 0)
        TRANVIA    = cols[5].number_input("TRANVIA", 0, 9999, 0)
        OMNIBUS    = cols[6].number_input("OMNIBUS", 0, 9999, 0)
        PASCAMION  = cols[7].number_input("PASCAMION", 0, 9999, 0)

        cols2 = st.columns(8)
        CAMPASAJ   = cols2[0].number_input("CAMPASAJ", 0, 9999, 0)
        MICROBUS   = cols2[1].number_input("MICROBUS", 0, 9999, 0)
        TRACTOR    = cols2[2].number_input("TRACTOR", 0, 9999, 0)
        FERROCARRI = cols2[3].number_input("FERROCARRI", 0, 9999, 0)
        OTROVEHIC  = cols2[4].number_input("OTROVEHIC", 0, 9999, 0)
        MUERTOS_TOTAL = cols2[5].number_input("MUERTOS_TOTAL", 0, 9999, 0)
        HERIDOS_TOTAL = cols2[6].number_input("HERIDOS_TOTAL", 0, 9999, 0)

    submitted = st.form_submit_button("Predecir")

if submitted:
    ui_dict = {
        "EDO": EDO, "TIPACCID": TIPACCID, "CAUSAACCI": CAUSAACCI, "CVEGEO5": CVEGEO5,
        "HORA": HORA, "MES": MES,
        "AUTOMOVIL": AUTOMOVIL, "CAMION": CAMION, "CAMIONETA": CAMIONETA,
        "MOTOCICLET": MOTOCICLET, "BICICLETA": BICICLETA,
        "TRANVIA": TRANVIA, "OMNIBUS": OMNIBUS,
        "PASCAMION": PASCAMION, "CAMPASAJ": CAMPASAJ, "MICROBUS": MICROBUS,
        "TRACTOR": TRACTOR, "FERROCARRI": FERROCARRI, "OTROVEHIC": OTROVEHIC,
        "MUERTOS_TOTAL": MUERTOS_TOTAL, "HERIDOS_TOTAL": HERIDOS_TOTAL,
    }
    X_user = build_X_for_model(ui_dict, clf)
    # Predicción y probabilidades
    proba = clf.predict_proba(X_user)[0]
    # Etiquetas (orden consistente con predict_proba)
    classes = list(target_classes) if target_classes is not None else list(clf.named_steps["clf"].classes_)
    pred_idx = int(np.argmax(proba))
    st.success(f"Clase predicha: {classes[pred_idx]}")
    st.write(pd.DataFrame({"Clase": classes, "Probabilidad": proba}).sort_values("Probabilidad", ascending=False).reset_index(drop=True))
    # Diagnóstico de columnas faltantes (si hubo)
    cat_cols, num_cols = _get_expected_cols_from_pipeline(clf)
    expected = set(cat_cols + num_cols)
    missing = expected - set(X_user.columns)
    if missing:
        st.info(f"Se completaron automáticamente columnas faltantes: {sorted(missing)}")

# ----------------------------
# Interpretación automática (texto)
# ----------------------------
st.header("Interpretación automática")
txt = []

# Distribución de clases
dist = y_true.value_counts(normalize=True).rename("pct")
mayoritaria = dist.idxmax()
txt.append(f"- La clase mayoritaria en el conjunto es **{mayoritaria}** ({dist.max():.1%}).")

# Peor clase por F1
rep_df_clean = report_df.loc[[c for c in report_df.index if c in y_true.unique()], ["precision","recall","f1-score"]]
worst_cls = rep_df_clean["f1-score"].idxmin()
txt.append(f"- La clase con menor F1 es **{worst_cls}** (F1={rep_df_clean.loc[worst_cls,'f1-score']:.3f}).")

# Desempeño global
txt.append(f"- Desempeño global: **accuracy={acc:.3f}**, **F1-macro={f1m:.3f}**. Recomendable complementar con PR-AUC por clase.")

# Confusiones principales
conf_tbl = pd.crosstab(y_true, y_pred, normalize="index")
if not conf_tbl.empty:
    # para cada clase, segunda mayor predicción
    conf_issues = []
    for c in conf_tbl.index:
        row = conf_tbl.loc[c].sort_values(ascending=False)
        if len(row) >= 2:
            conf_issues.append(f"**{c}** → se confunde con **{row.index[1]}** ({row.iloc[1]:.1%}).")
    if conf_issues:
        txt.append("- Principales confusiones: " + " ".join(conf_issues))

st.markdown("\n".join(txt))