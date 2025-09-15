import io, json
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
import gcsfs, joblib

# ---------------------------------------------------
# Si YA definiste estos helpers en otro archivo, reutilízalos
# ---------------------------------------------------
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

@st.cache_data(ttl=3600)
def read_parquet_gcs(path: str) -> pd.DataFrame:
    with fs.open(f"gs://{BUCKET}/{path}", "rb") as f:
        return pd.read_parquet(f)

@st.cache_data(ttl=3600)
def read_json_gcs(path: str) -> dict | None:
    p = f"gs://{BUCKET}/{path}"
    if fs.exists(p):
        with fs.open(p, "rb") as f:
            return json.load(f)
    return None

@st.cache_resource
def load_joblib(path: str):
    with fs.open(f"gs://{BUCKET}/{path}", "rb") as f:
        return joblib.load(f)

def gcs_exists(path: str) -> bool:
    return fs.exists(f"gs://{BUCKET}/{path}")

# ---------------------------------------------------
# Rutas por defecto y/o desde meta.info
# ---------------------------------------------------
META_PATH = "app_artifacts/meta/info.json"
meta = read_json_gcs(META_PATH)

def _path_from_meta_or_default(key: str, default_rel: str) -> str:
    if meta and "models" in meta and "clustering" in meta["models"]:
        val = meta["models"]["clustering"].get(key)
        if val:
            # admite ruta gs://... o relativa
            prefix = f"gs://{BUCKET}/"
            return val[len(prefix):] if isinstance(val, str) and val.startswith(prefix) else val
    return default_rel

MODEL_PATH   = _path_from_meta_or_default("path",              "app_artifacts/models/kmeans_k8.joblib")
ASSIGN_PATH  = _path_from_meta_or_default("assignments_path",  "app_artifacts/clusters/assignments_2023.parquet")
PROFILE_PATH = _path_from_meta_or_default("profiles_path",     "app_artifacts/clusters/cluster_profiles.parquet")
TOP_TIPO_PATH= _path_from_meta_or_default("top_tipaccid_path", "app_artifacts/clusters/top_tipaccid.parquet")
TOP_CAUSA_PATH=_path_from_meta_or_default("top_causaacci_path","app_artifacts/clusters/top_causaacci.parquet")

st.header("Clusterización – K-Means sobre ATUS 2023")

st.subheader(
    body = """
    Propósito. Agrupar registros 2023 en patrones homogéneos para descubrir perfiles de accidente y apoyar acciones preventivas/focalizadas.
    """)

# ---------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------
missing = [p for p in [MODEL_PATH, ASSIGN_PATH, PROFILE_PATH] if not gcs_exists(p)]
if missing:
    st.error(f"Faltan archivos requeridos: {missing}")
    st.write("Disponibles en /app_artifacts:", fs.ls(f"gs://{BUCKET}/app_artifacts/"))
    st.stop()

with st.spinner("Cargando artefactos de clusterización..."):
    kmeans_pipe = load_joblib(MODEL_PATH)           # Pipeline(prep + kmeans)
    assign_df   = read_parquet_gcs(ASSIGN_PATH)     # ROW_ID, HORA, MES, MUERTOS_TOTAL, HERIDOS_TOTAL, TIPACCID, CAUSAACCI, CVEGEO5, CLUSTER
    profiles    = read_parquet_gcs(PROFILE_PATH)    # CLUSTER, n, muertos_prom, heridos_prom
    top_tipo    = read_parquet_gcs(TOP_TIPO_PATH) if gcs_exists(TOP_TIPO_PATH) else None
    top_causa   = read_parquet_gcs(TOP_CAUSA_PATH) if gcs_exists(TOP_CAUSA_PATH) else None

st.success("Artefactos de clusterización cargados ✅")

# Asegura tipos
assign_df["CLUSTER"] = assign_df["CLUSTER"].astype(int)

# ---------------------------------------------------
# KPIs
# ---------------------------------------------------
n_clusters = int(assign_df["CLUSTER"].nunique())
total_n    = int(len(assign_df))
sil_meta   = meta["models"]["clustering"].get("silhouette_sampled") if (meta and "models" in meta and "clustering" in meta["models"]) else None

c1, c2, c3 = st.columns(3)
c1.metric("Clusters (K)", f"{n_clusters}")
c2.metric("Registros", f"{total_n:,}")
c3.metric("Silueta (muestra)", f"{sil_meta:.3f}" if sil_meta is not None else "N/D")

# ---------------------------------------------------
# Perfiles por cluster
# ---------------------------------------------------
st.subheader("Perfiles por cluster")
profiles = profiles.sort_values("n", ascending=False).reset_index(drop=True)
st.dataframe(profiles.style.format({"muertos_prom":"{:.4f}", "heridos_prom":"{:.4f}"}), use_container_width=True)

# Distribución de tamaños
st.subheader("Tamaño de clusters")
size_series = assign_df["CLUSTER"].value_counts().sort_index()
st.bar_chart(size_series, height=220, use_container_width=True)

# ---------------------------------------------------
# Exploración por cluster
# ---------------------------------------------------
st.subheader("Explorar un cluster")
clist = sorted(assign_df["CLUSTER"].unique().tolist())
sel = st.selectbox("Cluster", clist, index=0)

sub = assign_df[assign_df["CLUSTER"] == sel].copy()
cA, cB, cC = st.columns(3)
cA.metric("Registros (cluster)", f"{len(sub):,}")
cB.metric("Heridos promedio", f"{sub['HERIDOS_TOTAL'].mean():.3f}")
cC.metric("Muertos promedio", f"{sub['MUERTOS_TOTAL'].mean():.3f}")

# Top TIPACCID / CAUSAACCI (si existen tablas precomputadas)
if top_tipo is not None:
    st.markdown("**Top TIPACCID en el cluster**")
    ttip = top_tipo[top_tipo["CLUSTER"] == sel].copy()
    ttip["pct"] = (ttip["conteo"] / len(sub)).round(3)
    st.dataframe(ttip[["TIPACCID","conteo","pct"]], use_container_width=True)

if top_causa is not None:
    st.markdown("**Top CAUSAACCI en el cluster**")
    tcau = top_causa[top_causa["CLUSTER"] == sel].copy()
    tcau["pct"] = (tcau["conteo"] / len(sub)).round(3)
    st.dataframe(tcau[["CAUSAACCI","conteo","pct"]], use_container_width=True)

# Distribución horaria y mensual
cH, cM = st.columns(2)
with cH:
    st.markdown("**Distribución por hora**")
    by_h = sub.groupby("HORA").size()
    # asegura que estén todas las horas 0..23
    by_h = by_h.reindex(range(24), fill_value=0)
    st.bar_chart(by_h, height=220, use_container_width=True)
with cM:
    st.markdown("**Distribución por mes**")
    by_m = sub.groupby("MES").size()
    by_m = by_m.reindex(range(1,13), fill_value=0)
    st.bar_chart(by_m, height=220, use_container_width=True)

# Muestra de registros
st.markdown("**Muestra de registros del cluster**")
st.dataframe(sub.sample(min(200, len(sub))), use_container_width=True, height=260)

# ---------------------------------------------------
# What-if: asignación de un nuevo escenario a cluster
# ---------------------------------------------------
st.header("What-if: asignar un escenario a un cluster")

def _get_expected_cols_from_pipeline(pipe):
    prep = pipe.named_steps["prep"]
    cat_cols, num_cols = [], []
    for name, trans, cols in prep.transformers_:
        if name == "cat":
            cat_cols = list(cols)
        elif name == "num":
            num_cols = list(cols)
    # Fallback si cambiaron nombres
    if not cat_cols or not num_cols:
        for _, trans, cols in prep.transformers_:
            if hasattr(trans, "categories_") or hasattr(trans, "categories"):
                cat_cols = list(cols)
            else:
                num_cols = list(cols)
    return cat_cols, num_cols

def build_X_for_kmeans(ui_dict: dict, pipe) -> pd.DataFrame:
    cat_cols, num_cols = _get_expected_cols_from_pipeline(pipe)
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

with st.form("form_kmeans"):
    st.caption("Completa las variables; lo faltante se completa (categóricas='NA', numéricas=0).")
    colA, colB, colC = st.columns(3)
    TIPACCID = colA.text_input("TIPACCID", value="Colisión con vehículo automotor")
    CAUSAACCI= colB.text_input("CAUSAACCI", value="Conductor")
    CVEGEO5  = colC.text_input("CVEGEO5", value="090150001")

    col1, col2 = st.columns(2)
    HORA = col1.slider("HORA", 0, 23, 14)
    MES  = col2.slider("MES", 1, 12, 6)

    with st.expander("Vehículos y resultados (opcional)"):
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

    submitted = st.form_submit_button("Asignar cluster")

if submitted:
    ui = {
        "TIPACCID": TIPACCID, "CAUSAACCI": CAUSAACCI, "CVEGEO5": CVEGEO5,
        "HORA": HORA, "MES": MES,
        "AUTOMOVIL": AUTOMOVIL, "CAMION": CAMION, "CAMIONETA": CAMIONETA,
        "MOTOCICLET": MOTOCICLET, "BICICLETA": BICICLETA,
        "TRANVIA": TRANVIA, "OMNIBUS": OMNIBUS, "PASCAMION": PASCAMION,
        "CAMPASAJ": CAMPASAJ, "MICROBUS": MICROBUS, "TRACTOR": TRACTOR,
        "FERROCARRI": FERROCARRI, "OTROVEHIC": OTROVEHIC,
        "MUERTOS_TOTAL": MUERTOS_TOTAL, "HERIDOS_TOTAL": HERIDOS_TOTAL,
    }
    X_user = build_X_for_kmeans(ui, kmeans_pipe)
    # cluster predicho
    cluster_id = int(kmeans_pipe.predict(X_user)[0])
    st.success(f"Escenario asignado al cluster: **{cluster_id}**")

    # Distancias a centroides (menor = más cercano)
    prep = kmeans_pipe.named_steps["prep"]
    km   = kmeans_pipe.named_steps["kmeans"]
    Xs   = prep.transform(X_user)
    dists = km.transform(Xs)[0]  # distancias euclídeas a cada centro
    dist_tbl = pd.DataFrame({"CLUSTER": list(range(len(dists))), "distancia": dists}).sort_values("distancia")
    st.write("Distancias a centroides (menor es más parecido):")
    st.dataframe(dist_tbl, use_container_width=True)

# ---------------------------------------------------
# Descargas
# ---------------------------------------------------
st.subheader("Descargas")
st.download_button(
    "Descargar asignaciones (Parquet)",
    data=fs.open(f"gs://{BUCKET}/{ASSIGN_PATH}", "rb").read(),
    file_name="assignments_2023.parquet",
    mime="application/octet-stream"
)
st.download_button(
    "Descargar perfiles (Parquet)",
    data=fs.open(f"gs://{BUCKET}/{PROFILE_PATH}", "rb").read(),
    file_name="cluster_profiles.parquet",
    mime="application/octet-stream"
)
if gcs_exists(TOP_TIPO_PATH):
    st.download_button(
        "Descargar top TIPACCID (Parquet)",
        data=fs.open(f"gs://{BUCKET}/{TOP_TIPO_PATH}", "rb").read(),
        file_name="top_tipaccid.parquet",
        mime="application/octet-stream"
    )
if gcs_exists(TOP_CAUSA_PATH):
    st.download_button(
        "Descargar top CAUSAACCI (Parquet)",
        data=fs.open(f"gs://{BUCKET}/{TOP_CAUSA_PATH}", "rb").read(),
        file_name="top_causaacci.parquet",
        mime="application/octet-stream"
    )

# ---------------------------------------------------
# Notas
# ---------------------------------------------------
st.markdown("""
**Notas**
- Este K-Means se entrenó con One-Hot + escalado. El **what-if** reconvierte tus entradas con el mismo preprocesamiento del pipeline.
- `MUERTOS_TOTAL` y `HERIDOS_TOTAL` aparecen como features (versión descriptiva). Si buscas un uso preventivo puro, genera otra versión del modelo **sin outcomes** y publícala aparte.
- La **silueta** mostrada es la estimada en entrenamiento (muestra). Para recalcularla aquí (costo alto), es mejor hacerlo offline y guardar el valor en `meta/info.json`.
""")