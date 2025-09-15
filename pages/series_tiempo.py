import io, json, joblib, pickle
import numpy as np
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
import gcsfs

# ================================
# Config y utilidades de GCS
# ================================
st.set_page_config(page_title="Accidentes MX – Series de tiempo", layout="wide")

st.header(body = "Accidentes MX - Series de tiempo")

st.subheader(
    body = """
    Propósito. Mostrar el pronóstico horario (168 h) de accidentes a partir de un modelo SARIMAX ligero y ofrecer resúmenes operativos.
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

@st.cache_data(ttl=3600)
def read_json_gcs(path: str) -> dict | None:
    p = f"gs://{BUCKET}/{path}"
    if fs.exists(p):
        with fs.open(p, "rb") as f:
            return json.load(f)
    return None

def gcs_exists(path: str) -> bool:
    return fs.exists(f"gs://{BUCKET}/{path}")

st.set_page_config(
    page_title = "Accidentes MX - Series de tiempo",
    layout = "wide"
    )

st.header("Series de tiempo – Pronóstico horario (168 h)")

# ================================
# Carga de artefactos
# ================================
with st.spinner("Cargando forecast..."):
    if not gcs_exists("app_artifacts/forecasts/forecast_hourly_next168.parquet"):
        st.error(f"No se encontró app_artifacts/forecasts/forecast_hourly_next168.parquet.")
        st.write("Archivos disponibles en /forecasts/:", fs.ls(f"gs://{BUCKET}/app_artifacts/forecasts/"))
        st.stop()

    df_fore = read_parquet_gcs("app_artifacts/forecasts/forecast_hourly_next168.parquet")
    meta = read_json_gcs("app_artifacts/meta/info.json")

# Normaliza columnas esperadas
if "timestamp" not in df_fore.columns:
    st.error("El forecast no tiene columna 'timestamp'. Asegúrate de haberlo guardado con {'timestamp','yhat'}.")
    st.stop()
if "yhat" not in df_fore.columns:
    st.error("El forecast no tiene columna 'yhat'.")
    st.stop()

df_fore = df_fore.copy()
df_fore["timestamp"] = pd.to_datetime(df_fore["timestamp"])
df_fore = df_fore.sort_values("timestamp")

# ================================
# Metadatos y KPIs
# ================================
c1, c2, c3 = st.columns(3)
hstart = df_fore["timestamp"].min()
hend   = df_fore["timestamp"].max()
c1.metric("Inicio pronóstico", hstart.strftime("%Y-%m-%d %H:%M"))
c2.metric("Fin pronóstico",    hend.strftime("%Y-%m-%d %H:%M"))
c3.metric("Horizonte (horas)", f"{len(df_fore):,}")

if meta and "models" in meta and "timeseries" in meta["models"]:
    ts_meta = meta["models"]["timeseries"]
    st.caption(f"Fin de historial: {ts_meta.get('end_of_history','N/D')} · Estacionalidad: {ts_meta.get('seasonality','N/D')}")

# ================================
# Filtro de horizonte y gráfico
# ================================
st.subheader("Pronóstico (línea)")
default_hours = min(len(df_fore), 168)
h_slider = st.slider("Mostrar últimas N horas del pronóstico",
                     min_value=24, max_value=int(len(df_fore)), value=int(default_hours), step=24)
df_plot = df_fore.tail(h_slider).set_index("timestamp")

# Gráfico simple (línea)
st.line_chart(df_plot[["yhat"]], height=300, use_container_width=True)

# ================================
# Totales por día y picos horarios
# ================================
st.subheader("Resumen del horizonte")
df_fore["date"] = df_fore["timestamp"].dt.date
by_day = df_fore.groupby("date", as_index=False)["yhat"].sum().rename(columns={"yhat":"total_dia"})
st.bar_chart(by_day.set_index("date"), height=220, use_container_width=True)

# Top horas con mayor yhat
top_hours = (df_fore
             .sort_values("yhat", ascending=False)
             .head(15)[["timestamp","yhat"]])
top_hours["dow"] = top_hours["timestamp"].dt.day_name(locale="es_MX") if hasattr(pd.Series.dt, "day_name") else top_hours["timestamp"].dt.dayofweek
st.write("**Top 15 horas esperadas (yhat):**")
st.dataframe(top_hours.rename(columns={"yhat":"yhat (conteos esperados)"}), use_container_width=True)

# ================================
# Descarga y tabla
# ================================
st.subheader("Tabla del forecast y descarga")
st.dataframe(df_fore, use_container_width=True, height=320)

# Botón de descarga (CSV)
csv_bytes = df_fore.to_csv(index=False).encode("utf-8")
st.download_button("Descargar forecast (CSV)", data=csv_bytes, file_name="forecast_hourly_next168.csv", mime="text/csv")

# Nota interpretativa
st.markdown("""
**Notas de interpretación**
- `yhat` representa el **conteo esperado** de accidentes por hora en el próximo horizonte.
- El modelo SARIMAX se entrenó con estacionalidad **diaria (24h)** y componentes de **semana** vía Fourier (si aplicaste K>0), además de variables de calendario (DOW, hora en seno/coseno).
- Dado que el forecast se guardó sin bandas de confianza (por uso de `cov_type="none"`), se muestra la trayectoria puntual. Si necesitas intervalos, entrena con `cov_type="robust"` y guarda `yhat_lo/yhat_hi`.
- Usa el **resumen por día** y los **picos horarios** para planear recursos (operativos, preventivos) en las horas/días con mayor demanda esperada.
""")