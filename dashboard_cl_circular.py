"""

     CL CIRCULAR  Dashboard de Anlisis de Mercado           
     Exportaciones Mxico  EE.UU. (2021-2025)                

  Para correr:                                                 
     pip install streamlit pandas numpy matplotlib             
                 scikit-learn scipy openpyxl plotly            
     streamlit run dashboard_cl_circular.py                    

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import os
import re
import unicodedata
from itertools import product

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except Exception:
    SARIMA_AVAILABLE = False

# 
# CONFIGURACIN GENERAL
# 
st.set_page_config(
    page_title="CL Circular - Analisis de Mercado",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Fondo general */
    .stApp {
        background: #FFFFFF;
        color: #1A2B3C;
    }

    /* Texto general: párrafos y markdown */
    p, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #1A2B3C !important;
    }

    /* Etiquetas de widgets (selectbox, slider, radio, etc.) */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    label, .stLabel {
        color: #1A3355 !important;
        font-weight: 500;
    }

    /* st.caption */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #4A607A !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F0F4F8 !important;
        border-right: 1px solid #D0DCE8;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: #1A3355 !important;
    }

    /* Metricas */
    [data-testid="metric-container"] {
        background: #F0F4F8;
        border: 1px solid #D0DCE8;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label,
    [data-testid="metric-container"] [data-testid="stMetricLabel"] p {
        color: #1565A0 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="metric-container"] [data-testid="metric-value"],
    [data-testid="stMetricValue"] {
        color: #0D1F2D !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        font-family: 'DM Mono', monospace !important;
    }
    [data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 13px !important;
    }

    /* Headers */
    h1 { color: #0D5490 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h2 { color: #1565A0 !important; font-weight: 600 !important; }
    h3 { color: #1A3355 !important; font-weight: 600 !important; }
    h4 { color: #1A3355 !important; font-weight: 500 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #E8EEF4;
        border-radius: 8px;
        gap: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #1A3355;
        border-radius: 6px;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #1565A0 !important;
        color: #FFFFFF !important;
    }

    /* Selectbox y multiselect — campo cerrado */
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] {
        background: #F0F4F8 !important;
        border-color: #B0C4D8 !important;
        color: #1A2B3C !important;
    }
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] input,
    .stMultiSelect [data-baseweb="select"] div,
    .stMultiSelect [data-baseweb="select"] span,
    .stMultiSelect [data-baseweb="select"] input {
        background: #F0F4F8 !important;
        color: #1A2B3C !important;
    }

    /* Dropdown popup (portal BaseWeb — fuera del DOM normal) */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div {
        background: #FFFFFF !important;
    }
    [data-baseweb="menu"],
    [data-baseweb="menu"] ul,
    [data-baseweb="menu"] li {
        background: #FFFFFF !important;
        color: #1A2B3C !important;
    }
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [aria-selected="true"] {
        background: #EAF3FB !important;
        color: #0D1F2D !important;
    }
    [data-baseweb="option"] {
        background: #FFFFFF !important;
        color: #1A2B3C !important;
    }
    [data-baseweb="option"]:hover {
        background: #EAF3FB !important;
    }

    /* Tags de multiselect seleccionados */
    [data-baseweb="tag"] {
        background: #D8E8F8 !important;
        color: #0D3A5C !important;
        border: 1px solid #B0C4D8 !important;
    }
    [data-baseweb="tag"] span {
        color: #0D3A5C !important;
    }

    /* NumberInput / TextInput */
    .stNumberInput input,
    .stTextInput input,
    .stTextArea textarea {
        background: #F0F4F8 !important;
        color: #1A2B3C !important;
        border-color: #B0C4D8 !important;
    }

    /* Radio buttons */
    .stRadio label,
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: #1A3355 !important;
    }

    /* Checkbox */
    .stCheckbox label,
    .stCheckbox span {
        color: #1A3355 !important;
    }

    /* Expander */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        background: #F0F4F8 !important;
        color: #1A3355 !important;
    }
    [data-testid="stExpander"] {
        border-color: #D0DCE8 !important;
    }
    [data-testid="stExpanderDetails"] {
        background: #FFFFFF !important;
    }

    /* Tablas st.table() */
    .stTable table {
        background: #FFFFFF !important;
        color: #1A2B3C !important;
    }
    .stTable th {
        background: #F0F4F8 !important;
        color: #1A3355 !important;
        border-color: #D0DCE8 !important;
    }
    .stTable td {
        background: #FFFFFF !important;
        color: #1A2B3C !important;
        border-color: #D0DCE8 !important;
    }
    .stTable tr:nth-child(even) td {
        background: #F8FAFC !important;
    }

    /* Dataframe wrapper */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"],
    .stDataFrame,
    .stDataFrame > div {
        background: #FFFFFF !important;
        color: #1A2B3C !important;
        border: 1px solid #D0DCE8 !important;
        border-radius: 8px;
    }

    /* st.info / st.warning / st.error — fondo claro */
    [data-testid="stAlert"] {
        background: #EAF3FB !important;
        color: #0D3A5C !important;
        border-color: #B0C4D8 !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] {
        color: #1565A0;
    }
    .stSlider [data-testid="stTickBar"],
    .stSlider p {
        color: #1A3355 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #D0DCE8;
        border-radius: 8px;
    }

    /* Divider personalizado */
    hr { border-color: #D0DCE8; }

    /* Caption */
    .caption-text {
        color: #4A607A;
        font-size: 11px;
        text-align: center;
        margin-top: -8px;
        font-style: italic;
    }

    /* Badge */
    .sector-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }

    /* Info box */
    .info-box {
        background: #EAF3FB;
        border-left: 3px solid #1565A0;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        color: #0D3A5C;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# 
# CONSTANTES Y CONFIGURACIN
# 
COLORS = {
    "Cerveza":      "#1A7AB5",
    "Autopartes":   "#7B2FBE",
    "Harina_Maiz":  "#C4621A",
    "Harina_Trigo": "#B54127",
}

LABELS = {
    "Cerveza":      "Cerveza (HS 220300)",
    "Autopartes":   "Autopartes (HS 8708)",
    "Harina_Maiz":  "Harina de Maiz (HS 110220)",
    "Harina_Trigo": "Harina de Trigo (HS 1101)",
}

_AXIS_DARK = dict(
    gridcolor="#D0DCE8",
    linecolor="#B0C4D8",
    zerolinecolor="#B0C4D8",
    tickfont=dict(family="DM Sans", color="#1A3355", size=13),
    title_font=dict(family="DM Sans", color="#0D1F2D", size=14),
    tickcolor="#6B8CAE",
)
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(family="DM Sans", color="#1A3355", size=14),
        xaxis=_AXIS_DARK,
        yaxis=_AXIS_DARK,
        legend=dict(
            bgcolor="#F0F4F8",
            bordercolor="#D0DCE8",
            borderwidth=1,
            font=dict(family="DM Sans", color="#1A3355", size=13),
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(family="DM Sans", color="#1A3355", size=12),
            title_font=dict(family="DM Sans", color="#0D1F2D", size=13),
        ),
        margin=dict(t=40, b=40, l=50, r=20),
        uniformtext=dict(minsize=13, mode="hide"),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#B0C4D8",
            font=dict(family="DM Sans", color="#0D1F2D", size=20),
            align="left",
        ),
    )
)

def apply_template_layout(fig, **overrides):
    """Aplicar layout base y forzar fuentes oscuras en todos los ejes."""
    # Normalizar title: si es string convertir a dict para evitar subtitle=undefined
    # en Plotly 5.24+ donde title.subtitle fue introducido
    if 'title' in overrides:
        t = overrides['title']
        if isinstance(t, str):
            overrides['title'] = dict(
                text=t,
                font=dict(family="DM Sans", color="#0D1F2D", size=17),
            )
        elif isinstance(t, dict) and 'font' not in t:
            overrides['title'] = dict(font=dict(family="DM Sans", color="#0D1F2D", size=17), **t)
    layout = dict(PLOTLY_TEMPLATE['layout'])
    layout.update(overrides)
    fig.update_layout(**layout)
    # Forzar tickfont y title_font oscuros en todos los ejes (incluyendo secundarios y facets)
    fig.update_xaxes(
        tickfont=dict(family="DM Sans", color="#1A3355", size=13),
        title_font=dict(family="DM Sans", color="#0D1F2D", size=14),
        tickcolor="#6B8CAE",
        gridcolor="#D0DCE8",
        linecolor="#B0C4D8",
    )
    fig.update_yaxes(
        tickfont=dict(family="DM Sans", color="#1A3355", size=13),
        title_font=dict(family="DM Sans", color="#0D1F2D", size=14),
        tickcolor="#6B8CAE",
        gridcolor="#D0DCE8",
        linecolor="#B0C4D8",
    )
    fig.update_annotations(font=dict(size=13, color="#1A3355"))

# 
# CARGA DE DATOS
# 

# Detectar rutas de los archivos
def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def find_file(filename):
    """Buscar archivo en directorio actual, uploads, o ingresar ruta manualmente."""
    candidates = [
        filename,
        os.path.join("uploads", filename),
        os.path.join("/mnt/user-data/uploads", filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

FILES = {
    "Cerveza":      "Comercio_Cerveza.xlsx",
    "Autopartes":   "Comercio_Autopartes.xlsx",
    "Harina_Maiz":  "Comercio_Harina_Maiz.xlsx",
    "Harina_Trigo": "Comercio_Harina_Trigo.xlsx",
}

# Archivos con datos de volumen (netWgt/qty) por sector
VOLUME_FILES = {
    "Cerveza":      "Cerveza 2021-2025.xlsx",
    "Autopartes":   "Comercio_Autopartes_Volumen.xlsx",
    "Harina_Maiz":  "harina maiz 2021-2025.xlsx",
    "Harina_Trigo": "harina trigo 2021-2025.xlsx",
}

# Capacidad aproximada por contenedor/camion en unidades nativas del sector
CONTAINER_CAPACITY = {
    "Cerveza":      26_000,   # litros por camion (26 paletas × ~1,000L)
    "Autopartes":   20_000,   # kg por camion (carga mixta)
    "Harina_Maiz":  25_000,   # kg por camion (granel seco)
    "Harina_Trigo": 25_000,   # kg por camion (granel seco)
}

VOL_UNIT_LABEL = {
    "Cerveza":      "litros",
    "Autopartes":   "kg",
    "Harina_Maiz":  "kg",
    "Harina_Trigo": "kg",
}

PROSPECTS_FILE = "prospectos_clcircular.csv"
CRL_FILE = "CRL.xlsx"
RISK_BASE_FILES = [
    "montecarlo_riesgo_clcircular.csv",
    os.path.join("data", "montecarlo_riesgo_clcircular.csv"),
    r"c:\Users\pppab\Downloads\montecarlo_riesgo_clcircular.csv",
    r"c:\Users\pppab\Downloads\montecarlo_riesgo_clcircular (1).csv",
]

RISK_LEVEL_COLORS = {
    "Bajo": "#1A7A3C",
    "Moderado": "#B07D10",
    "Alto": "#C4621A",
    "Muy_Alto": "#C53030",
    "Critico": "#8B0000",
}

SIM_DEFAULTS = {
    "Cerveza": {"p_contam_a": 6.0, "p_contam_b": 994.0, "p_reg_a": 2.1, "p_reg_b": 97.0, "valor_mu": 65000.0},
    "Autopartes": {"p_contam_a": 3.0, "p_contam_b": 997.0, "p_reg_a": 5.0, "p_reg_b": 95.0, "valor_mu": 200000.0},
    "Harinas": {"p_contam_a": 8.0, "p_contam_b": 992.0, "p_reg_a": 3.0, "p_reg_b": 97.0, "valor_mu": 35000.0},
}

FACTOR_THRESHOLDS = {
    "neg_max": -5.0,   # <= -5% YoY mensual
    "pos_min": 5.0,    # >= +5% YoY mensual
}

SECTOR_CONTACT_MAP = {
    "Cerveza": "Cerveza",
    "Autopartes": "Autopartes",
    "Harina_Maiz": "Harinas",
    "Harina_Trigo": "Harinas",
}

MEX_STATE_COORDS = {
    "aguascalientes": (21.8853, -102.2916),
    "baja california": (32.6245, -115.4523),
    "baja california sur": (24.1426, -110.3128),
    "campeche": (19.8301, -90.5349),
    "chiapas": (16.7528, -93.1167),
    "chihuahua": (28.6353, -106.0889),
    "ciudad de mexico": (19.4326, -99.1332),
    "cdmx": (19.4326, -99.1332),
    "coahuila": (25.4260, -100.9950),
    "coahuila de zaragoza": (25.4260, -100.9950),
    "colima": (19.2452, -103.7241),
    "durango": (24.0277, -104.6532),
    "estado de mexico": (19.2920, -99.6530),
    "mexico": (19.2920, -99.6530),
    "guanajuato": (21.0190, -101.2574),
    "guerrero": (17.5515, -99.5006),
    "hidalgo": (20.0911, -98.7624),
    "jalisco": (20.6597, -103.3496),
    "michoacan": (19.7008, -101.1844),
    "michoacan de ocampo": (19.7008, -101.1844),
    "morelos": (18.9242, -99.2216),
    "nayarit": (21.5059, -104.8957),
    "nuevo leon": (25.6866, -100.3161),
    "oaxaca": (17.0732, -96.7266),
    "puebla": (19.0414, -98.2063),
    "queretaro": (20.5888, -100.3899),
    "quintana roo": (21.1619, -86.8515),
    "san luis potosi": (22.1565, -100.9855),
    "sinaloa": (24.8091, -107.3940),
    "sonora": (29.0729, -110.9559),
    "tabasco": (17.9892, -92.9475),
    "tamaulipas": (23.7369, -99.1411),
    "tlaxcala": (19.3182, -98.2375),
    "veracruz": (19.1738, -96.1342),
    "veracruz de ignacio de la llave": (19.1738, -96.1342),
    "yucatan": (20.9674, -89.5926),
    "zacatecas": (22.7709, -102.5832),
}

@st.cache_data(show_spinner=False)
def cargar_datos(files: dict) -> dict:
    """Cargar y limpiar todos los datasets."""
    dfs = {}
    VARIABLES = ['refYear', 'refMonth', 'period', 'fobvalue', 'netWgt', 'qty']

    for name, filename in files.items():
        path = find_file(filename)
        if path is None:
            st.error(f" Archivo no encontrado: {filename}. Coloca los .xlsx en la misma carpeta que este script.")
            continue
        df = pd.read_excel(path, sheet_name='Sheet1')
        df = df[VARIABLES].copy()
        df = df[df['fobvalue'] > 0]
        df['year'] = df['refYear'].astype(int)
        df['month'] = df['refMonth'].astype(int)
        df['fecha'] = pd.to_datetime(df['period'].astype(str), format='%Y%m')
        df['qty_clean'] = df['qty'].replace(0, np.nan)
        df['precio_unitario'] = df['fobvalue'] / df['qty_clean']
        df['fob_millon'] = df['fobvalue'] / 1_000_000
        df['peso_kt'] = df['netWgt'] / 1_000_000
        df['sector'] = name
        df['trimestre'] = df['month'].apply(lambda m: f"Q{(m-1)//3+1}")
        df = df.sort_values('fecha').reset_index(drop=True)
        df['yoy_mensual_pct'] = df['fob_millon'].pct_change(12) * 100
        df['factor_mercado'] = df['yoy_mensual_pct'].apply(_factor_from_yoy)
        dfs[name] = df
    return dfs


def _anios_factor_dominante(df_sector: pd.DataFrame, factor: str) -> list:
    """Devuelve los años donde el factor dado fue el más frecuente en ese sector."""
    if "year" not in df_sector.columns or "factor_mercado" not in df_sector.columns:
        return []
    dom = (
        df_sector.groupby("year")["factor_mercado"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Neutral")
    )
    return dom[dom == factor].index.tolist()


def _factor_from_yoy(yoy_value: float) -> str:
    """Clasifica factor de mercado segun umbral estandar tipo semaforo."""
    if pd.isna(yoy_value):
        return "Neutral"
    if yoy_value <= FACTOR_THRESHOLDS["neg_max"]:
        return "Negativo"
    if yoy_value >= FACTOR_THRESHOLDS["pos_min"]:
        return "Positivo"
    return "Neutral"


def _factor_from_irc(irc_value: float) -> str:
    """Clasifica factor de riesgo (favorable/neutral/adverso) en escala IRC 0-100."""
    if pd.isna(irc_value):
        return "Neutral"
    if irc_value <= 33:
        return "Positivo"
    if irc_value <= 66:
        return "Neutral"
    return "Negativo"


def _parse_viajes_estimados(value) -> int:
    s = str(value)
    m = re.search(r"\d+", s)
    if not m:
        return 0
    n = int(m.group())
    if "+" in s:
        n += 1
    return n


@st.cache_data(show_spinner=False)
def cargar_contactos(filename: str) -> pd.DataFrame:
    """Cargar prospectos para contacto comercial."""
    path = find_file(filename)
    if path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    if df.empty:
        return df

    required = [
        "empresa", "sector", "subsector", "estado", "ciudad", "cargo_contacto",
        "relevancia_clcircular", "viajes_mes_estimados", "modo_transporte",
        "puertos_frontera", "linkedin_empresa", "sitio_web"
    ]
    for c in required:
        if c not in df.columns:
            df[c] = ""

    rel_rank = {"Muy Alta": 5, "Alta": 4, "Media": 3, "Baja": 2, "Muy Baja": 1}
    df["sector"] = df["sector"].astype(str).str.strip()
    df["relev_rank"] = df["relevancia_clcircular"].map(rel_rank).fillna(0).astype(int)
    df["viajes_num"] = df["viajes_mes_estimados"].apply(_parse_viajes_estimados)

    return df


@st.cache_data(show_spinner=False)
def cargar_riesgo_base(candidates: tuple) -> tuple[pd.DataFrame, str]:
    """Carga CSV base Monte Carlo de riesgo desde rutas candidatas."""
    path = None
    for cand in candidates:
        p = cand if os.path.isabs(cand) else find_file(cand)
        if p and os.path.exists(p):
            path = p
            break
    if path is None:
        return pd.DataFrame(), ""

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    return df, path


def _normalize_text(value) -> str:
    txt = str(value or "").strip().lower()
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    return " ".join(txt.split())


def _extract_label_value_table(df_raw: pd.DataFrame, max_cols: int = 4) -> pd.DataFrame:
    rows = []
    sub = df_raw.iloc[:, :max_cols].copy()
    for _, row in sub.iterrows():
        vals = row.tolist()
        label = vals[0] if len(vals) > 0 else np.nan
        value = vals[1] if len(vals) > 1 else np.nan
        unit = vals[2] if len(vals) > 2 else np.nan
        note = vals[3] if len(vals) > 3 else np.nan
        if pd.isna(label) or pd.isna(value):
            continue
        rows.append({
            "Concepto": str(label).strip(),
            "Valor": pd.to_numeric(value, errors="coerce") if not isinstance(value, str) else value,
            "Unidad": "" if pd.isna(unit) else str(unit).strip(),
            "Nota": "" if pd.isna(note) else str(note).strip(),
        })
    return pd.DataFrame(rows)


def _parse_eur_mxn_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    header_idx = None
    for idx, row in df_raw.iterrows():
        if any(_normalize_text(v) == "fecha" for v in row.tolist()):
            header_idx = idx
            break
    if header_idx is None:
        return pd.DataFrame(columns=["Fecha", "Valor"])

    sub = df_raw.iloc[header_idx + 1:, :3].copy()
    sub.columns = ["dummy", "Fecha", "Valor"]
    sub["Fecha"] = pd.to_numeric(sub["Fecha"], errors="coerce")
    sub["Valor"] = pd.to_numeric(sub["Valor"], errors="coerce")
    sub = sub.dropna(subset=["Fecha", "Valor"])
    if sub.empty:
        return pd.DataFrame(columns=["Fecha", "Valor"])
    sub["Fecha"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(sub["Fecha"], unit="D")
    return sub[["Fecha", "Valor"]].sort_values("Fecha").reset_index(drop=True)


def _parse_named_series(df_raw: pd.DataFrame, kind: str) -> pd.DataFrame:
    records = []
    for _, row in df_raw.iterrows():
        vals = row.tolist()
        if len(vals) < 2:
            continue
        x_raw, y_raw = vals[0], vals[1]
        if pd.isna(x_raw) or pd.isna(y_raw):
            continue
        y = pd.to_numeric(y_raw, errors="coerce")
        if pd.isna(y):
            continue
        x_txt = str(x_raw).strip()
        if kind == "inflacion":
            dt = pd.to_datetime(x_txt, format="%Y/%m", errors="coerce")
            if pd.isna(dt):
                continue
            y = float(y) / 100.0
        else:
            dt = pd.to_datetime(x_txt, format="%d.%m.%Y", errors="coerce")
            if pd.isna(dt):
                continue
            y = float(y)
            if y > 100:
                y = y / 100000.0
        records.append({"Fecha": dt, "Valor": float(y)})
    return pd.DataFrame(records).sort_values("Fecha").reset_index(drop=True) if records else pd.DataFrame(columns=["Fecha", "Valor"])


def _parse_projection_sheet(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    year_idx = None
    year_cols = []
    years = []
    for idx, row in df_raw.iterrows():
        vals = [pd.to_numeric(v, errors="coerce") for v in row.tolist()]
        yr_positions = [(i, int(v)) for i, v in enumerate(vals) if pd.notna(v) and 2025 <= float(v) <= 2035 and float(v).is_integer()]
        if len(yr_positions) >= 5:
            year_idx = idx
            year_cols = [i for i, _ in yr_positions]
            years = [y for _, y in yr_positions]
            break
    if year_idx is None:
        return pd.DataFrame(), {}

    rows = []
    for _, row in df_raw.iloc[year_idx + 1:].iterrows():
        concept = row.iloc[0] if len(row) > 0 else np.nan
        if pd.isna(concept):
            continue
        concept = str(concept).strip()
        vals = []
        for col in year_cols:
            vals.append(pd.to_numeric(row.iloc[col], errors="coerce") if col < len(row) else np.nan)
        if all(pd.isna(v) for v in vals):
            continue
        item = {"Concepto": concept}
        for y, v in zip(years, vals):
            item[y] = float(v) if pd.notna(v) else np.nan
        rows.append(item)

    meta = {}
    try:
        for idx, row in df_raw.iterrows():
            row_norm = [_normalize_text(v) for v in row.tolist()]
            if "unidades a comprar" in row_norm:
                pos = row_norm.index("unidades a comprar")
                val = pd.to_numeric(row.iloc[pos + 1], errors="coerce") if pos + 1 < len(row) else np.nan
                if pd.notna(val):
                    meta["unidades_a_comprar"] = int(val)
                break
    except Exception:
        pass

    out = pd.DataFrame(rows)
    return out, meta


@st.cache_data(show_spinner=False)
def cargar_crl(filename: str) -> tuple[dict, str]:
    path = find_file(filename)
    if path is None:
        return {}, ""

    xls = pd.ExcelFile(path)
    raw = {sheet: pd.read_excel(path, sheet_name=sheet, header=None) for sheet in xls.sheet_names}

    sheet_map = {_normalize_text(s): s for s in xls.sheet_names}

    def by_token(token: str) -> str | None:
        for key, val in sheet_map.items():
            if token in key:
                return val
        return None

    supuestos = _extract_label_value_table(raw.get(by_token("supuestos"), pd.DataFrame()))
    clcircular = _extract_label_value_table(raw.get(by_token("datos proporcionados"), pd.DataFrame()), max_cols=4)
    eur_mxn = _parse_eur_mxn_sheet(raw.get(by_token("eur a mxn"), pd.DataFrame()))
    bono = _parse_named_series(raw.get(by_token("bono"), pd.DataFrame()), "bono")
    inflacion = _parse_named_series(raw.get(by_token("inflacion"), pd.DataFrame()), "inflacion")

    escenarios = {}
    meta = {}
    for token, label in [
        ("sin financiamiento", "Sin financiamiento"),
        ("con finacimiento optimista", "Con financiamiento optimista"),
        ("con finacimiento pesimista", "Con financiamiento pesimista"),
        ("con finacimiento", "Con financiamiento base"),
    ]:
        sheet = by_token(token)
        if not sheet:
            continue
        parsed, m = _parse_projection_sheet(raw[sheet])
        if not parsed.empty:
            escenarios[label] = parsed
        if m:
            meta[label] = m

    return {
        "supuestos": supuestos,
        "clcircular": clcircular,
        "eur_mxn": eur_mxn,
        "bono": bono,
        "inflacion": inflacion,
        "escenarios": escenarios,
        "meta": meta,
    }, path


def _dominant_risk_component(df: pd.DataFrame) -> str:
    means = {
        "Contaminacion": float(df.get("perdida_contaminacion_usd", pd.Series([0])).mean()),
        "Regulatorio": float(df.get("perdida_regulatoria_usd", pd.Series([0])).mean()),
        "Retraso aduanal": float(df.get("costo_retraso_usd", pd.Series([0])).mean()),
    }
    return max(means, key=means.get)


def _sim_irc_level(x: float) -> str:
    if x <= 20:
        return "Bajo"
    if x <= 40:
        return "Moderado"
    if x <= 60:
        return "Alto"
    if x <= 80:
        return "Muy_Alto"
    return "Critico"


def run_montecarlo(
    sector: str,
    n: int,
    p_contam_a: float,
    p_contam_b: float,
    p_reg_a: float,
    p_reg_b: float,
    valor_mu: float,
    base_df_sector: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:
    """Ejecuta simulacion Monte Carlo y devuelve columnas compatibles con CSV base."""
    rng = np.random.default_rng(seed)
    n = int(n)
    if n <= 0:
        return pd.DataFrame()

    base = base_df_sector.copy()
    if base.empty:
        return pd.DataFrame()

    # Muestras base para preservar rutas, modo, droga y anclas CBP.
    sampled = base.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    # Costos por sector (lognormal) tomados de calibracion.
    if sector == "Cerveza":
        cc_mu, cc_sigma = 11.8, 1.1
        cr_mu, cr_sigma = 12.5, 1.0
        valor_sigma = 15000.0
        penal_dia = 5000.0
    elif sector == "Autopartes":
        cc_mu, cc_sigma = 13.0, 1.2
        cr_mu, cr_sigma = 14.0, 1.2
        valor_sigma = 80000.0
        penal_dia = 50000.0
    else:
        cc_mu, cc_sigma = 11.0, 0.9
        cr_mu, cr_sigma = 11.0, 0.8
        valor_sigma = 8000.0
        penal_dia = 3000.0

    p_contam = rng.beta(max(p_contam_a, 0.001), max(p_contam_b, 0.001), size=n)
    e_contam = rng.binomial(1, p_contam, size=n)
    c_contam = rng.lognormal(mean=cc_mu, sigma=cc_sigma, size=n)
    loss_contam = e_contam * c_contam

    p_reg = rng.beta(max(p_reg_a, 0.001), max(p_reg_b, 0.001), size=n)
    e_reg = rng.binomial(1, p_reg, size=n)
    c_reg = rng.lognormal(mean=cr_mu, sigma=cr_sigma, size=n)
    loss_reg = e_reg * c_reg

    scen_probs = sampled["escenario_aduanal"].value_counts(normalize=True)
    scen_values = scen_probs.index.to_list()
    scen_p = scen_probs.values
    escenario = rng.choice(scen_values, size=n, p=scen_p)
    dias = np.where(
        escenario == "retencion", rng.integers(3, 11, size=n),
        np.where(escenario == "secundaria", rng.integers(1, 5, size=n), rng.integers(0, 2, size=n))
    )
    costo_retraso = dias * penal_dia

    valor = rng.normal(loc=max(valor_mu, 1000.0), scale=max(valor_sigma, 1000.0), size=n)
    valor = np.clip(valor, 1000, None)

    perdida_total = loss_contam + loss_reg + costo_retraso
    irc_raw = perdida_total / valor
    p995 = float(np.percentile(base.get("irc_raw", pd.Series(irc_raw)), 99.5))
    p995 = max(p995, 1e-6)
    irc_norm = np.clip((irc_raw / p995) * 100, 0, 100)

    out = sampled.copy()
    out["iteracion"] = np.arange(1, n + 1)
    out["sector"] = sector
    out["prob_contaminacion"] = p_contam
    out["evento_contaminacion"] = e_contam
    out["costo_contaminacion_usd"] = c_contam
    out["perdida_contaminacion_usd"] = loss_contam
    out["prob_rechazo_regulatorio"] = p_reg
    out["evento_regulatorio"] = e_reg
    out["costo_regulatorio_usd"] = c_reg
    out["perdida_regulatoria_usd"] = loss_reg
    out["escenario_aduanal"] = escenario
    out["dias_retraso"] = dias
    out["penalizacion_dia_usd"] = penal_dia
    out["costo_retraso_usd"] = costo_retraso
    out["valor_carga_usd"] = valor
    out["perdida_total_usd"] = perdida_total
    out["irc_raw"] = irc_raw
    out["irc_normalizado"] = irc_norm
    out["nivel_riesgo"] = pd.Series(irc_norm).map(_sim_irc_level).values
    return out


def obtener_contactos_sector(df_contactos: pd.DataFrame, sector_key: str, top_n: int = 30) -> pd.DataFrame:
    """Filtrar y ordenar contactos por sector del dashboard."""
    if df_contactos.empty:
        return pd.DataFrame()

    target = SECTOR_CONTACT_MAP.get(sector_key, sector_key)
    out = df_contactos[df_contactos["sector"] == target].copy()
    if out.empty:
        return out

    out = out.sort_values(
        by=["relev_rank", "viajes_num", "empresa"],
        ascending=[False, False, True]
    ).head(top_n)
    return out.reset_index(drop=True)


def _norm_state_name(value: str) -> str:
    """Normaliza nombre de estado para matching de coordenadas."""
    txt = str(value or "").strip().lower()
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    txt = txt.replace(".", " ").replace("-", " ")
    txt = " ".join(txt.split())
    return txt


def _serie_mensual_sector(df_sector: pd.DataFrame) -> pd.Series:
    """Serie mensual FOB (MUSD) con frecuencia MS."""
    s = (
        df_sector.groupby("fecha")["fob_millon"]
        .sum()
        .sort_index()
    )
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("MS").fillna(0.0)
    return s.astype(float)


def _fit_best_sarima(series: pd.Series):
    """Buscar SARIMA con mejor AIC en una grilla acotada."""
    best_model = None
    best_cfg = None
    best_aic = np.inf

    pdq_grid = list(product([0, 1, 2], [1], [0, 1, 2]))
    seasonal_grid = list(product([0, 1], [1], [0, 1], [12]))

    for order in pdq_grid:
        for seasonal_order in seasonal_grid:
            try:
                model = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=200)
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic = fit.aic
                    best_model = fit
                    best_cfg = (order, seasonal_order)
            except Exception:
                continue

    return best_model, best_cfg, best_aic


@st.cache_data(show_spinner=False)
def sarima_forecast_2026(dfs_all: dict):
    """Forecast mensual 2026 para todos los sectores disponibles."""
    if not SARIMA_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame()

    forecast_rows = []
    model_rows = []

    for sector, df in dfs_all.items():
        if "fecha" not in df.columns or "fob_millon" not in df.columns:
            continue

        serie = _serie_mensual_sector(df[df["fecha"] <= pd.Timestamp("2025-12-01")])
        if len(serie) < 24:
            continue

        fit, cfg, best_aic = _fit_best_sarima(serie)
        if fit is None or cfg is None:
            continue

        pred = fit.get_forecast(steps=12)
        mean_fc = pred.predicted_mean
        conf = pred.conf_int(alpha=0.20)
        conf_cols = conf.columns.tolist()
        low_col = conf_cols[0]
        high_col = conf_cols[1]

        for dt, yhat, lo, hi in zip(mean_fc.index, mean_fc.values, conf[low_col].values, conf[high_col].values):
            forecast_rows.append({
                "sector": sector,
                "fecha": pd.to_datetime(dt),
                "forecast_musd": float(yhat),
                "low80_musd": float(lo),
                "high80_musd": float(hi),
            })

        model_rows.append({
            "sector": sector,
            "order": str(cfg[0]),
            "seasonal_order": str(cfg[1]),
            "aic": float(best_aic),
            "forecast_2026_total_musd": float(mean_fc.sum()),
        })

    return pd.DataFrame(forecast_rows), pd.DataFrame(model_rows)


@st.cache_data(show_spinner=False)
def sarima_forecast_dinamico(dfs_input: dict, factor_label: str):
    """
    SARIMA que reacciona al filtro de factor de mercado.
    - 'Todos': serie mensual completa sin gaps.
    - Otro filtro: construye la serie desde los meses filtrados y rellena
      los gaps con interpolacion lineal para mantener continuidad de la serie.
    """
    if not SARIMA_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame()

    DATE_MAX = pd.Timestamp("2025-12-01")
    forecast_rows = []
    model_rows = []

    for sector, df in dfs_input.items():
        if "fecha" not in df.columns or "fob_millon" not in df.columns or df.empty:
            continue

        s = (
            df.groupby("fecha")["fob_millon"]
            .sum()
            .sort_index()
        )
        s.index = pd.to_datetime(s.index)
        s = s[s.index <= DATE_MAX]

        if s.empty:
            continue

        full_idx = pd.date_range(s.index.min(), DATE_MAX, freq="MS")
        serie = (
            s.reindex(full_idx)
            .interpolate(method="linear")
            .bfill()
            .ffill()
            .clip(lower=0)
            .astype(float)
        )

        if len(serie) < 24:
            continue

        fit, cfg, best_aic = _fit_best_sarima(serie)
        if fit is None or cfg is None:
            continue

        pred = fit.get_forecast(steps=12)
        mean_fc = pred.predicted_mean
        conf = pred.conf_int(alpha=0.20)
        low_col, high_col = conf.columns[0], conf.columns[1]

        for dt, yhat, lo, hi in zip(
            mean_fc.index, mean_fc.values,
            conf[low_col].values, conf[high_col].values
        ):
            forecast_rows.append({
                "sector": sector,
                "fecha":         pd.to_datetime(dt),
                "forecast_musd": float(max(yhat, 0)),
                "low80_musd":    float(max(lo,   0)),
                "high80_musd":   float(max(hi,   0)),
            })

        model_rows.append({
            "sector":          sector,
            "order":           str(cfg[0]),
            "seasonal_order":  str(cfg[1]),
            "aic":             float(best_aic),
            "forecast_2026_total_musd": float(max(mean_fc.sum(), 0)),
        })

    df_fc = pd.DataFrame(forecast_rows)
    df_mod = pd.DataFrame(model_rows)
    if not df_fc.empty:
        df_fc = df_fc.sort_values(["sector", "fecha"]).reset_index(drop=True)
    return df_fc, df_mod


def _construir_serie_filtrada(df_sector: pd.DataFrame) -> pd.Series:
    """Serie mensual interpolada desde datos filtrados por factor."""
    DATE_MAX = pd.Timestamp("2025-12-01")
    s = df_sector.groupby("fecha")["fob_millon"].sum().sort_index()
    s.index = pd.to_datetime(s.index)
    s = s[s.index <= DATE_MAX]
    if s.empty:
        return pd.Series(dtype=float)
    full_idx = pd.date_range(s.index.min(), DATE_MAX, freq="MS")
    return (
        s.reindex(full_idx)
        .interpolate(method="linear")
        .bfill()
        .ffill()
        .clip(lower=0)
        .astype(float)
    )


def _simulate_sector_montecarlo(
    serie_mensual: pd.Series,
    n_sims: int,
    seed: int,
    vol_multiplier: float,
    drift_adjust: float,
    shock_prob: float,
    shock_severity: float,
) -> dict:
    """Simula 12 meses usando retornos lognormales con choques discretos."""
    serie = serie_mensual.dropna().astype(float)
    if len(serie) < 24:
        return {}

    rets = np.log(serie / serie.shift(1)).dropna()
    if rets.empty:
        return {}

    mu = float(rets.mean()) + float(drift_adjust)
    sigma = max(float(rets.std()), 1e-6) * float(vol_multiplier)
    base_level = float(serie.iloc[-1])
    base_annual = float(serie.iloc[-12:].sum())

    rng = np.random.default_rng(seed)
    monthly = rng.normal(loc=mu, scale=sigma, size=(n_sims, 12))
    if shock_prob > 0 and shock_severity > 0:
        shocks = rng.binomial(1, shock_prob, size=(n_sims, 12))
        monthly = monthly - shocks * shock_severity

    paths = base_level * np.exp(np.cumsum(monthly, axis=1))
    annual = paths.sum(axis=1)
    ret_annual = (annual / base_annual) - 1.0

    p_down = float(np.mean(ret_annual < 0))
    var5 = float(np.percentile(ret_annual, 5))
    tail = ret_annual[ret_annual <= var5]
    cvar5 = float(tail.mean()) if len(tail) > 0 else var5
    exp_ret = float(ret_annual.mean())
    vol_ret = float(ret_annual.std())

    # Indice compuesto 0-100 (mayor = mayor riesgo)
    comp_prob = np.clip(p_down / 0.80, 0, 1)
    comp_var = np.clip(abs(min(var5, 0)) / 0.35, 0, 1)
    comp_cvar = np.clip(abs(min(cvar5, 0)) / 0.45, 0, 1)
    comp_vol = np.clip(vol_ret / 0.30, 0, 1)
    risk_index = float(100 * (0.35 * comp_prob + 0.30 * comp_var + 0.20 * comp_cvar + 0.15 * comp_vol))

    return {
        "mu": mu,
        "sigma": sigma,
        "base_annual": base_annual,
        "expected_annual_musd": float(np.mean(annual)),
        "p_down": p_down,
        "var5": var5,
        "cvar5": cvar5,
        "exp_ret": exp_ret,
        "vol_ret": vol_ret,
        "risk_index": risk_index,
        "annual_samples": annual,
        "ret_samples": ret_annual,
    }


@st.cache_data(show_spinner=False)
def montecarlo_sector_risk(
    dfs_all: dict,
    n_sims: int,
    seed: int,
    vol_multiplier: float,
    drift_adjust: float,
    shock_prob: float,
    shock_severity: float,
):
    """Calcula indice de riesgo Monte Carlo por sector."""
    rows = []
    samples = {}

    for sector, df in dfs_all.items():
        if "fecha" not in df.columns or "fob_millon" not in df.columns:
            continue
        serie = _serie_mensual_sector(df[df["fecha"] <= pd.Timestamp("2025-12-01")])
        res = _simulate_sector_montecarlo(
            serie_mensual=serie,
            n_sims=n_sims,
            seed=seed + abs(hash(sector)) % 10000,
            vol_multiplier=vol_multiplier,
            drift_adjust=drift_adjust,
            shock_prob=shock_prob,
            shock_severity=shock_severity,
        )
        if not res:
            continue

        idx = res["risk_index"]
        if idx < 33:
            level = "Bajo"
        elif idx < 66:
            level = "Medio"
        else:
            level = "Alto"

        rows.append({
            "sector": sector,
            "risk_index": idx,
            "risk_level": level,
            "prob_caida": res["p_down"],
            "var5": res["var5"],
            "cvar5": res["cvar5"],
            "ret_esperado": res["exp_ret"],
            "vol_retorno": res["vol_ret"],
            "base_2025_musd": res["base_annual"],
            "forecast_esperado_2026_musd": res["expected_annual_musd"],
        })
        samples[sector] = res["annual_samples"]

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("risk_index", ascending=False).reset_index(drop=True)
    return out, samples


@st.cache_data(show_spinner=False)
def construir_panel(dfs: dict) -> pd.DataFrame:
    """Panel anual con features para clustering."""
    rows = []
    for name, df in dfs.items():
        anual = df[df['year'] <= 2024].groupby('year').agg(
            fob_millon=('fob_millon', 'sum'),
            peso_mt=('netWgt', lambda x: x.sum() / 1e6),
            precio_prom=('precio_unitario', 'mean'),
        ).reset_index()
        anual['sector'] = name
        anual['yoy'] = anual['fob_millon'].pct_change() * 100
        rows.append(anual)
    if not rows:
        return pd.DataFrame(columns=['year', 'fob_millon', 'peso_mt', 'precio_prom', 'sector', 'yoy'])
    df_panel = pd.concat(rows, ignore_index=True).dropna(subset=['fob_millon'])
    df_panel['yoy'] = df_panel['yoy'].fillna(0)
    df_panel['precio_prom'] = df_panel['precio_prom'].fillna(0)
    return df_panel


@st.cache_data(show_spinner=False)
def cargar_volumen(files_tuple: tuple) -> dict:
    """Cargar datos de volumen (netWgt/qty) de los archivos nuevos por sector."""
    files_dict = dict(files_tuple)
    dfs = {}
    for sector, fname in files_dict.items():
        path = find_file(fname)
        if path is None:
            continue
        try:
            df = pd.read_excel(path, sheet_name=0)
            needed = ['refYear', 'refMonth', 'netWgt', 'qty', 'fobvalue']
            for col in needed:
                if col not in df.columns:
                    df[col] = 0
            df = df[needed].copy()
            df.columns = ['year', 'month', 'netWgt', 'qty', 'fobvalue']
            df['year']     = pd.to_numeric(df['year'], errors='coerce')
            df['month']    = pd.to_numeric(df['month'], errors='coerce')
            df['netWgt']   = pd.to_numeric(df['netWgt'], errors='coerce').fillna(0)
            df['qty']      = pd.to_numeric(df['qty'], errors='coerce').fillna(0)
            df['fobvalue'] = pd.to_numeric(df['fobvalue'], errors='coerce').fillna(0)
            df = df.dropna(subset=['year', 'month'])
            df['fecha']    = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))
            df['sector']   = sector
            cap = CONTAINER_CAPACITY.get(sector, 20_000)
            df['viajes']   = (df['netWgt'] / cap).clip(lower=0).round(0)
            df['netWgt_M'] = df['netWgt'] / 1_000_000   # millones de unidades
            dfs[sector] = df.sort_values('fecha').reset_index(drop=True)
        except Exception:
            continue
    return dfs


@st.cache_data(show_spinner=False)
def run_clustering(df_panel: pd.DataFrame, k: int) -> tuple:
    """Ejecutar K-Means + PCA."""
    if df_panel.empty:
        raise ValueError("No hay datos suficientes para clustering.")

    feats = ['fob_millon', 'yoy', 'precio_prom', 'peso_mt']
    X = df_panel[feats].values
    n_samples = len(df_panel)
    if n_samples < 3:
        raise ValueError("Se requieren al menos 3 observaciones para K-Means y silhouette.")

    k_eff = min(max(2, k), n_samples - 1)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    km = KMeans(n_clusters=k_eff, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    sil = silhouette_score(X_sc, labels)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    df_out = df_panel.copy()
    df_out['cluster'] = labels.astype(str)
    df_out['PC1'] = X_pca[:, 0]
    df_out['PC2'] = X_pca[:, 1]

    # Metricas de seleccion
    metrics = []
    max_k_for_metrics = min(7, n_samples - 1)
    for ki in range(2, max_k_for_metrics + 1):
        km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
        li = km_i.fit_predict(X_sc)
        metrics.append({
            'k': ki,
            'inertia': km_i.inertia_,
            'silhouette': silhouette_score(X_sc, li)
        })

    return df_out, pca, sil, pd.DataFrame(metrics), k_eff


# 
# SIDEBAR
# 
with st.sidebar:
    _logo_path = find_file("CLC Logo.webp")
    if _logo_path:
        st.image(_logo_path, use_container_width=True)
    else:
        st.markdown("## CL Circular")
    st.markdown("**Plataforma Analitica de Mercado**")
    st.markdown("---")

    st.markdown("#### Configuracion")
    sectores_sel = st.multiselect(
        "Sectores a analizar",
        options=list(FILES.keys()),
        default=list(FILES.keys()),
        format_func=lambda x: LABELS[x],
    )

    year_range = st.slider(
        "Rango de años",
        min_value=2021, max_value=2025,
        value=(2021, 2024), step=1
    )

    k_clusters = st.slider(
        "Numero de clusters (K-Means)",
        min_value=2, max_value=6,
        value=3, step=1
    )

    factor_filter = st.selectbox(
        "Factor de escenario",
        options=["Todos", "Positivo", "Neutral", "Negativo"],
        index=0,
        help=(
            "Estandar aplicado: Positivo >= +5% YoY mensual, "
            "Neutral entre -5% y +5%, Negativo <= -5%."
        ),
    )
    with st.expander("Como funciona el selector de escenario"):
        st.markdown(
            "**En la tab Proyecciones** filtra los periodos historicos usados para calibrar la simulacion:\n"
            "- `Positivo`: solo años con crecimiento YoY >= +5% (expansion)\n"
            "- `Neutral`: años con YoY entre -5% y +5% (estabilidad)\n"
            "- `Negativo`: años con YoY <= -5% (contraccion)\n"
            "- `Todos`: usa toda la historia 2021-2025 (recomendado)"
        )

    st.markdown("---")
    st.markdown("####  Archivos requeridos")
    for name, fname in FILES.items():
        found = find_file(fname) is not None
        icon = "[OK]" if found else "[X]"
        st.markdown(f"{icon} `{fname}`")
    contacts_found = find_file(PROSPECTS_FILE) is not None
    contacts_icon = "[OK]" if contacts_found else "[X]"
    st.markdown(f"{contacts_icon} `{PROSPECTS_FILE}`")
    crl_found = find_file(CRL_FILE) is not None
    crl_icon = "[OK]" if crl_found else "[X]"
    st.markdown(f"{crl_icon} `{CRL_FILE}`")
    risk_found = any((p if os.path.isabs(p) else find_file(p)) for p in RISK_BASE_FILES)
    risk_icon = "[OK]" if risk_found else "[X]"
    st.markdown(f"{risk_icon} `montecarlo_riesgo_clcircular.csv`")

    st.markdown("---")
    st.markdown(
        '<div style="color:#1A3355;font-size:11px">Reto CL Circular<br>Tecnologico de Monterrey<br>Feb 2026</div>',
        unsafe_allow_html=True
    )

# 
# CARGA
# 
with st.spinner("Cargando y procesando datos..."):
    dfs_all = cargar_datos(FILES)
    dfs_vol = cargar_volumen(tuple(VOLUME_FILES.items()))
    df_contactos_all = cargar_contactos(PROSPECTS_FILE)
    crl_data, crl_path = cargar_crl(CRL_FILE)
    df_risk_base, risk_path = cargar_riesgo_base(tuple(RISK_BASE_FILES))

if not dfs_all:
    st.error("No se pudo cargar ningun dataset. Verifica que los archivos .xlsx estan disponibles.")
    st.stop()

if not sectores_sel:
    st.warning("Selecciona al menos un sector en la barra lateral para continuar.")
    st.stop()

dfs = {k: v for k, v in dfs_all.items() if k in sectores_sel}
if not dfs:
    st.error("Los sectores seleccionados no tienen datos disponibles.")
    st.stop()

# dfs_base: solo filtrado por sector, nunca por factor_filter
# Se usa en Evolucion & Tendencias para que los datos historicos sean siempre completos
dfs_base = {k: v for k, v in dfs_all.items() if k in sectores_sel}

if factor_filter != "Todos":
    dfs = {
        k: v[v["factor_mercado"] == factor_filter].copy()
        for k, v in dfs.items()
    }
    dfs = {k: v for k, v in dfs.items() if not v.empty}
    if not dfs:
        st.warning("El filtro de factor no dejo registros para los sectores seleccionados.")
        st.stop()

df_panel = construir_panel(dfs)
try:
    df_clusters, pca_model, sil_score, df_metrics, k_clusters_eff = run_clustering(df_panel, k_clusters)
except ValueError as e:
    st.error(str(e))
    st.stop()

if k_clusters_eff != k_clusters:
    st.info(f"K ajustado automaticamente de {k_clusters} a {k_clusters_eff} por tamaño de muestra.")

# 
# HEADER
# 
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("# CL Circular - Analisis de Mercado")
    st.markdown(
        "**Exportaciones Mexico -> EE.UU.** &nbsp;|&nbsp; "
        "Identificacion de sectores con potencial para economia circular",
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="info-box"><b>Objetivo:</b> evaluar la penetracion de CL Circular en el mercado mexicaño usando unicamente los datos ya integrados en el proyecto: comercio exterior, prospectos comerciales, indice de riesgo y modelo financiero.</div>',
        unsafe_allow_html=True,
    )
    if factor_filter != "Todos":
        st.caption(
            f"Filtro activo: factor `{factor_filter}` "
            "(Positivo >= +5% YoY, Neutral [-5%, +5%], Negativo <= -5%)."
        )
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:#EAF3FB;border:1px solid #B8D0E8;border-radius:8px;padding:10px 16px;text-align:center;">'
        f'<div style="color:#1565A0;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;font-weight:600">Silhouette Score</div>'
        f'<div style="color:#0D1F2D;font-size:28px;font-weight:700;font-family:DM Mono">{sil_score:.3f}</div>'
        f'<div style="color:#4A607A;font-size:10px">k={k_clusters_eff} clusters</div></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# 
# KPIs PRINCIPALES
# 
kpi_cols = st.columns(len(dfs_base))
for i, (name, df_full) in enumerate(dfs_base.items()):
    # Siempre datos completos (sin filtro de factor) para KPIs globales
    df_yr = df_full[(df_full['year'] >= year_range[0]) & (df_full['year'] <= year_range[1])]
    total = df_yr['fob_millon'].sum()
    # CAGR sobre años completos (excluye 2025 si está incompleto)
    anual = (
        df_full[df_full['year'].isin(range(year_range[0], min(year_range[1] + 1, 2025)))]
        .groupby('year')['fob_millon'].sum()
        .sort_index()
    )
    if len(anual) >= 2 and anual.iloc[0] > 0:
        cagr = ((anual.iloc[-1] / anual.iloc[0]) ** (1 / (len(anual) - 1)) - 1) * 100
        delta_str = f"CAGR {cagr:+.1f}%"
    else:
        delta_str = None
    with kpi_cols[i]:
        st.metric(
            label=LABELS[name].split(" (")[0],
            value=f"${total:,.0f}M",
            delta=delta_str,
        )

st.markdown("<br>", unsafe_allow_html=True)

# 
# TABS
# 
tab_exec, tab1, tab_vol, tab_analisis, tab4, tab_fin, tab_comercial, tab_prospectos = st.tabs([
    "Resumen Ejecutivo",
    "Evolucion y Tendencias",
    "Volumen & Sensores",
    "Analisis",
    "Proyecciones",
    "Finanzas CRL",
    "Propuesta Comercial",
    "Prospectos B2B",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────
with tab_exec:

    # ── Estilos internos del tab ──────────────────────────────────────────────
    st.markdown("""
    <style>
    .exec-card {
        background: #F7FAFD;
        border: 1px solid #D0DCE8;
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
    }
    .exec-card-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #1565A0;
        margin-bottom: 6px;
    }
    .exec-card-value {
        font-size: 28px;
        font-weight: 700;
        color: #0D1F2D;
        font-family: 'DM Mono', monospace;
        line-height: 1.1;
    }
    .exec-card-sub {
        font-size: 11px;
        color: #4A607A;
        margin-top: 4px;
    }
    .sector-card {
        border-radius: 10px;
        padding: 20px;
        height: 100%;
    }
    .client-card {
        background: #F7FAFD;
        border: 1px solid #D0DCE8;
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
    }
    .client-logo {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .client-stat {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #E8EFF6;
        padding: 5px 0;
        font-size: 12px;
        color: #1A2B3C;
    }
    .client-stat-val {
        font-weight: 700;
        color: #1565A0;
        font-family: 'DM Mono', monospace;
    }
    .pain-tag {
        display: inline-block;
        background: #FFF3E0;
        color: #E65100;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px 2px 2px 0;
    }
    .clc-tag {
        display: inline-block;
        background: #E3F2FD;
        color: #1565A0;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px 2px 2px 0;
    }
    .roadmap-cell-short  { background:#E8F5E9; color:#1B5E20; border-radius:6px; padding:10px 14px; font-size:12px; }
    .roadmap-cell-medium { background:#E3F2FD; color:#0D47A1; border-radius:6px; padding:10px 14px; font-size:12px; }
    .roadmap-cell-long   { background:#EDE7F6; color:#311B92; border-radius:6px; padding:10px 14px; font-size:12px; }
    .roadmap-label { font-weight:700; font-size:13px; margin-bottom:6px; }
    .comp-better { color:#1A7A3C; font-weight:700; }
    .comp-worse  { color:#C53030; font-weight:700; }
    .comp-equal  { color:#4A607A; }
    .section-header {
        font-size: 15px;
        font-weight: 700;
        color: #0D1F2D;
        border-left: 4px solid #1565A0;
        padding-left: 10px;
        margin: 28px 0 14px 0;
    }
    .why-now-card {
        background: #EAF3FB;
        border: 1px solid #B8D0E8;
        border-radius: 10px;
        padding: 16px 18px;
    }
    .why-now-icon { font-size: 22px; margin-bottom: 6px; }
    .why-now-title { font-size: 13px; font-weight: 700; color: #1565A0; margin-bottom: 4px; }
    .why-now-body { font-size: 12px; color: #1A2B3C; line-height: 1.5; }
    </style>
    """, unsafe_allow_html=True)

    # ── SECCION 1: KPIs Macro ─────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Objetivo del Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box"><b>Objetivo ejecutivo:</b> priorizar la penetracion de CL Circular en el mercado mexicaño con base en los datos ya cargados en el proyecto: flujos de exportacion, prospectos comerciales, indice de riesgo y modelo financiero.</div>',
        unsafe_allow_html=True,
    )

    auto_fob_2024 = float(dfs_base.get("Autopartes", pd.DataFrame()).query("year == 2024")["fob_millon"].sum()) if "Autopartes" in dfs_base else 0.0
    auto_cagr = np.nan
    if "Autopartes" in dfs_base:
        _auto_anual = (
            dfs_base["Autopartes"][dfs_base["Autopartes"]["year"].isin([2021, 2024])]
            .groupby("year")["fob_millon"].sum()
            .sort_index()
        )
        if len(_auto_anual) == 2 and _auto_anual.iloc[0] > 0:
            auto_cagr = ((_auto_anual.iloc[-1] / _auto_anual.iloc[0]) ** (1 / 3) - 1) * 100
    auto_contacts = 0
    auto_high = 0
    if not df_contactos_all.empty and "sector" in df_contactos_all.columns:
        auto_contacts = int((df_contactos_all["sector"] == "Autopartes").sum())
        if "relev_rank" in df_contactos_all.columns:
            auto_high = int(((df_contactos_all["sector"] == "Autopartes") & (df_contactos_all["relev_rank"] >= 4)).sum())
    auto_irc = np.nan
    if not df_risk_base.empty and {"sector", "irc_normalizado"}.issubset(df_risk_base.columns):
        _auto_risk = df_risk_base[df_risk_base["sector"] == "Autopartes"]["irc_normalizado"]
        if not _auto_risk.empty:
            auto_irc = float(_auto_risk.mean())

    st.markdown('<div class="section-header">Sector Prioritario: Autopartes</div>', unsafe_allow_html=True)
    pr1, pr2, pr3, pr4 = st.columns(4)
    with pr1:
        st.metric("FOB 2024", f"${auto_fob_2024:,.0f}M")
    with pr2:
        st.metric("CAGR 2021-2024", f"{auto_cagr:+.1f}%" if pd.notna(auto_cagr) else "N/D")
    with pr3:
        st.metric("Prospectos", f"{auto_contacts}", delta=f"{auto_high} alta prioridad")
    with pr4:
        st.metric("IRC promedio", f"{auto_irc:.2f}" if pd.notna(auto_irc) else "N/D")
    st.markdown(
        '<div class="info-box"><b>Justificacion del foco:</b> con la informacion disponible, Autopartes combina tamaño de mercado, base comercial accionable y una exposicion operativa relevante. Por eso se toma como eje principal de penetracion en Mexico.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Cadena de Suministro y Dolor Operativo</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(
            '<div class="why-now-card"><div class="why-now-title">Cadena de suministro observada</div><div class="why-now-body">El proyecto analiza exportaciones Mexico → EE.UU. y prospectos del sector autopartes. La entrada comercial se apoya en volumen exportador, ubicacion industrial y acceso a rutas/fronteras ya registradas en la base.</div></div>',
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            '<div class="why-now-card"><div class="why-now-title">Dolor operativo que resuelve CLC</div><div class="why-now-body">Con los datos actuales, el problema se organiza en tres capas: riesgo operativo en ruta (IRC), necesidad de trazabilidad comercial y concentracion de prospectos con mayor intensidad logistica. El sensor se posiciona como herramienta de monitoreo y evidencia durante el trayecto.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Estrategia Comercial y Marketing</div>', unsafe_allow_html=True)
    em1, em2, em3 = st.columns(3)
    with em1:
        st.markdown(
            '<div class="why-now-card"><div class="why-now-title">Corto plazo</div><div class="why-now-body">Prospeccion directa a cuentas de autopartes con prioridad alta, mayor numero de viajes estimados y presencia en estados ya visibles en la tab comercial.</div></div>',
            unsafe_allow_html=True,
        )
    with em2:
        st.markdown(
            '<div class="why-now-card"><div class="why-now-title">Mediaño plazo</div><div class="why-now-body">Escalar por concentracion geografica y subsector. El criterio comercial sigue la distribucion de prospectos y la evidencia de riesgo por sector.</div></div>',
            unsafe_allow_html=True,
        )
    with em3:
        st.markdown(
            '<div class="why-now-card"><div class="why-now-title">Largo plazo</div><div class="why-now-body">Vincular expansion comercial con los escenarios financieros de CRL.xlsx para decidir ritmo de inversion, capacidad operativa y escalamiento con o sin financiamiento.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Contexto de Mercado — Por que ahora</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="exec-card">
            <div class="exec-card-title">Mexico → EE.UU. (2024)</div>
            <div class="exec-card-value">$503B</div>
            <div class="exec-card-sub">#1 proveedor de EE.UU.<br>Supero a China y Canada</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="exec-card">
            <div class="exec-card-title">Autopartes exportadas 2024</div>
            <div class="exec-card-value">$40.3B</div>
            <div class="exec-card-sub">42.9% de importaciones EE.UU.<br>Mexico: 4to exportador mundial</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="exec-card">
            <div class="exec-card-title">Costo downtime automotriz</div>
            <div class="exec-card-value">$2.3M</div>
            <div class="exec-card-sub">por hora de paro de linea<br>$22K–$50K por minuto (Siemens 2024)</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="exec-card">
            <div class="exec-card-title">Mercado IoT Cargo Tracking</div>
            <div class="exec-card-value">$2.9B</div>
            <div class="exec-card-sub">CAGR 10.8% | Proyeccion $4.9B (2029)<br>29M unidades activas en 2025</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECCION 2: Por que ahora ──────────────────────────────────────────────
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
        <div class="why-now-card">
            <div class="why-now-title">Nearshoring en maximo historico</div>
            <div class="why-now-body">
                Mexico desplazo a China como principal proveedor de EE.UU. en 2024.
                Las exportaciones manufactureras crecen +6% anual con proyeccion de
                <b>$700B en 2026</b>. FDI de $43B proyectado en 2025.<br><br>
                Chihuahua, N.L., Coahuila, B.C. y Tamaulipas concentran
                <b>+50% de las exportaciones</b> — exactamente el corredor donde opera CLC.
            </div>
        </div>""", unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="why-now-card">
            <div class="why-now-title">Revision USMCA — 1 julio 2026</div>
            <div class="why-now-body">
                La USITC inicio revision formal de reglas de origen automotriz en feb 2026.
                <b>"Undocumented compliance = non-compliance"</b>: exportadores necesitan
                trazabilidad documentada de cada embarque para mantener tratamiento
                arancelario preferencial (75% contenido regional).<br><br>
                El sensor CLC genera el registro digital de cadena de custodia
                que CBP puede exigir como evidencia.
            </div>
        </div>""", unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="why-now-card">
            <div class="why-now-title">Presion de tarifas Trump 2025</div>
            <div class="why-now-body">
                Tarifas sobre Mexico y Canada generan mayor escrutinio en origen
                de partes y condicion de carga en frontera. Robo de carga en
                Mexico <b>+16% en 2024</b>. Responsabilidad de carriers en Mexico
                limitada a $2 USD/libra.<br><br>
                Los exportadores necesitan documentacion de condicion en transito
                para reclamaciones y cumplimiento aduanal. CLC lo provee.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── SECCION 3: Propuesta de Valor por Sector ──────────────────────────────
    st.markdown('<div class="section-header">Propuesta de Valor — Por Sector</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="sector-card" style="background:#EAF3FB;border:1px solid #B8D0E8;">
            <div style="font-size:13px;font-weight:700;color:#1565A0;margin-bottom:10px;">
                Autopartes (HS 8708)
            </div>
            <div style="font-size:11px;font-weight:700;color:#C53030;margin-bottom:4px;">DOLOR DEL CLIENTE</div>
            <div style="font-size:12px;color:#1A2B3C;margin-bottom:10px;">
                Entrega JIT critica — un camion tarde para una linea de ensamble
                cuesta <b>$22K–$50K por minuto</b>. Componentes electricos y arneses
                sensibles a humedad, temperatura y vibracion. IATF 16949 exige
                trazabilidad documentada de cada lote.
            </div>
            <div style="margin-bottom:8px;">
                <span class="pain-tag">Riesgo regulatorio mas alto (5%)</span>
                <span class="pain-tag">JIT = cero tolerancia</span>
                <span class="pain-tag">Daño por vibración</span>
            </div>
            <div style="font-size:11px;font-weight:700;color:#1565A0;margin-bottom:4px;">SOLUCION CLC</div>
            <div style="font-size:12px;color:#1A2B3C;margin-bottom:8px;">
                Sensor 5G monitorea shock, vibracion, temperatura y humedad en
                tiempo real. Dashboard de trazabilidad = evidencia para IATF,
                USMCA y reclamos con carrier. <b>ROI: 1.3 segundos de linea
                evitada paga el sensor.</b>
            </div>
            <div>
                <span class="clc-tag">Clientes: Lear + Nemak</span>
                <span class="clc-tag">~10,812 viajes/mes</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="sector-card" style="background:#F3F8EE;border:1px solid #C5DDB0;">
            <div style="font-size:13px;font-weight:700;color:#2E7D32;margin-bottom:10px;">
                Cerveza (HS 220300)
            </div>
            <div style="font-size:11px;font-weight:700;color:#C53030;margin-bottom:4px;">DOLOR DEL CLIENTE</div>
            <div style="font-size:12px;color:#1A2B3C;margin-bottom:10px;">
                Temperatura critica 4–8°C durante transporte. Exposicion a luz
                causa "skunking" (oxidacion). Humedad alta deteriora empaque.
                Rechazos en destino implican perdida total del lote
                (~$150K–$300K por camion a precio FOB).
            </div>
            <div style="margin-bottom:8px;">
                <span class="pain-tag">Temperatura controlada 4-8°C</span>
                <span class="pain-tag">Sensible a luz</span>
                <span class="pain-tag">Estacionalidad marcada</span>
            </div>
            <div style="font-size:11px;font-weight:700;color:#2E7D32;margin-bottom:4px;">SOLUCION CLC</div>
            <div style="font-size:12px;color:#1A2B3C;margin-bottom:8px;">
                Sensor BG monitorea temperatura, humedad y luz durante todo el
                trayecto. Alerta en tiempo real si hay desviacion. Registro
                digital = defensa ante rechazo en aduanas EE.UU. o cliente.
            </div>
            <div>
                <span class="clc-tag">Mayor volumen: ~12,838 viajes/mes</span>
                <span class="clc-tag">Pico veraño y Q4</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="sector-card" style="background:#FFF8E1;border:1px solid #FFD54F;">
            <div style="font-size:13px;font-weight:700;color:#F57F17;margin-bottom:10px;">
                Harinas (HS 1101 / 110220)
            </div>
            <div style="font-size:11px;font-weight:700;color:#C53030;margin-bottom:4px;">DOLOR DEL CLIENTE</div>
            <div style="font-size:12px;color:#1A2B3C;margin-bottom:10px;">
                Humedad critica (&lt;14%) — exceso genera moho y micotoxinas.
                Contaminacion cruzada con otros productos en bodega o camion.
                FDA FSMA exige trazabilidad de alimentos importados a EE.UU.
                Probabilidad de contaminacion mas alta del portafolio (Beta ≈ 0.8%).
            </div>
            <div style="margin-bottom:8px;">
                <span class="pain-tag">Humedad critica &lt;14%</span>
                <span class="pain-tag">FSMA FDA</span>
                <span class="pain-tag">Contaminacion cruzada</span>
            </div>
            <div style="font-size:11px;font-weight:700;color:#F57F17;margin-bottom:4px;">SOLUCION CLC</div>
            <div style="font-size:12px;color:#1A2B3C;margin-bottom:8px;">
                Monitor continuo de humedad y temperatura. Log digital como
                soporte a certificado sanitario ante FDA. Complementa la
                capacidad de flota en temporadas bajas de cerveza y autopartes.
            </div>
            <div>
                <span class="clc-tag">Volumen complementario: ~790 viajes/mes</span>
                <span class="clc-tag">Llenar capacidad ociosa</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── SECCION 4: Los 3 Clientes Objetivo ───────────────────────────────────
    st.markdown('<div class="section-header">Los 3 Clientes Objetivo — Datos de Reportes Anuales 2025</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="client-card">
            <div class="client-logo" style="color:#1565A0;">LEAR CORPORATION (NYSE: LEA)</div>
            <div style="font-size:11px;color:#4A607A;margin-bottom:10px;">
                Lider global en Seating y E-Systems | 10-K FY2025
            </div>
            <div class="client-stat"><span>Revenue 2025</span><span class="client-stat-val">~$23.5B</span></div>
            <div class="client-stat"><span>Instalaciones globales</span><span class="client-stat-val">258 en 36 paises</span></div>
            <div class="client-stat"><span>Plantas North America</span><span class="client-stat-val">74 plantas</span></div>
            <div class="client-stat"><span>Market share seating</span><span class="client-stat-val">26% global</span></div>
            <div class="client-stat"><span>Plataformas con contenido Lear</span><span class="client-stat-val">500+</span></div>
            <div class="client-stat"><span>Produccion Mexico</span><span class="client-stat-val">Wire harnesses (primaria)</span></div>
            <div style="margin-top:10px;font-size:11px;color:#1A2B3C;">
                <b>Por que CLC:</b> Arneses de cableado se producen en Mexico con modelo JIT.
                Lear cita "logistics issues" como factor de riesgo en su 10-K.
                USMCA 2026 aumenta presion de trazabilidad. Un camion tarde = linea parada.
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="client-card">
            <div class="client-logo" style="color:#2E7D32;">NEMAK (BMV: NEMAK)</div>
            <div style="font-size:11px;color:#4A607A;margin-bottom:10px;">
                Lider en aluminio ligero para automotriz | Annual Report 2025
            </div>
            <div class="client-stat"><span>Revenue 2025</span><span class="client-stat-val">$4.93B</span></div>
            <div class="client-stat"><span>Revenue North America (53%)</span><span class="client-stat-val">~$2.6B</span></div>
            <div class="client-stat"><span>EBITDA 2025</span><span class="client-stat-val">$591M</span></div>
            <div class="client-stat"><span>Plantas Mexico</span><span class="client-stat-val">Saltillo + Monterrey</span></div>
            <div class="client-stat"><span>Empleados</span><span class="client-stat-val">23,400</span></div>
            <div class="client-stat"><span>Volumen produccion</span><span class="client-stat-val">38.4M unidades eq.</span></div>
            <div style="margin-top:10px;font-size:11px;color:#1A2B3C;">
                <b>Por que CLC:</b> Componentes de aluminio fundido requieren control de
                temperatura y humedad en transporte. OEMs exigen trazabilidad IATF 16949.
                Ruta Saltillo-Laredo es diaria y critica para sus clientes GM/Ford/Stellantis.
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="client-card">
            <div class="client-logo" style="color:#C53030;">DHL SUPPLY CHAIN</div>
            <div style="font-size:11px;color:#4A607A;margin-bottom:10px;">
                Lider mundial en contrato logistico | Annual Report 2025
            </div>
            <div class="client-stat"><span>Revenue Supply Chain 2025</span><span class="client-stat-val">~$19.2B</span></div>
            <div class="client-stat"><span>Revenue Americas (42%)</span><span class="client-stat-val">~$8.0B</span></div>
            <div class="client-stat"><span>Market share global</span><span class="client-stat-val">6.1% (lider)</span></div>
            <div class="client-stat"><span>Empleados Supply Chain</span><span class="client-stat-val">187,000</span></div>
            <div class="client-stat"><span>Warehousing</span><span class="client-stat-val">17.5M m²</span></div>
            <div class="client-stat"><span>Mexico en Strategy 2030</span><span class="client-stat-val">Growth Region</span></div>
            <div style="margin-top:10px;font-size:11px;color:#1A2B3C;">
                <b>Por que CLC:</b> DHL ya tiene contratos con Lear y Nemak en Mexico.
                Un acuerdo B2B con DHL = acceso directo a toda su red de clientes automotrices.
                DHL busca diferenciarse con temperatura controlada e IoT vs CEVA, GXO, Kuehne+Nagel.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── SECCION 5: Benchmark Competidores ────────────────────────────────────
    st.markdown('<div class="section-header">Posicionamiento Competitivo</div>', unsafe_allow_html=True)

    comp_col, roadmap_col = st.columns([1, 1])

    with comp_col:
        st.markdown("""
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <thead>
            <tr style="background:#1565A0;color:white;">
                <th style="padding:8px 10px;text-align:left;">Atributo</th>
                <th style="padding:8px 10px;text-align:center;">CL Circular</th>
                <th style="padding:8px 10px;text-align:center;">Tive Solo 5G</th>
                <th style="padding:8px 10px;text-align:center;">Sensitech</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background:#F7FAFD;">
                <td style="padding:7px 10px;color:#1A2B3C;">Precio / viaje</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">$30 USD</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-equal">~$15–30</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-worse">~$30–80</td>
            </tr>
            <tr>
                <td style="padding:7px 10px;color:#1A2B3C;">Recuperacion activa</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">98%</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-worse">0% (rebate)</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-worse">0% (reciclaje)</td>
            </tr>
            <tr style="background:#F7FAFD;">
                <td style="padding:7px 10px;color:#1A2B3C;">Modelo circular</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-worse">NO</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-worse">NO</td>
            </tr>
            <tr>
                <td style="padding:7px 10px;color:#1A2B3C;">Temperatura</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
            </tr>
            <tr style="background:#F7FAFD;">
                <td style="padding:7px 10px;color:#1A2B3C;">Shock / Vibracion</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
            </tr>
            <tr>
                <td style="padding:7px 10px;color:#1A2B3C;">GPS tiempo real</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI (5G)</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI (5G)</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
            </tr>
            <tr style="background:#F7FAFD;">
                <td style="padding:7px 10px;color:#1A2B3C;">Especializ. Mexico-EE.UU.</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-worse">NO</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-worse">NO</td>
            </tr>
            <tr>
                <td style="padding:7px 10px;color:#1A2B3C;">Dashboard trazabilidad</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
                <td style="padding:7px 10px;text-align:center;" class="comp-better">SI</td>
            </tr>
        </tbody>
        </table>
        <div style="font-size:10px;color:#4A607A;margin-top:6px;">
            Fuentes: Tive.com | Sensitech.com | IoT M2M Council 2024 | CL Circular
        </div>
        """, unsafe_allow_html=True)

    # ── SECCION 6: Roadmap ────────────────────────────────────────────────────
    with roadmap_col:
        st.markdown("""
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <thead>
            <tr style="background:#1565A0;color:white;">
                <th style="padding:8px 10px;text-align:left;width:30%;">Horizonte</th>
                <th style="padding:8px 10px;text-align:left;">Acciones y Metas</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding:10px;vertical-align:top;">
                    <div style="background:#E8F5E9;color:#1B5E20;border-radius:6px;padding:8px 10px;">
                        <div style="font-weight:700;font-size:13px;">Corto Plazo</div>
                        <div style="font-size:11px;">0 – 6 meses</div>
                    </div>
                </td>
                <td style="padding:10px;font-size:12px;color:#1A2B3C;vertical-align:top;">
                    <b>Piloto Nemak Saltillo:</b> 100 sensores en ruta Monterrey–Laredo<br>
                    <b>Trial DHL Supply Chain MX:</b> rutas de autopartes existentes<br>
                    <b>Certificacion USMCA:</b> posicionar log CLC como evidencia CBP<br>
                    <b>Meta:</b> 500 viajes/mes → <b>$15,000 USD/mes</b>
                </td>
            </tr>
            <tr style="background:#F7FAFD;">
                <td style="padding:10px;vertical-align:top;">
                    <div style="background:#E3F2FD;color:#0D47A1;border-radius:6px;padding:8px 10px;">
                        <div style="font-weight:700;font-size:13px;">Mediaño Plazo</div>
                        <div style="font-size:11px;">6 – 18 meses</div>
                    </div>
                </td>
                <td style="padding:10px;font-size:12px;color:#1A2B3C;vertical-align:top;">
                    <b>Escalar con Lear MX:</b> wire harnesses en 16 plantas E-Systems<br>
                    <b>Expansion Cerveza:</b> capturar pico estacional (veraño / Q4)<br>
                    <b>Contrato marco DHL:</b> CLC como servicio estandar en rutas MX<br>
                    <b>Meta:</b> 2,000 viajes/mes → <b>$60,000 USD/mes</b>
                </td>
            </tr>
            <tr>
                <td style="padding:10px;vertical-align:top;">
                    <div style="background:#EDE7F6;color:#311B92;border-radius:6px;padding:8px 10px;">
                        <div style="font-weight:700;font-size:13px;">Largo Plazo</div>
                        <div style="font-size:11px;">18 – 36 meses</div>
                    </div>
                </td>
                <td style="padding:10px;font-size:12px;color:#1A2B3C;vertical-align:top;">
                    <b>Partnership DHL:</b> canal B2B para toda su red de clientes automotrices<br>
                    <b>Expansion flota 2x:</b> 34,000 sensores → 8,500 viajes/mes<br>
                    <b>Nuevo corredor:</b> rutas internas Mexico (post-nearshoring)<br>
                    <b>Meta:</b> 8,500 viajes/mes → <b>$255,000 USD/mes</b>
                </td>
            </tr>
        </tbody>
        </table>
        <div style="font-size:10px;color:#4A607A;margin-top:6px;">
            Capacidad actual: 4,250 viajes/mes (17,000 sensores x 3 usos/año ÷ 12)
        </div>
        """, unsafe_allow_html=True)

    # ── Fuentes ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Fuentes y referencias"):
        st.markdown("""
        | Dato | Fuente | Año |
        |------|--------|-----|
        | Costo downtime automotriz $2.3M/hr | Siemens True Cost of Downtime | 2024 |
        | Mexico #1 proveedor EE.UU. $503B | Mexico Business News / COMCE | 2024 |
        | Autopartes Mexico 42.9% mercado EE.UU. | INA / Co-Production International | 2024 |
        | Mercado IoT cargo tracking $2.9B | IoT M2M Council / SCMR | 2024 |
        | Revision USMCA 1 julio 2026 | USITC / Federal Register | 2026 |
        | Robo de carga Mexico +16% | BTS / Landline Media | 2024 |
        | Tive: modelo single-use sin recuperacion activa | Tive.com | 2025 |
        | Lear: wire harnesses en Mexico, riesgo logistico | Lear 10-K FY2025 | 2025 |
        | Nemak: Revenue $4.93B, 53% North America | Nemak Annual Report 2025 | 2025 |
        | DHL: Mexico como Growth Region, Revenue SC ~$19.2B | DHL Annual Report 2025 | 2025 |
        """)

#
# TAB 1: EVOLUCIN & TENDENCIAS
#
with tab1:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("### Exportaciones FOB Anuales")
        fig_ev = go.Figure()
        for name, df in dfs_base.items():
            anual = df[
                (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
            ].groupby('year')['fob_millon'].sum()
            fig_ev.add_trace(go.Scatter(
                x=anual.index, y=anual.values,
                name=LABELS[name].split(" (")[0],
                mode='lines+markers',
                line=dict(color=COLORS[name], width=3),
                marker=dict(size=9, symbol='circle',
                            line=dict(color='#1A3355', width=2)),
                hovertemplate='<b>%{x}</b><br>$%{y:,.1f}M USD<extra>' + name + '</extra>'
            ))
        apply_template_layout(fig_ev,
                              height=360, showlegend=True,
                              yaxis_title="Millones USD",
                              xaxis_title="Año")
        fig_ev.update_xaxes(dtick=1, tickformat="d")
        st.plotly_chart(fig_ev, use_container_width=True)
        st.markdown('<div class="caption-text">Fuente: UN Comtrade. Elaboracin propia.</div>',
                    unsafe_allow_html=True)

    with c2:
        st.markdown("### Participacion 2024")
        totals = {}
        for name, df in dfs_base.items():
            t = df[df['year'] == 2024]['fob_millon'].sum()
            totals[LABELS[name].split(" (")[0]] = t
        fig_pie = go.Figure(go.Pie(
            labels=list(totals.keys()),
            values=list(totals.values()),
            marker_colors=[COLORS[k] for k in dfs_base.keys()],
            hole=0.55,
            textinfo='label+percent',
            textfont_size=11,
            hovertemplate='<b>%{label}</b><br>$%{value:,.0f}M<br>%{percent}<extra></extra>'
        ))
        apply_template_layout(fig_pie, height=360, showlegend=False)
        fig_pie.add_annotation(text="2024", x=0.5, y=0.5,
                               font_size=18, font_color="#1A3355",
                               showarrow=False, font_family="DM Mono")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Mapa de calor crecimiento mensual
    st.markdown("### Crecimiento Interanual Mensual (% YoY)")
    pivot_rows = []
    for name, df in dfs_base.items():
        for yr in [2022, 2023, 2024]:
            if yr not in df['year'].values or yr-1 not in df['year'].values:
                continue
            for m in range(1, 13):
                prev = df[(df['year'] == yr-1) & (df['month'] == m)]['fob_millon'].sum()
                curr = df[(df['year'] == yr) & (df['month'] == m)]['fob_millon'].sum()
                if prev > 0:
                    pct = (curr / prev - 1) * 100
                    pivot_rows.append({
                        'label': f"{name}{yr}",
                        'mes': m, 'yoy': pct
                    })

    if pivot_rows:
        df_heat = pd.DataFrame(pivot_rows)
        # Usar pivot_table evita errores si hay pares label/mes repetidos.
        heat_pivot = df_heat.pivot_table(
            index='label',
            columns='mes',
            values='yoy',
            aggfunc='mean'
        ).fillna(0)
        fig_heat = go.Figure(go.Heatmap(
            z=heat_pivot.values,
            x=[f"{'EFMAMJJASOND'[c-1]}" for c in heat_pivot.columns],
            y=heat_pivot.index.tolist(),
            colorscale=[[0, '#C73E1D'], [0.5, '#F8FAFC'], [1, '#1A7A3C']],
            zmid=0, zmin=-40, zmax=40,
            text=np.round(heat_pivot.values, 1),
            texttemplate='%{text}%',
            textfont_size=9,
            hovertemplate='%{y}<br>Mes %{x}<br>YoY: %{z:.1f}%<extra></extra>',
            colorbar=dict(title='YoY %', tickfont_color='#1A3355',
                          title_font_color='#1A3355')
        ))
        apply_template_layout(fig_heat, height=320)
        st.plotly_chart(fig_heat, use_container_width=True)

#
# TAB VOL: VOLUMEN & SENSORES CLCIRCULAR
#
with tab_vol:
    st.markdown("### Volumen de Exportaciones & Mercado Direccionable CLCircular")
    st.markdown(
        '<div class="info-box">'
        '<b>Modelo CLCircular:</b> Renta de sensores IoT reutilizables (~$30 USD/viaje) para monitoreo '
        'de temperatura, humedad, golpes, luz y geolocalizacion en cadena de suministro. '
        'Flota actual: <b>17,000 sensores</b> | Recuperacion: <b>98%</b> | Vida util: <b>3 años</b> | '
        'Sensores: <b>5G</b> (tiempo real, &gt;60 dias bateria) y <b>BG</b> (Bluetooth, hasta 1 año bateria). '
        'Un <b>viaje estimado</b> = netWgt / capacidad contenedor = 1 sensor desplegado.</div>',
        unsafe_allow_html=True
    )

    dfs_vol_sel = {k: v for k, v in dfs_vol.items() if k in sectores_sel}

    if not dfs_vol_sel:
        st.warning("No se encontraron archivos de volumen. Verifica que los archivos .xlsx esten en la misma carpeta.")
    else:
        # ── KPIs de volumen ──────────────────────────────────────────────────
        latest_year_vol = max(df['year'].max() for df in dfs_vol_sel.values())
        prev_year_vol   = latest_year_vol - 1

        trips_latest = sum(
            df[df['year'] == latest_year_vol]['viajes'].sum()
            for df in dfs_vol_sel.values()
        )
        trips_prev = sum(
            df[df['year'] == prev_year_vol]['viajes'].sum()
            for df in dfs_vol_sel.values()
        )
        vol_latest_M = sum(
            df[df['year'] == latest_year_vol]['netWgt'].sum() / 1_000_000
            for df in dfs_vol_sel.values()
        )
        trips_monthly_avg = trips_latest / 12 if trips_latest > 0 else 0
        yoy_trips = ((trips_latest / trips_prev) - 1) * 100 if trips_prev > 0 else 0

        # Datos reales CLCircular (documentos tecnicos)
        FLOTA_SENSORES    = 17_000
        USOS_SENSOR_ANIO  = 3          # actual; objetivo 4
        PRECIO_VIAJE_REAL = 30.0       # USD — precio de mercado confirmado
        RECUPERACION_PCT  = 98.0       # % recuperacion de sensores
        cap_flota_mensual = int(FLOTA_SENSORES * USOS_SENSOR_ANIO / 12)  # ~4,250/mes

        kv1, kv2, kv3, kv4, kv5 = st.columns(5)
        with kv1:
            st.metric("Viajes Totales Estimados", f"{trips_latest:,.0f}",
                      f"YoY {yoy_trips:+.1f}% ({int(latest_year_vol)})")
        with kv2:
            st.metric("Promedio Mensual", f"{trips_monthly_avg:,.0f}", "viajes/mes mercado")
        with kv3:
            st.metric("Capacidad Flota CLC", f"{cap_flota_mensual:,}",
                      f"sensores/mes (17k x {USOS_SENSOR_ANIO}usos/año)")
        with kv4:
            viajes_meta15  = int(15_000 / PRECIO_VIAJE_REAL)
            pct_pen_15k    = (viajes_meta15 / trips_monthly_avg * 100) if trips_monthly_avg > 0 else 0
            st.metric("Penetracion para $15k/mes", f"{pct_pen_15k:.3f}%",
                      f"{viajes_meta15:,} viajes a $30/viaje")
        with kv5:
            viajes_meta70  = int(70_000 / PRECIO_VIAJE_REAL)
            pct_pen_70k    = (viajes_meta70 / trips_monthly_avg * 100) if trips_monthly_avg > 0 else 0
            st.metric("Penetracion para $70k/mes (Q4)", f"{pct_pen_70k:.3f}%",
                      f"{viajes_meta70:,} viajes a $30/viaje")

        st.markdown("---")

        # ── Grafica 1: Volumen mensual por sector ────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Volumen Mensual por Sector")
            fig_vol_line = go.Figure()
            for sector, df in dfs_vol_sel.items():
                label = LABELS[sector].split(" (")[0]
                unit  = VOL_UNIT_LABEL[sector]
                fig_vol_line.add_trace(go.Scatter(
                    x=df['fecha'], y=df['netWgt_M'],
                    name=label,
                    mode='lines+markers',
                    line=dict(color=COLORS[sector], width=2.5),
                    marker=dict(size=5),
                    hovertemplate=f'<b>{label}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}M {unit}<extra></extra>'
                ))
            apply_template_layout(
                fig_vol_line, height=360,
                title="Volumen Mensual Exportado (netWgt)",
                yaxis_title="Millones de unidades (L o kg)",
                xaxis_title="Fecha",
                showlegend=True,
            )
            st.plotly_chart(fig_vol_line, use_container_width=True)
            st.markdown('<div class="caption-text">Fuente: UN Comtrade. netWgt en litros (Cerveza) o kg (resto).</div>',
                        unsafe_allow_html=True)

        with c2:
            st.markdown("#### Viajes Estimados por Mes")
            fig_trips = go.Figure()
            for sector, df in dfs_vol_sel.items():
                label = LABELS[sector].split(" (")[0]
                cap   = CONTAINER_CAPACITY[sector]
                fig_trips.add_trace(go.Bar(
                    x=df['fecha'], y=df['viajes'],
                    name=label,
                    marker_color=COLORS[sector],
                    opacity=0.85,
                    hovertemplate=f'<b>{label}</b><br>%{{x|%b %Y}}<br>%{{y:,.0f}} viajes<br>Cap: {cap:,} u/camion<extra></extra>'
                ))
            apply_template_layout(
                fig_trips, height=360,
                title="Contenedores/Camiones Estimados por Mes",
                yaxis_title="Viajes estimados",
                xaxis_title="Fecha",
                barmode="stack",
                showlegend=True,
            )
            st.plotly_chart(fig_trips, use_container_width=True)
            st.markdown('<div class="caption-text">Viaje = netWgt / capacidad contenedor. Cerveza: 26,000 L | Autopartes: 20,000 kg | Harinas: 25,000 kg.</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # ── Grafica 2: Anual por sector (barras agrupadas) ───────────────────
        st.markdown("#### Comparativo Anual de Viajes Estimados")
        rows_anual = []
        for sector, df in dfs_vol_sel.items():
            for yr, grp in df.groupby('year'):
                rows_anual.append({
                    'Sector': LABELS[sector].split(" (")[0],
                    'Año': int(yr),
                    'viajes': grp['viajes'].sum(),
                    'netWgt_M': grp['netWgt_M'].sum(),
                })
        df_anual_vol = pd.DataFrame(rows_anual)
        if not df_anual_vol.empty:
            fig_bar_vol = px.bar(
                df_anual_vol, x='Año', y='viajes', color='Sector',
                barmode='group',
                color_discrete_map={LABELS[k].split(' (')[0]: COLORS[k] for k in dfs_vol_sel},
                labels={'viajes': 'Viajes estimados', 'Año': 'Año'},
            )
            apply_template_layout(fig_bar_vol, height=320,
                                  title="Viajes Anuales por Sector",
                                  showlegend=True)
            st.plotly_chart(fig_bar_vol, use_container_width=True)

        st.markdown("---")

        # ── Calculadora de Revenue ───────────────────────────────────────────
        st.markdown("### Calculadora de Revenue Potencial CLCircular")
        st.caption(
            "Precio real confirmado: $30 USD/viaje. Flota: 17,000 sensores, 98% recuperacion, "
            "3 usos/sensor/año (~4,250 despliegues/mes de capacidad maxima actual)."
        )

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            precio_por_viaje = st.number_input(
                "Precio por viaje (USD)", min_value=1.0, max_value=500.0,
                value=30.0, step=5.0, key="vol_precio"
            )
        with rc2:
            penetracion_pct = st.slider(
                "Penetracion de mercado (%)", min_value=0.01, max_value=2.0,
                value=0.1, step=0.01, key="vol_penetracion"
            )
        with rc3:
            usos_sensor_anio = st.slider(
                "Usos sensor/año", min_value=1, max_value=6,
                value=3, step=1, key="vol_usos"
            )
        with rc4:
            meta_q4 = st.number_input(
                "Meta Q4 USD/mes", min_value=1000, value=70_000,
                step=5_000, key="vol_meta_q4"
            )

        # Capacidad de flota con usos configurables
        cap_flota_calc = int(FLOTA_SENSORES * usos_sensor_anio / 12)
        rev_techo_flota = cap_flota_calc * precio_por_viaje

        # Construir serie mensual de viajes totales
        all_vol_df = pd.concat(
            [df[['fecha', 'year', 'viajes']].assign(sector=s) for s, df in dfs_vol_sel.items()],
            ignore_index=True
        )
        monthly_total = all_vol_df.groupby('fecha', as_index=False)['viajes'].sum()
        monthly_total['rev_mercado']    = monthly_total['viajes'] * precio_por_viaje
        monthly_total['rev_clcircular'] = (monthly_total['viajes'] * (penetracion_pct / 100)
                                           ).clip(upper=cap_flota_calc) * precio_por_viaje

        # KPIs de la calculadora
        ck1, ck2, ck3, ck4 = st.columns(4)
        last_month_rev = monthly_total['rev_clcircular'].iloc[-1] if len(monthly_total) > 0 else 0
        with ck1:
            st.metric("Revenue ultimo mes (estimado)", f"${last_month_rev:,.0f}",
                      f"{penetracion_pct:.2f}% del mercado")
        with ck2:
            viajes_meta15 = int(15_000 / precio_por_viaje) if precio_por_viaje > 0 else 0
            pct_15 = viajes_meta15 / trips_monthly_avg * 100 if trips_monthly_avg > 0 else 0
            st.metric("Viajes para $15k/mes", f"{viajes_meta15:,}",
                      f"{pct_15:.3f}% del mercado")
        with ck3:
            viajes_meta70 = int(meta_q4 / precio_por_viaje) if precio_por_viaje > 0 else 0
            pct_70 = viajes_meta70 / trips_monthly_avg * 100 if trips_monthly_avg > 0 else 0
            st.metric(f"Viajes para ${meta_q4:,}/mes (Q4)", f"{viajes_meta70:,}",
                      f"{pct_70:.3f}% del mercado")
        with ck4:
            st.metric("Techo revenue flota actual", f"${rev_techo_flota:,.0f}/mes",
                      f"{cap_flota_calc:,} sensores/mes max")

        # Grafica de revenue
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(
            x=monthly_total['fecha'], y=monthly_total['rev_mercado'],
            name='Mercado total (100%)',
            marker_color='#D0DCE8', opacity=0.4,
            hovertemplate='%{x|%b %Y}<br>Mercado: $%{y:,.0f}<extra></extra>'
        ))
        fig_rev.add_trace(go.Scatter(
            x=monthly_total['fecha'], y=monthly_total['rev_clcircular'],
            name=f'CL Circular ({penetracion_pct:.2f}%)',
            line=dict(color=COLORS.get('Cerveza', '#1A7AB5'), width=3),
            mode='lines+markers', marker=dict(size=5),
            hovertemplate='%{x|%b %Y}<br>CLC: $%{y:,.0f}<extra></extra>'
        ))
        fig_rev.add_hline(
            y=15_000, line_dash='dash', line_color='#1A7A3C', line_width=1.5,
            annotation_text='Meta $15k/mes', annotation_font_color='#1A7A3C',
        )
        fig_rev.add_hline(
            y=meta_q4, line_dash='dash', line_color='#C4621A', line_width=1.5,
            annotation_text=f'Meta Q4 ${meta_q4:,}', annotation_font_color='#C4621A',
        )
        fig_rev.add_hline(
            y=rev_techo_flota, line_dash='dot', line_color='#7B2FBE', line_width=1.5,
            annotation_text=f'Techo flota actual (${rev_techo_flota:,.0f})',
            annotation_font_color='#7B2FBE',
        )
        apply_template_layout(
            fig_rev, height=420,
            title="Revenue Potencial vs Metas y Capacidad de Flota CLCircular",
            yaxis_title="USD / mes",
            xaxis_title="Fecha",
            showlegend=True,
        )
        st.plotly_chart(fig_rev, use_container_width=True)

        # ── Tabla resumen operativo ──────────────────────────────────────────
        c_tbl1, c_tbl2 = st.columns(2)
        with c_tbl1:
            st.markdown("#### Supuestos de capacidad por sector")
            cap_df = pd.DataFrame([
                {
                    "Sector": LABELS[s].split(" (")[0],
                    "Unidad": VOL_UNIT_LABEL[s],
                    "Cap. contenedor": f"{CONTAINER_CAPACITY[s]:,} {VOL_UNIT_LABEL[s]}",
                    "Viajes/mes promedio": f"{dfs_vol_sel[s]['viajes'].mean():,.0f}",
                    "Viajes/mes ultimo año": f"{dfs_vol_sel[s][dfs_vol_sel[s]['year'] == latest_year_vol]['viajes'].sum() / 12:,.0f}",
                }
                for s in dfs_vol_sel
            ])
            st.dataframe(cap_df, use_container_width=True, hide_index=True)

        with c_tbl2:
            st.markdown("#### Ficha operativa CLCircular")
            ficha = pd.DataFrame([
                {"Parametro": "Flota de sensores",         "Valor": "17,000 unidades"},
                {"Parametro": "Precio por viaje",          "Valor": f"~${precio_por_viaje:.0f} USD"},
                {"Parametro": "Usos/sensor/año (actual)",  "Valor": "3 (objetivo: 4)"},
                {"Parametro": "Recuperacion sensores",     "Valor": "98%"},
                {"Parametro": "Vida util sensor",          "Valor": "3 años"},
                {"Parametro": "Capacidad flota/mes",       "Valor": f"{cap_flota_calc:,} sensores"},
                {"Parametro": "Techo revenue (flota actual)", "Valor": f"${rev_techo_flota:,.0f}/mes"},
                {"Parametro": "Sensores disponibles",      "Valor": "5G (tiempo real) / BG (Bluetooth)"},
                {"Parametro": "Retencion de clientes",     "Valor": "~100%"},
                {"Parametro": "Modelo export. (dinamico)", "Valor": "CLC gestiona logistica inversa"},
            ])
            st.dataframe(ficha, use_container_width=True, hide_index=True)

#
# TAB ANALISIS: CLUSTERING + PCA + DATOS
#
with tab_analisis:
    st.markdown("## Clustering K-Means")
with tab_analisis:
    st.markdown("### Seleccion del Numero optimo de Clusters")

    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(go.Scatter(
            x=df_metrics['k'], y=df_metrics['inertia'],
            mode='lines+markers', name='Inercia (Elbow)',
            line=dict(color='#1565A0', width=2.5),
            marker=dict(size=8)
        ), secondary_y=False)
        fig_elbow.add_trace(go.Scatter(
            x=df_metrics['k'], y=df_metrics['silhouette'],
            mode='lines+markers', name='Silhouette Score',
            line=dict(color='#C4621A', width=2.5, dash='dot'),
            marker=dict(size=8, symbol='diamond')
        ), secondary_y=True)
        fig_elbow.add_vline(x=k_clusters_eff, line_dash='dash',
                             line_color='#1A7A3C', line_width=2,
                             annotation_text=f"k={k_clusters_eff} seleccionado",
                             annotation_font_color='#1A7A3C')
        apply_template_layout(
            fig_elbow,
            height=320,
            title='Elbow Method + Silhouette',
            legend=dict(x=0.5, y=-0.15, orientation='h')
        )
        fig_elbow.update_yaxes(
            title_text="Inercia", secondary_y=False,
            title_font=dict(family="DM Sans", color="#0D1F2D", size=12),
            tickfont=dict(family="DM Sans", color="#1A3355", size=11),
            gridcolor='#D0DCE8',
        )
        fig_elbow.update_yaxes(
            title_text="Silhouette", secondary_y=True,
            title_font=dict(family="DM Sans", color="#0D1F2D", size=12),
            tickfont=dict(family="DM Sans", color="#1A3355", size=11),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        st.markdown("#### Perfil de Clusters")
        profile = df_clusters.groupby('cluster').agg(
            FOB_Prom_MUSD=('fob_millon', lambda x: f"${x.mean():,.0f}M"),
            Crecimiento_YoY=('yoy', lambda x: f"{x.mean():+.1f}%"),
            Precio_Prom=('precio_prom', lambda x: f"${x.mean():.2f}"),
            Sectores=('sector', lambda x: ', '.join(sorted(x.unique()))),
            N_obs=('sector', 'count'),
        ).reset_index()
        st.dataframe(
            profile, use_container_width=True, hide_index=True,
            column_config={
                "cluster": st.column_config.TextColumn("Cluster"),
                "FOB_Prom_MUSD": st.column_config.TextColumn("FOB Prom/año"),
                "Crecimiento_YoY": st.column_config.TextColumn("Crecim. YoY"),
                "Precio_Prom": st.column_config.TextColumn("Precio Unitario"),
                "Sectores": st.column_config.TextColumn("Sectores"),
                "N_obs": st.column_config.NumberColumn("Obs.", format="%d"),
            }
        )
        st.markdown(
            f'<div class="info-box"> <b>Silhouette Score = {sil_score:.3f}</b> con k={k_clusters_eff}. '
            f'Valores >0.5 indican clusters bien definidos.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Distribucion de Clusters por Año y Sector")
    fig_bar = px.bar(
        df_clusters, x='year', y='fob_millon',
        color='cluster', facet_col='sector',
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'fob_millon': 'FOB (MUSD)', 'year': 'Año'},
        category_orders={"sector": list(dfs.keys())},
        text_auto='.0f',
    )
    apply_template_layout(fig_bar, height=320, showlegend=True)
    fig_bar.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1], font_color="#1A3355"))
    st.plotly_chart(fig_bar, use_container_width=True)

#
# (PCA - dentro de tab_analisis)
#
with tab_analisis:
    st.markdown("---")
    st.markdown("## Analisis de Componentes Principales (PCA)")
with tab_analisis:
    var_exp = pca_model.explained_variance_ratio_
    st.markdown(
        f'<div class="info-box">'
        f'PC1 explica <b>{var_exp[0]:.1%}</b> de la varianza  '
        f'PC2 explica <b>{var_exp[1]:.1%}</b>  '
        f'<b>Total: {sum(var_exp):.1%}</b></div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        fig_pca = go.Figure()
        cluster_colors_pca = px.colors.qualitative.Bold
        for cl in sorted(df_clusters['cluster'].unique()):
            sub = df_clusters[df_clusters['cluster'] == cl]
            fig_pca.add_trace(go.Scatter(
                x=sub['PC1'], y=sub['PC2'],
                mode='markers+text',
                name=f'Cluster {cl}',
                marker=dict(
                    size=16,
                    color=cluster_colors_pca[int(cl) % len(cluster_colors_pca)],
                    line=dict(color='#1A3355', width=1.5),
                    symbol='circle',
                ),
                text=[f"{r['sector'][:4]}\n'{str(r['year'])[2:]}"
                      for _, r in sub.iterrows()],
                textposition='top center',
                textfont=dict(size=9, color='#1A3355'),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'PC1=%{x:.2f}, PC2=%{y:.2f}<extra>Cluster ' + cl + '</extra>'
                )
            ))
        apply_template_layout(
            fig_pca, height=420,
            xaxis_title=f"PC1 ({var_exp[0]:.0%} var. explicada)",
            yaxis_title=f"PC2 ({var_exp[1]:.0%} var. explicada)",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with c2:
        st.markdown("#### Varianza Explicada por Componente")
        n_comp = len(var_exp)
        df_var = pd.DataFrame({
            'Componente': [f'PC{i+1}' for i in range(n_comp)],
            'Varianza (%)': [v*100 for v in var_exp],
            'Acumulada (%)': [sum(var_exp[:i+1])*100 for i in range(n_comp)]
        })
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=df_var['Componente'], y=df_var['Varianza (%)'],
            name='Individual', marker_color='#1565A0',
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        ))
        fig_scree.add_trace(go.Scatter(
            x=df_var['Componente'], y=df_var['Acumulada (%)'],
            name='Acumulada', mode='lines+markers',
            line=dict(color='#C4621A', width=2, dash='dot'),
            marker=dict(size=8),
            hovertemplate='Acum. %{y:.1f}%<extra></extra>'
        ))
        fig_scree.add_hline(y=80, line_dash='dash', line_color='#B54127',
                             annotation_text='80% umbral',
                             annotation_font_color='#B54127')
        apply_template_layout(fig_scree, height=260, showlegend=True)
        st.plotly_chart(fig_scree, use_container_width=True)

        st.markdown("#### Pesos de Variables (Loadings)")
        feats = ['FOB (MUSD)', 'Crec. YoY (%)', 'Precio Prom', 'Peso (MT)']
        df_load = pd.DataFrame(
            pca_model.components_[:2].T,
            index=feats,
            columns=['PC1', 'PC2']
        ).round(3)
        st.dataframe(df_load, use_container_width=True)

# 
# TAB 4: PROYECCIONES
# 
with tab4:
    # ── Badge de filtro activo ────────────────────────────────────────────────
    _FILTRO_STYLE = {
        "Todos":    ("#1565A0", "Todos los periodos (2021–2025)"),
        "Positivo": ("#1A7A3C", "Solo periodos con crecimiento YoY ≥ +5 %"),
        "Neutral":  ("#B07D10", "Solo periodos estables  (YoY entre −5 % y +5 %)"),
        "Negativo": ("#C53030", "Solo periodos con contraccion YoY ≤ −5 %"),
    }
    _fc, _fl = _FILTRO_STYLE.get(factor_filter, ("#1565A0", factor_filter))
    st.markdown(
        f'<span style="background:{_fc};color:#FFF;padding:4px 12px;'
        f'border-radius:14px;font-size:12px;font-weight:700">'
        f'Filtro activo: {_fl}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("### Proyecciones 2025-2027 por Sector")
    st.caption(
        "Regresion sobre totales anuales FOB completos. "
        "Para filtros distintos de 'Todos' se usan solo los años donde ese factor fue dominante."
    )

    # Siempre mostrar todos los sectores seleccionados con historico completo.
    # El filtro cambia el METODO de proyeccion, no los datos mostrados.
    sectores_proj = [s for s in sectores_sel if s in dfs_all]

    if not sectores_proj:
        st.warning("No hay sectores con datos disponibles.")
    else:
        proj_cols = st.columns(len(sectores_proj))
        for i, name in enumerate(sectores_proj):
            df_full = dfs_all[name]

            # Historico anual completo — siempre todos los años
            anual_full = (
                df_full.groupby("year")["fob_millon"]
                .sum()
                .reset_index()
            )
            anual_full = anual_full[anual_full["fob_millon"] > 0].copy()
            if anual_full.empty:
                continue

            last_yr   = int(anual_full["year"].max())
            last_val  = float(anual_full.loc[anual_full["year"] == last_yr, "fob_millon"].iloc[0])
            base      = int(anual_full["year"].min())
            future_yrs = [last_yr + 1, last_yr + 2, last_yr + 3]

            # R² siempre sobre todos los años (da una medida consistente)
            anual_full["yr_idx"] = anual_full["year"] - base
            reg_full = LinearRegression()
            reg_full.fit(anual_full[["yr_idx"]], anual_full["fob_millon"])
            r2 = reg_full.score(anual_full[["yr_idx"]], anual_full["fob_millon"])
            slope_full, intercept_full, _, p_val_full, _ = stats.linregress(
                anual_full["year"].astype(float),
                anual_full["fob_millon"].astype(float),
            )

            if factor_filter == "Todos":
                # Proyeccion por regresion lineal
                X_fut     = np.array([y - base for y in future_yrs]).reshape(-1, 1)
                y_fut     = np.clip(reg_full.predict(X_fut), 0, None)
                slope     = slope_full
                intercept = intercept_full
                caption_txt = (
                    f"Regresion lineal — n={len(anual_full)} años  |  "
                    f"p={p_val_full:.3f}"
                )
            else:
                # Proyeccion compuesta con tasa YoY media de los meses del filtro
                df_filtrado = df_full[df_full["factor_mercado"] == factor_filter]
                yoy_vals    = df_filtrado["yoy_mensual_pct"].dropna()
                n_meses     = len(yoy_vals)
                avg_yoy_pct = float(yoy_vals.mean()) if n_meses > 0 else 0.0
                avg_rate    = avg_yoy_pct / 100.0
                y_fut = np.array([last_val * (1 + avg_rate) ** k for k in range(1, 4)])
                y_fut = np.clip(y_fut, 0, None)
                slope     = last_val * avg_rate
                intercept = last_val - slope * last_yr
                caption_txt = (
                    f"Proyeccion compuesta con YoY media de periodos {factor_filter.lower()}s: "
                    f"**{avg_yoy_pct:+.1f}%/año** ({n_meses} meses)"
                )

            with proj_cols[i]:
                st.markdown(f"**{str(LABELS.get(name, name)).split(' (')[0]}**")
                st.caption(caption_txt)
                st.metric("R²", f"{r2:.3f}", delta=f"n={len(anual_full)} años")

                fig_proj = go.Figure()
                # Barras historicas: siempre todos los años
                fig_proj.add_trace(go.Bar(
                    x=anual_full["year"], y=anual_full["fob_millon"],
                    name="Real", marker_color=COLORS[name], opacity=0.85,
                    hovertemplate="%{x}: $%{y:,.1f}M<extra>Real</extra>",
                ))
                # Barras proyectadas
                fig_proj.add_trace(go.Bar(
                    x=future_yrs, y=y_fut,
                    name="Proyeccion", marker_color=COLORS[name],
                    opacity=0.4, marker_pattern_shape="/",
                    hovertemplate="%{x}: $%{y:,.1f}M<extra>Proyectado</extra>",
                ))
                # Linea de tendencia desde primer año real hasta ultimo proyectado
                x_line = np.linspace(base, future_yrs[-1], 120)
                y_line = slope * x_line + intercept
                fig_proj.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    name="Tendencia",
                    line=dict(color="#1A3355", width=1.5, dash="dot"),
                ))
                apply_template_layout(
                    fig_proj,
                    height=290,
                    showlegend=False,
                    barmode="group",
                    yaxis_title="MUSD",
                    margin=dict(t=20, b=30, l=40, r=10),
                )
                st.plotly_chart(fig_proj, use_container_width=True)

                st.dataframe(
                    pd.DataFrame({
                        "Año": future_yrs,
                        "Proyectado (MUSD)": [f"${v:,.1f}M" for v in y_fut],
                    }),
                    use_container_width=True, hide_index=True,
                )

# 
    st.markdown("---")
    st.markdown("### Forecast Mensual 2026")

    if not SARIMA_AVAILABLE:
        st.error("statsmodels no esta disponible. Instala con: pip install statsmodels")
    else:
        with st.spinner("Ajustando modelos SARIMA..."):
            df_fc_2026, df_sarima_models = sarima_forecast_2026(dfs_all)

        if df_fc_2026.empty or df_sarima_models.empty:
            st.warning("No fue posible estimar SARIMA para todos los sectores con los datos actuales.")
        else:
            # ── Controles de filtro y visualizacion ──────────────────────────
            HS_CODES = {
                "Cerveza":      "HS 220300",
                "Autopartes":   "HS 8708",
                "Harina_Maiz":  "HS 110220",
                "Harina_Trigo": "HS 1101",
            }
            hs_options = [
                f"{LABELS[s].split(' (')[0]}  ({HS_CODES[s]})"
                for s in dfs.keys() if s in df_fc_2026["sector"].values
            ]
            hs_key_map = {
                f"{LABELS[s].split(' (')[0]}  ({HS_CODES[s]})": s
                for s in dfs.keys() if s in df_fc_2026["sector"].values
            }

            ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 2, 1, 1])
            with ctrl1:
                hs_sel = st.multiselect(
                    "Filtrar por codigo arancelario (HS)",
                    options=hs_options,
                    default=hs_options,
                    key="sarima_hs_filter",
                )
            with ctrl2:
                meses_hist = st.slider(
                    "Meses de historico a mostrar",
                    min_value=6, max_value=60, value=24, step=6,
                    key="sarima_hist_window",
                )
            with ctrl3:
                chart_height = st.slider(
                    "Altura grafica", min_value=300, max_value=800,
                    value=460, step=50, key="sarima_height",
                )
            with ctrl4:
                show_band = st.checkbox("Banda confianza 80%", value=True, key="sarima_band")

            sectores_sarima = [hs_key_map[h] for h in hs_sel if h in hs_key_map]

            if not sectores_sarima:
                st.info("Selecciona al menos un sector para visualizar el forecast.")
            else:
                # Columna del escenario segun filtro activo
                _ESC_COL = {
                    "Todos":    "forecast_musd",
                    "Positivo": "high80_musd",
                    "Neutral":  "forecast_musd",
                    "Negativo": "low80_musd",
                }
                _ESC_NOMBRE = {
                    "Todos":    "Forecast central",
                    "Positivo": "Escenario Optimista",
                    "Neutral":  "Escenario Neutral",
                    "Negativo": "Escenario Pesimista",
                }
                fc_col    = _ESC_COL.get(factor_filter, "forecast_musd")
                esc_label = _ESC_NOMBRE.get(factor_filter, "Forecast")

                # Explicacion de notacion SARIMA
                with st.expander("¿Qué significa la notación SARIMA (p,d,q)×(P,D,Q,12)?"):
                    st.markdown(
                        "El modelo **SARIMA** combina dos partes:\n\n"
                        "- **(p, d, q)** — componente no estacional:\n"
                        "  - `p` = términos autorregresivos (cuántos rezagos pasados influyen)\n"
                        "  - `d` = diferenciaciones para hacer la serie estacionaria\n"
                        "  - `q` = términos de media móvil (errores pasados)\n\n"
                        "- **(P, D, Q, 12)** — componente estacional con periodo 12 meses:\n"
                        "  - `P`, `D`, `Q` = equivalentes estacionales de p, d, q\n"
                        "  - `12` = periodo estacional (datos mensuales, ciclo anual)\n\n"
                        "El modelo seleccionado para cada sector es el de **menor AIC** "
                        "(Criterio de Información de Akaike) entre todas las combinaciones exploradas. "
                        "Menor AIC = mejor balance entre ajuste y complejidad del modelo.\n\n"
                        "Los valores mostrados son la **suma de los 12 meses proyectados** del "
                        "escenario seleccionado (en millones USD)."
                    )

                df_models_sel = df_sarima_models[df_sarima_models["sector"].isin(sectores_sarima)]
                mcols = st.columns(max(len(df_models_sel), 1))
                for i, row in enumerate(df_models_sel.sort_values("sector").itertuples(index=False)):
                    with mcols[i]:
                        hs = HS_CODES.get(row.sector, "")
                        # Calcular total del escenario correcto desde df_fc_2026
                        sec_fc = df_fc_2026[df_fc_2026["sector"] == row.sector]
                        if fc_col in sec_fc.columns and not sec_fc.empty:
                            total_esc = float(sec_fc[fc_col].clip(lower=0).sum())
                        else:
                            total_esc = float(row.forecast_2026_total_musd)
                        lbl = str(LABELS.get(row.sector, row.sector)).split(" (")[0]
                        mensual_avg = total_esc / 12
                        st.metric(
                            f"{lbl} ({hs})",
                            f"${total_esc:,.0f}M",
                            f"{esc_label} · SARIMA {row.order}×{row.seasonal_order}",
                        )
                        st.caption(f"Total anual 2026 (suma 12 meses) · Promedio mensual: **${mensual_avg:,.0f}M/mes**")

                # ── Grafica principal SARIMA ──────────────────────────────
                fecha_inicio_hist = pd.Timestamp.now().normalize() - pd.DateOffset(months=meses_hist)
                fig_sarima = go.Figure()

                # Mapa: filtro activo -> columna y nombre del escenario a destacar
                _SARIMA_ESCENARIO = {
                    "Todos":    None,
                    "Positivo": ("high80_musd",   "Optimista",  "dash"),
                    "Neutral":  ("forecast_musd", "Neutral",    "dot"),
                    "Negativo": ("low80_musd",    "Pesimista",  "dash"),
                }
                _esc_info = _SARIMA_ESCENARIO.get(factor_filter)

                for sector in sectores_sarima:
                    if sector not in dfs_all:
                        continue
                    color   = COLORS.get(sector, "#1565A0")
                    label   = str(LABELS.get(sector, sector)).split(" (")[0]
                    hs_code = HS_CODES.get(sector, "")
                    sub_fc  = df_fc_2026[df_fc_2026["sector"] == sector].sort_values("fecha")
                    if sub_fc.empty:
                        continue

                    # Historico: siempre desde datos completos (dfs_all)
                    hist = _serie_mensual_sector(
                        dfs_all[sector][dfs_all[sector]["fecha"] >= fecha_inicio_hist]
                    )

                    fig_sarima.add_trace(go.Scatter(
                        x=hist.index, y=hist.values,
                        mode="lines",
                        name=f"{label} Real ({hs_code})",
                        line=dict(color=color, width=2.5),
                        hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>$%{{y:.2f}}M FOB<extra>Real</extra>",
                    ))

                    has_ic = "low80_musd" in sub_fc.columns and "high80_musd" in sub_fc.columns

                    if _esc_info is None:
                        # "Todos" → mostrar los 3 escenarios
                        if has_ic and show_band:
                            x_band = list(sub_fc["fecha"]) + list(sub_fc["fecha"])[::-1]
                            y_band = list(sub_fc["high80_musd"]) + list(sub_fc["low80_musd"])[::-1]
                            fig_sarima.add_trace(go.Scatter(
                                x=x_band, y=y_band, fill="toself",
                                fillcolor=hex_to_rgba(color, 0.12) if color.startswith("#") else color,
                                line=dict(color="rgba(0,0,0,0)"),
                                showlegend=False, hoverinfo="skip", opacity=1,
                            ))
                        if has_ic:
                            fig_sarima.add_trace(go.Scatter(
                                x=sub_fc["fecha"], y=sub_fc["low80_musd"],
                                mode="lines", name=f"{label} Pesimista",
                                line=dict(color=color, dash="dash", width=1.5),
                                hovertemplate=f"<b>{label} Pesimista</b><br>%{{x|%b %Y}}<br>$%{{y:.2f}}M<extra>IC80% inf</extra>",
                            ))
                        fig_sarima.add_trace(go.Scatter(
                            x=sub_fc["fecha"], y=sub_fc["forecast_musd"],
                            mode="lines+markers", name=f"{label} Neutral",
                            line=dict(color=color, dash="dot", width=2.5),
                            marker=dict(size=7, symbol="circle-open", line=dict(width=2, color=color)),
                            hovertemplate=f"<b>{label} Neutral</b><br>%{{x|%b %Y}}<br>$%{{y:.2f}}M<extra>Central</extra>",
                        ))
                        if has_ic:
                            fig_sarima.add_trace(go.Scatter(
                                x=sub_fc["fecha"], y=sub_fc["high80_musd"],
                                mode="lines", name=f"{label} Optimista",
                                line=dict(color=color, dash="dash", width=1.5),
                                hovertemplate=f"<b>{label} Optimista</b><br>%{{x|%b %Y}}<br>$%{{y:.2f}}M<extra>IC80% sup</extra>",
                            ))
                    else:
                        # Filtro activo → un solo escenario destacado + banda de fondo
                        col_esc, nombre_esc, dash_esc = _esc_info
                        y_esc = sub_fc[col_esc] if col_esc in sub_fc.columns else sub_fc["forecast_musd"]
                        if has_ic and show_band:
                            x_band = list(sub_fc["fecha"]) + list(sub_fc["fecha"])[::-1]
                            y_band = list(sub_fc["high80_musd"]) + list(sub_fc["low80_musd"])[::-1]
                            fig_sarima.add_trace(go.Scatter(
                                x=x_band, y=y_band, fill="toself",
                                fillcolor=hex_to_rgba(color, 0.08) if color.startswith("#") else color,
                                line=dict(color="rgba(0,0,0,0)"),
                                showlegend=False, hoverinfo="skip", opacity=1,
                            ))
                        fig_sarima.add_trace(go.Scatter(
                            x=sub_fc["fecha"], y=y_esc,
                            mode="lines+markers",
                            name=f"{label} {nombre_esc} ({hs_code})",
                            line=dict(color=color, dash=dash_esc, width=2.5),
                            marker=dict(size=7, symbol="circle-open", line=dict(width=2, color=color)),
                            hovertemplate=f"<b>{label} {nombre_esc}</b><br>%{{x|%b %Y}}<br>$%{{y:.2f}}M<extra>{factor_filter}</extra>",
                        ))

                # Linea vertical: inicio del forecast
                if not df_fc_2026.empty:
                    primer_fc = df_fc_2026["fecha"].min()
                    fig_sarima.add_vline(
                        x=primer_fc.timestamp() * 1000,
                        line_dash="dash", line_color="#6B8CAE", line_width=1.5,
                        annotation_text="Inicio forecast",
                        annotation_font_color="#1A3355",
                        annotation_position="top left",
                    )

                apply_template_layout(
                    fig_sarima,
                    height=chart_height,
                    title="Historico vs Forecast 2026",
                    xaxis_title="Fecha",
                    yaxis_title="FOB (MUSD)",
                    showlegend=True,
                )
                st.plotly_chart(fig_sarima, use_container_width=True)

                # ── Tabla forecast ────────────────────────────────────────
                st.markdown("#### Forecast mensual 2026 (MUSD)")
                df_fc_table = (
                    df_fc_2026[df_fc_2026["sector"].isin(sectores_sarima)]
                    .assign(
                        Mes=lambda d: d["fecha"].dt.strftime("%Y-%m"),
                        Sector=lambda d: d["sector"].map(
                            lambda x: f"{LABELS.get(x, x).split(' (')[0]} ({HS_CODES.get(x, '')})"
                        ),
                    )[["Mes", "Sector", "forecast_musd", "low80_musd", "high80_musd"]]
                    .rename(columns={
                        "forecast_musd": "Forecast (MUSD)",
                        "low80_musd":    "IC80% inferior",
                        "high80_musd":   "IC80% superior",
                    })
                    .sort_values(["Mes", "Sector"])
                )
                st.dataframe(df_fc_table.round(3), use_container_width=True,
                             hide_index=True, height=340)

                # ── Parametros SARIMA ─────────────────────────────────────
                with st.expander("Ver parametros SARIMA seleccionados por AIC"):
                    st.dataframe(
                        df_models_sel.rename(columns={
                            "sector": "Sector",
                            "order": "(p,d,q)",
                            "seasonal_order": "(P,D,Q,12)",
                            "aic": "AIC",
                            "forecast_2026_total_musd": "Forecast 2026 total (MUSD)",
                        }).round(3),
                        use_container_width=True, hide_index=True,
                    )

            st.markdown("---")
            st.markdown("### Forecast por escenarios 2026")
            st.caption(
                "Los 3 escenarios provienen directamente del intervalo de confianza 80 % "
                "del modelo SARIMA: **Pesimista** = límite inferior IC80%, "
                "**Neutral** = pronóstico central, **Optimista** = límite superior IC80%."
            )

            scenario_colors = {
                "Pesimista": "#C53030",
                "Neutral":   "#B07D10",
                "Optimista": "#1A7A3C",
            }

            # Escenarios derivados directamente de los IC80% del modelo SARIMA
            esc_col_map = {
                "Pesimista": "low80_musd",
                "Neutral":   "forecast_musd",
                "Optimista": "high80_musd",
            }
            scen_rows = []
            for esc, col in esc_col_map.items():
                tmp = df_fc_2026.copy()
                tmp["escenario"] = esc
                tmp["forecast_escenario_musd"] = tmp[col]
                scen_rows.append(tmp)
            df_fc_scen = pd.concat(scen_rows, ignore_index=True)

            # Totales 2026 por sector/escenario
            df_scen_sector = (
                df_fc_scen.groupby(["sector", "escenario"], as_index=False)["forecast_escenario_musd"]
                .sum()
                .sort_values(["sector", "escenario"])
            )
            df_scen_sector["Sector"] = df_scen_sector["sector"].map(
                lambda x: LABELS.get(x, x).split(" (")[0]
            )

            # KPIs por escenario (total combinado de los sectores)
            total_by_scen = (
                df_scen_sector.groupby("escenario", as_index=False)["forecast_escenario_musd"]
                .sum()
                .set_index("escenario")["forecast_escenario_musd"]
                .to_dict()
            )
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Total 2026 Pesimista", f"${total_by_scen.get('Pesimista', 0):,.1f}M")
            with k2:
                st.metric("Total 2026 Neutral", f"${total_by_scen.get('Neutral', 0):,.1f}M")
            with k3:
                st.metric("Total 2026 Optimista", f"${total_by_scen.get('Optimista', 0):,.1f}M")

            fig_scen = px.bar(
                df_scen_sector,
                x="Sector",
                y="forecast_escenario_musd",
                color="escenario",
                barmode="group",
                color_discrete_map=scenario_colors,
                labels={
                    "forecast_escenario_musd": "Forecast 2026 (MUSD)",
                    "escenario": "Escenario",
                },
                title="Comparativo 2026 por sector y escenario",
            )
            apply_template_layout(fig_scen, height=360)
            st.plotly_chart(fig_scen, use_container_width=True)

            df_scen_table = (
                df_scen_sector[["Sector", "escenario", "forecast_escenario_musd"]]
                .rename(columns={
                    "escenario": "Escenario",
                    "forecast_escenario_musd": "Forecast 2026 (MUSD)",
                })
                .sort_values(["Sector", "Escenario"])
            )
            st.dataframe(df_scen_table.round(3), use_container_width=True, hide_index=True)

# 
# (Datos & Estadisticas - dentro de tab_analisis)
#
with tab_analisis:
    st.markdown("---")
    st.markdown("## Datos y Estadisticas")
with tab_analisis:
    st.markdown("### Estadisticas Descriptivas")
    sector_tab = st.selectbox(
        "Seleccionar sector",
        options=list(dfs.keys()),
        format_func=lambda x: LABELS[x]
    )

    df_sel = dfs[sector_tab]
    df_yr = df_sel[
        (df_sel['year'] >= year_range[0]) & (df_sel['year'] <= year_range[1])
    ]

    # Stats
    stats_data = {
        'Metrica': ['Registros', 'FOB Total (MUSD)', 'FOB Promedio (MUSD)', 'FOB Maximo (MUSD)',
                    'FOB Minimo (MUSD)', 'Desv. Estandar (MUSD)', 'Precio Prom (USD/u)'],
        'Valor': [
            len(df_yr),
            f"${df_yr['fob_millon'].sum():,.1f}",
            f"${df_yr['fob_millon'].mean():,.1f}",
            f"${df_yr['fob_millon'].max():,.1f}",
            f"${df_yr['fob_millon'].min():,.1f}",
            f"${df_yr['fob_millon'].std():,.1f}",
            f"${df_yr['precio_unitario'].mean():,.2f}" if df_yr['precio_unitario'].notna().any() else "N/A"
        ]
    }

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    with c2:
        # Box plots
        df_box = df_yr.copy()
        fig_box = px.box(
            df_box, x='year', y='fob_millon',
            color_discrete_sequence=[COLORS[sector_tab]],
            labels={'fob_millon': 'FOB (MUSD)', 'year': 'Año'},
            title=f"Distribucion Mensual FOB  {LABELS[sector_tab].split(' (')[0]}"
        )
        apply_template_layout(fig_box, height=300)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown("### Datos Crudos")
    cols_show = ['fecha', 'year', 'month', 'trimestre', 'fob_millon', 'peso_kt',
                 'precio_unitario', 'yoy_mensual_pct', 'factor_mercado', 'sector']
    cols_show = [c for c in cols_show if c in df_yr.columns]
    st.dataframe(
        df_yr[cols_show].rename(columns={
            'fecha': 'Fecha', 'year': 'Año', 'month': 'Mes',
            'trimestre': 'Trimestre', 'fob_millon': 'FOB (MUSD)',
            'peso_kt': 'Peso (KT)', 'precio_unitario': 'Precio Unitario',
            'yoy_mensual_pct': 'YoY Mensual (%)', 'factor_mercado': 'Factor',
            'sector': 'Sector'
        }).round(3),
        use_container_width=True,
        height=320
    )

#
# TAB FINANZAS CRL
#
with tab_fin:
    st.markdown("### Modelo Financiero CLCircular")
    st.markdown(
        '<div class="info-box"><b>Objetivo:</b> consolidar los supuestos financieros del archivo `CRL.xlsx` '
        'y compararlos contra los estados proyectados con y sin financiamiento para apoyar decisiones de inversion.</div>',
        unsafe_allow_html=True,
    )

    if not crl_data:
        st.warning(f"No se encontro `{CRL_FILE}` o no fue posible leerlo.")
    else:
        st.caption(f"Fuente activa: {crl_path}")

        df_sup = crl_data.get("supuestos", pd.DataFrame()).copy()
        df_ops = crl_data.get("clcircular", pd.DataFrame()).copy()
        escenarios = crl_data.get("escenarios", {})
        meta_esc = crl_data.get("meta", {})

        def _lookup(df_in: pd.DataFrame, token: str):
            if df_in.empty or "Concepto" not in df_in.columns:
                return np.nan
            mask = df_in["Concepto"].astype(str).map(_normalize_text).str.contains(token, na=False)
            if not mask.any():
                return np.nan
            return df_in.loc[mask, "Valor"].iloc[0]

        fx_prom = pd.to_numeric(_lookup(df_sup, "tipo de cambio promedio eur mxn"), errors="coerce")
        infl_prom = pd.to_numeric(_lookup(df_sup, "inflacion en mexico"), errors="coerce")
        bono_prom = pd.to_numeric(_lookup(df_sup, "tasa bono"), errors="coerce")
        cont_exp = pd.to_numeric(_lookup(df_sup, "contenedores exportados"), errors="coerce")
        cont_cl = pd.to_numeric(_lookup(df_sup, "contenedores que ya tienen cl circular"), errors="coerce")
        part_mkt = pd.to_numeric(_lookup(df_sup, "participacion potencial"), errors="coerce")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("EUR/MXN promedio", f"{fx_prom:,.2f}" if pd.notna(fx_prom) else "N/D")
        with m2:
            st.metric("Inflacion promedio", f"{infl_prom:.2%}" if pd.notna(infl_prom) else "N/D")
        with m3:
            st.metric("Bono español 10Y", f"{bono_prom:.2%}" if pd.notna(bono_prom) else "N/D")
        with m4:
            st.metric("Participacion potencial", f"{part_mkt:.2%}" if pd.notna(part_mkt) else "N/D")

        st.markdown(
            f"**Lectura ejecutiva:** el modelo financiero parte de un universo aproximado de "
            f"`{cont_exp:,.0f}` contenedores exportados y `{cont_cl:,.0f}` contenedores ya atendidos por CL Circular."
            if pd.notna(cont_exp) and pd.notna(cont_cl)
            else "**Lectura ejecutiva:** el archivo CRL concentra supuestos macro y escenarios de crecimiento para evaluar escalamiento comercial."
        )

        c_fin_1, c_fin_2 = st.columns([1.6, 1.0])
        with c_fin_1:
            st.markdown("#### Escenarios proyectados 2025-2035")
            metric_options = ["Ingreso", "EBITDA", "EBIT", "Utilidad Neta", "Margen bruto", "Costo de Venta"]
            metric_sel = st.selectbox("Metrica financiera", metric_options, key="crl_metric_sel")

            rows = []
            for scenario_name, df_sc in escenarios.items():
                if df_sc.empty or "Concepto" not in df_sc.columns:
                    continue
                match = df_sc["Concepto"].astype(str).map(_normalize_text) == _normalize_text(metric_sel)
                if not match.any():
                    continue
                row = df_sc.loc[match].iloc[0]
                for col in df_sc.columns:
                    if isinstance(col, (int, float)) and 2025 <= int(col) <= 2035:
                        val = pd.to_numeric(row[col], errors="coerce")
                        if pd.notna(val):
                            rows.append({
                                "Escenario": scenario_name,
                                "Año": int(col),
                                "Valor": float(val),
                            })
            df_metric_sc = pd.DataFrame(rows)
            if df_metric_sc.empty:
                st.info("No fue posible extraer esa metrica del modelo financiero.")
            else:
                scenario_colors = {
                    "Sin financiamiento": "#4A607A",
                    "Con financiamiento base": "#1565A0",
                    "Con financiamiento pesimista": "#B54127",
                    "Con financiamiento optimista": "#1A7A3C",
                }
                fig_fin = px.line(
                    df_metric_sc,
                    x="Año",
                    y="Valor",
                    color="Escenario",
                    markers=True,
                    color_discrete_map=scenario_colors,
                    title=f"{metric_sel} proyectado por escenario",
                )
                apply_template_layout(fig_fin, height=360, yaxis_title="Valor")
                st.plotly_chart(fig_fin, use_container_width=True)

                latest = (
                    df_metric_sc[df_metric_sc["Año"] == df_metric_sc["Año"].max()]
                    .sort_values("Valor", ascending=False)
                )
                fig_bar_fin = px.bar(
                    latest,
                    x="Escenario",
                    y="Valor",
                    color="Escenario",
                    color_discrete_map=scenario_colors,
                    title=f"Cierre {int(df_metric_sc['Año'].max())}: {metric_sel}",
                )
                apply_template_layout(fig_bar_fin, height=280, showlegend=False)
                st.plotly_chart(fig_bar_fin, use_container_width=True)

        with c_fin_2:
            st.markdown("#### Drivers macro del modelo")
            macro_choice = st.selectbox(
                "Serie macro",
                ["EUR/MXN", "Inflacion Mexico", "Bono español 10Y"],
                key="crl_macro_sel",
            )
            if macro_choice == "EUR/MXN":
                df_macro = crl_data.get("eur_mxn", pd.DataFrame()).copy()
                y_title = "Tipo de cambio"
            elif macro_choice == "Inflacion Mexico":
                df_macro = crl_data.get("inflacion", pd.DataFrame()).copy()
                y_title = "Inflacion"
            else:
                df_macro = crl_data.get("bono", pd.DataFrame()).copy()
                y_title = "Tasa"

            if df_macro.empty:
                st.info("No se pudo extraer la serie macro seleccionada.")
            else:
                fig_macro = go.Figure()
                fig_macro.add_trace(go.Scatter(
                    x=df_macro["Fecha"],
                    y=df_macro["Valor"],
                    mode="lines",
                    line=dict(color="#1565A0", width=2.5),
                    fill="tozeroy",
                    fillcolor=hex_to_rgba("#1565A0", 0.12),
                    name=macro_choice,
                ))
                apply_template_layout(fig_macro, height=320, title=macro_choice, yaxis_title=y_title)
                st.plotly_chart(fig_macro, use_container_width=True)
                st.metric(
                    "Ultimo dato",
                    f"{df_macro['Valor'].iloc[-1]:,.4f}" if macro_choice == "EUR/MXN" else f"{df_macro['Valor'].iloc[-1]:.2%}"
                )

            unidades = []
            for esc_name, meta_vals in meta_esc.items():
                if "unidades_a_comprar" in meta_vals:
                    unidades.append({"Escenario": esc_name, "Unidades a comprar": meta_vals["unidades_a_comprar"]})
            if unidades:
                st.markdown("#### Expansion estimada")
                st.dataframe(pd.DataFrame(unidades), use_container_width=True, hide_index=True)

        st.markdown("---")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("#### Supuestos del modelo")
            if df_sup.empty:
                st.info("No se pudieron leer los supuestos.")
            else:
                st.dataframe(df_sup, use_container_width=True, hide_index=True, height=320)
        with t2:
            st.markdown("#### Datos operativos CLCircular")
            if df_ops.empty:
                st.info("No se pudieron leer los datos operativos.")
            else:
                st.dataframe(df_ops, use_container_width=True, hide_index=True, height=320)

        with st.expander("Tablas proyectadas completas"):
            if not escenarios:
                st.info("No hay tablas proyectadas disponibles.")
            else:
                esc_view = st.selectbox("Escenario a revisar", list(escenarios.keys()), key="crl_scenario_view")
                st.dataframe(escenarios[esc_view], use_container_width=True, hide_index=True, height=380)

#
# TAB COMERCIAL 
#
with tab_comercial:
    st.markdown("### Red de Inteligencia Logistica de CL Circular")
    st.markdown(
        '<div class="info-box"><b>Piloto ilustrativo:</b> el siguiente modulo muestra un dashboard demostrativo '
        'con <b>datos fijos</b> para visualizar como CL Circular podria agregar, añonimizar y convertir datos operativos '
        'en inteligencia compartida de rutas, cruces y transportistas. No representa informacion real operativa.</div>',
        unsafe_allow_html=True
    )

    df_demo_red = pd.DataFrame([
        {"Corredor": "Monterrey-Laredo", "Cruce": "Laredo", "Transportista": "Carrier A", "Envios": 128, "Golpes_pct": 4.7, "Aperturas_pct": 1.2, "Temp_alertas_pct": 2.4, "Transit_time_h": 19.6},
        {"Corredor": "Saltillo-Laredo", "Cruce": "Laredo", "Transportista": "Carrier B", "Envios": 96, "Golpes_pct": 3.1, "Aperturas_pct": 0.8, "Temp_alertas_pct": 1.9, "Transit_time_h": 17.8},
        {"Corredor": "Queretaro-El Paso", "Cruce": "El Paso", "Transportista": "Carrier C", "Envios": 82, "Golpes_pct": 5.4, "Aperturas_pct": 1.6, "Temp_alertas_pct": 2.8, "Transit_time_h": 22.4},
        {"Corredor": "Guanajuato-Laredo", "Cruce": "Laredo", "Transportista": "Carrier D", "Envios": 74, "Golpes_pct": 2.6, "Aperturas_pct": 0.5, "Temp_alertas_pct": 1.1, "Transit_time_h": 18.2},
        {"Corredor": "Puebla-El Paso", "Cruce": "El Paso", "Transportista": "Carrier E", "Envios": 61, "Golpes_pct": 6.2, "Aperturas_pct": 1.9, "Temp_alertas_pct": 3.5, "Transit_time_h": 24.1},
    ])
    df_demo_red["Incidencia_total_pct"] = (
        df_demo_red["Golpes_pct"] + df_demo_red["Aperturas_pct"] + df_demo_red["Temp_alertas_pct"]
    )
    df_demo_red["Tiempo_aduana_h"] = [6.4, 5.1, 8.7, 4.8, 9.3]
    df_demo_red["Tiempo_transito_sin_aduana_h"] = df_demo_red["Transit_time_h"] - df_demo_red["Tiempo_aduana_h"]
    df_demo_red["Indice_seguridad"] = np.clip(100 - (df_demo_red["Incidencia_total_pct"] * 8 + df_demo_red["Tiempo_aduana_h"] * 1.5), 0, 100)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Envios analizados", f"{int(df_demo_red['Envios'].sum())}")
    with d2:
        st.metric("Corredores monitoreados", f"{df_demo_red['Corredor'].nunique()}")
    with d3:
        st.metric("Cruce con mayor demora", df_demo_red.sort_values("Transit_time_h", ascending=False).iloc[0]["Cruce"])
    with d4:
        st.metric("Carrier mejor desempeno", df_demo_red.sort_values("Incidencia_total_pct").iloc[0]["Transportista"])

    r1, r2 = st.columns([1.1, 1.0])
    with r1:
        fig_red_corr = px.bar(
            df_demo_red.sort_values("Incidencia_total_pct", ascending=False),
            x="Corredor",
            y="Incidencia_total_pct",
            color="Cruce",
            text="Envios",
            title="Benchmark piloto: incidencia agregada por corredor",
            labels={"Incidencia_total_pct": "Incidencia total (%)"},
        )
        apply_template_layout(fig_red_corr, height=320)
        st.plotly_chart(fig_red_corr, use_container_width=True)
    with r2:
        fig_red_scatter = px.scatter(
            df_demo_red,
            x="Transit_time_h",
            y="Golpes_pct",
            size="Envios",
            color="Transportista",
            hover_name="Corredor",
            title="Piloto: tiempo de transito vs golpes",
            labels={"Transit_time_h": "Tiempo de transito (h)", "Golpes_pct": "Golpes (%)"},
        )
        apply_template_layout(fig_red_scatter, height=320)
        st.plotly_chart(fig_red_scatter, use_container_width=True)

    st.markdown("#### Inteligencia operativa del piloto")
    i1, i2, i3 = st.columns(3)
    with i1:
        df_aduana_sort = df_demo_red.sort_values("Tiempo_aduana_h", ascending=True)
        fig_aduana = go.Figure()
        fig_aduana.add_trace(go.Scatter(
            x=df_aduana_sort["Tiempo_aduana_h"],
            y=df_aduana_sort["Corredor"],
            mode="markers",
            marker=dict(
                size=16,
                color=df_aduana_sort["Tiempo_aduana_h"],
                colorscale="Blues",
                line=dict(color="#1A3355", width=1),
                showscale=False,
            ),
            hovertemplate="Corredor: %{y}<br>Horas en aduana: %{x:.1f}<extra></extra>",
            showlegend=False,
        ))
        for _, row in df_aduana_sort.iterrows():
            fig_aduana.add_shape(
                type="line",
                x0=0,
                x1=float(row["Tiempo_aduana_h"]),
                y0=row["Corredor"],
                y1=row["Corredor"],
                line=dict(color="#B8D0E8", width=3),
            )
        apply_template_layout(fig_aduana, height=300, title="Tiempo en aduanas por corredor", xaxis_title="Horas en aduana")
        st.plotly_chart(fig_aduana, use_container_width=True)
    with i2:
        df_cruces = (
            df_demo_red.groupby("Cruce", as_index=False)
            .agg(Tiempo_aduana_h=("Tiempo_aduana_h", "mean"), Envios=("Envios", "sum"))
            .sort_values("Tiempo_aduana_h", ascending=False)
        )
        fig_cruces = go.Figure(go.Pie(
            labels=df_cruces["Cruce"],
            values=df_cruces["Tiempo_aduana_h"],
            hole=0.55,
            textinfo="label+percent",
            marker_colors=["#1565A0", "#7B2FBE", "#C4621A", "#B54127"][:len(df_cruces)],
            hovertemplate="%{label}<br>Promedio: %{value:.1f} h<extra></extra>",
        ))
        apply_template_layout(fig_cruces, height=300, title="Cruces fronterizos mas lentos", showlegend=False)
        st.plotly_chart(fig_cruces, use_container_width=True)
    with i3:
        fig_seguras = px.scatter(
            df_demo_red.sort_values("Indice_seguridad", ascending=False),
            x="Indice_seguridad",
            y="Indice_seguridad",
            size="Envios",
            color="Cruce",
            hover_name="Corredor",
            title="Rutas mas seguras",
            labels={"Indice_seguridad": "Indice de seguridad"},
        )
        fig_seguras.update_yaxes(showticklabels=False, title="")
        apply_template_layout(fig_seguras, height=300)
        st.plotly_chart(fig_seguras, use_container_width=True)

    st.markdown("#### Reporte piloto de benchmarking")
    resumen_red = pd.DataFrame([
        {
            "Insight": "Corredor con mayor incidencia",
            "Resultado": df_demo_red.sort_values("Incidencia_total_pct", ascending=False).iloc[0]["Corredor"],
            "Lectura": "Priorizar revision operativa y validacion de transportista.",
        },
        {
            "Insight": "Cruce con mayor tiempo promedio",
            "Resultado": df_demo_red.groupby("Cruce")["Tiempo_aduana_h"].mean().sort_values(ascending=False).index[0],
            "Lectura": "Monitorear ventanas de cruce, saturacion y variabilidad de aduana.",
        },
        {
            "Insight": "Ruta mas segura",
            "Resultado": df_demo_red.sort_values("Indice_seguridad", ascending=False).iloc[0]["Corredor"],
            "Lectura": "Usar como referencia de desempeno para planificacion y benchmarking.",
        },
    ])
    st.dataframe(resumen_red, use_container_width=True, hide_index=True)

#
# TAB PROSPECTOS B2B
#
with tab_prospectos:
    st.markdown("### Prospectos Comerciales B2B")
    st.markdown(
        '<div class="info-box"><b>Objetivo:</b> priorizar empresas con mayor ajuste para contacto inmediato '
        'usando relevancia CL Circular, intensidad logistica y acceso directo (LinkedIn / sitio web).</div>',
        unsafe_allow_html=True
    )

    if df_contactos_all.empty:
        st.warning(f"No se encontro `{PROSPECTS_FILE}` o esta vacio.")
    else:
        c_filters_1, c_filters_2 = st.columns([2, 1])
        with c_filters_1:
            sector_contacto = st.selectbox(
                "Sector objetivo",
                options=list(dfs.keys()),
                format_func=lambda x: LABELS[x],
                key="contact_sector_select",
            )
        with c_filters_2:
            top_n_contactos = st.slider(
                "Top empresas",
                min_value=10,
                max_value=30,
                value=30,
                step=1,
                key="contact_top_n",
            )

        sector_target = SECTOR_CONTACT_MAP.get(sector_contacto, sector_contacto)
        df_sector_all = df_contactos_all[df_contactos_all["sector"] == sector_target].copy()
        df_contactos = obtener_contactos_sector(df_contactos_all, sector_contacto, top_n=top_n_contactos)

        if df_sector_all.empty:
            st.info("No hay empresas para este sector en el CSV.")
        else:
            df_sector_all["tiene_linkedin"] = df_sector_all["linkedin_empresa"].astype(str).str.contains("linkedin", case=False, na=False)
            df_sector_all["tiene_web"] = df_sector_all["sitio_web"].astype(str).str.len() > 3
            pct_linkedin = 100 * df_sector_all["tiene_linkedin"].mean()
            pct_web = 100 * df_sector_all["tiene_web"].mean()
            high_priority = int((df_sector_all["relev_rank"] >= 4).sum())

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Empresas del sector", f"{len(df_sector_all)}")
            with m2:
                st.metric("Alta/Muy alta prioridad", f"{high_priority}")
            with m3:
                st.metric("Con LinkedIn", f"{pct_linkedin:.0f}%")
            with m4:
                st.metric("Con sitio web", f"{pct_web:.0f}%")

            top_estado = df_sector_all["estado"].fillna("Sin dato").value_counts().idxmax()
            top_rel = df_sector_all["relevancia_clcircular"].fillna("Sin dato").value_counts().idxmax()
            st.markdown(
                f"**Resumen ejecutivo:** {len(df_sector_all)} empresas en `{sector_target}`; "
                f"predomina relevancia `{top_rel}` y la mayor concentracion geografica esta en `{top_estado}`."
            )

            c_vis_1, c_vis_2 = st.columns([2, 1])
            with c_vis_1:
                df_geo = df_sector_all.copy()
                df_geo["state_key"] = df_geo["estado"].apply(_norm_state_name)
                df_geo = df_geo[df_geo["state_key"].isin(MEX_STATE_COORDS)].copy()

                if df_geo.empty:
                    st.info("No hay estados mapeables para el sector seleccionado.")
                else:
                    df_map = (
                        df_geo.groupby("state_key")
                        .agg(
                            empresas=("empresa", "count"),
                            alta_prioridad=("relev_rank", lambda x: int((x >= 4).sum())),
                            ciudades=("ciudad", lambda x: ", ".join(sorted({str(v) for v in x.dropna()})[:3])),
                        )
                        .reset_index()
                    )
                    df_map["pct_alta"] = np.where(
                        df_map["empresas"] > 0,
                        100 * df_map["alta_prioridad"] / df_map["empresas"],
                        0,
                    )
                    df_map["lat"] = df_map["state_key"].map(lambda x: MEX_STATE_COORDS[x][0])
                    df_map["lon"] = df_map["state_key"].map(lambda x: MEX_STATE_COORDS[x][1])
                    df_map["estado"] = df_map["state_key"].str.title()

                    fig_map = px.scatter_geo(
                        df_map,
                        lat="lat",
                        lon="lon",
                        size="empresas",
                        color="pct_alta",
                        size_max=36,
                        color_continuous_scale="Blues",
                        hover_name="estado",
                        hover_data={
                            "empresas": True,
                            "alta_prioridad": True,
                            "pct_alta": ":.1f",
                            "ciudades": True,
                            "lat": False,
                            "lon": False,
                        },
                    )
                    fig_map.update_geos(
                        scope="north america",
                        projection_type="mercator",
                        lataxis_range=[14, 33],
                        lonaxis_range=[-119, -86],
                        showcountries=True,
                        countrycolor="#6B8CAE",
                        showland=True,
                        landcolor="#E8EEF4",
                        bgcolor="#FFFFFF",
                    )
                    apply_template_layout(
                        fig_map,
                        height=360,
                        title="Mapa de clientes potenciales por estado (sector seleccionado)",
                        margin=dict(t=55, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

            with c_vis_2:
                rel_counts = (
                    df_sector_all["relevancia_clcircular"]
                    .fillna("Sin dato")
                    .value_counts()
                )
                fig_rel = go.Figure(go.Pie(
                    labels=rel_counts.index.tolist(),
                    values=rel_counts.values.tolist(),
                    hole=0.5,
                    textinfo="percent",
                    marker_colors=px.colors.sequential.Blues[-len(rel_counts):],
                ))
                apply_template_layout(
                    fig_rel,
                    height=320,
                    title="Nivel de relevancia",
                    showlegend=True,
                )
                st.plotly_chart(fig_rel, use_container_width=True)

            st.markdown("#### Lista priorizada de contacto")
            cols_contacto = [
                "empresa", "subsector", "estado", "ciudad", "cargo_contacto",
                "relevancia_clcircular", "viajes_mes_estimados", "modo_transporte",
                "puertos_frontera", "linkedin_empresa", "sitio_web"
            ]
            for c in cols_contacto:
                if c not in df_contactos.columns:
                    df_contactos[c] = ""

            df_contactos = df_contactos.copy()
            df_contactos["prioridad"] = np.where(
                df_contactos["relev_rank"] >= 4, "Alta", "Media"
            )
            df_contactos_show = df_contactos[[
                "empresa", "prioridad", "relevancia_clcircular", "subsector",
                "estado", "ciudad", "cargo_contacto", "viajes_mes_estimados",
                "modo_transporte", "puertos_frontera", "linkedin_empresa", "sitio_web"
            ]].rename(columns={
                "empresa": "Empresa",
                "prioridad": "Prioridad",
                "relevancia_clcircular": "Relevancia",
                "subsector": "Subsector",
                "estado": "Estado",
                "ciudad": "Ciudad",
                "cargo_contacto": "Contacto sugerido",
                "viajes_mes_estimados": "Viajes/mes",
                "modo_transporte": "Transporte",
                "puertos_frontera": "Puertos/Frontera",
                "linkedin_empresa": "LinkedIn",
                "sitio_web": "Sitio web",
            })

            st.dataframe(
                df_contactos_show,
                use_container_width=True,
                hide_index=True,
                height=420,
                column_config={
                    "LinkedIn": st.column_config.LinkColumn("LinkedIn"),
                    "Sitio web": st.column_config.LinkColumn("Sitio web"),
                },
            )

            st.download_button(
                "Descargar lista priorizada (CSV)",
                data=df_contactos_show.to_csv(index=False).encode("utf-8"),
                mime="text/csv",
            )



# 
# FOOTER
# 
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#4A607A;font-size:12px;padding:8px 0">'
    ' CL Circular  Plataforma Analtica de Mercado  '
    'Reto Tecnolgico de Monterrey  Febrero 2026<br>'
    'Fuente de datos: UN Comtrade Database | '
    'Modelos: K-Means Clustering + PCA + Regresion Lineal (scikit-learn)'
    '</div>',
    unsafe_allow_html=True
)
