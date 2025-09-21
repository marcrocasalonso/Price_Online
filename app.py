# app.py
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import StringIO  # <-- para leer CSV desde string

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import time  # (solo una vez)

# =================== CONFIG & THEME ===================
st.set_page_config(
    page_title="POC (Price Online Comparator)",
    page_icon=None,
    layout="wide",
)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
    st.error("Falta OPENAI_API_KEY (en .env o entorno).")
    raise SystemExit("OPENAI_API_KEY is required")

from crewai import Agent, Task, Crew  # usa OPENAI_API_KEY

# ---- Global styles (limpio y minimal) ----
st.markdown("""
<style>
:root{
  --navy:#0b2b4c;
  --gray-900:#111827;
  --gray-700:#374151;
  --gray-500:#6b7280;
  --gray-300:#d1d5db;
  --gray-200:#e5e7eb;
  --gray-100:#f3f4f6;  /* gris sidebar */
  --white:#ffffff;
  --green:#2e7d32;
}

/* Tipografía */
html, body, [class*="css"]  { 
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; 
  font-size:10px;
  color:var(--gray-900);
}
h1 { font-size:20px; font-weight:700; color:var(--navy); }
h2 { font-size:14px; font-weight:700; color:var(--navy); }
h3 { font-size:12px; font-weight:600; color:var(--navy); }
small { font-size:10px; color:var(--gray-500); }

/* Sidebar izquierdo (nativo) */
section[data-testid="stSidebar"] > div {
  width: 280px;
  min-width: 280px;
}

/* Tarjetas */
.card {
  border: 1px solid var(--gray-200);
  border-radius: 12px; padding: 10px 12px;
  background: var(--white); box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  margin-bottom:10px;
}
.pill {
  display:inline-block; padding:2px 8px; border-radius:999px; 
  background: var(--gray-100); font-size:11px; font-weight:600; margin-right:6px; color:var(--gray-700);
}
.pill-green { background:#e8f5e9; color:var(--green); }
.pill-navy  { background:#e8eef6; color:var(--navy); }

/* Tabs */
.stTabs [role="tablist"] button[role="tab"] {
  font-size:13px;
}

/* ===== Chat tipo sidebar derecho (scroll independiente) ===== */
.chat-panel{
  position: sticky; 
  top: 12px;
  display:flex; flex-direction:column;
  background: var(--gray-100);
  border: 1px solid var(--gray-200);
  border-radius: 12px;
  height: calc(100vh - 24px);
  overflow: hidden;
}
.chat-head{ 
  padding:10px 12px; 
  border-bottom:1px solid var(--gray-200); 
  background:var(--gray-100);
  font-weight:600;
}
.chat-body{
  flex:1; 
  overflow:auto;               /* scroll independiente */
  padding:10px 12px;
}
.chat-input{
  border-top:1px solid var(--gray-200);
  padding:8px 10px;
  background:var(--gray-100);
}

/* Mensajes muy sencillos */
.msg{ 
  padding:8px 10px; border-radius:8px; margin:6px 0; 
  border:1px solid var(--gray-200); background:var(--white);
  word-wrap:break-word; overflow-wrap:anywhere;
}
.msg.user{ 
  background:var(--navy); color:white; border-color:var(--navy);
}

/* Toggle fila superior (alinear a la derecha) */
.header-row{
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# ======================== CONSTANTES / UTILS =======================
VALID_SECTIONS = [
    "General","0. Basic Data","1. Engine / Drive Train","2. Chassis / Wheels","3. Safety / Light",
    "4. Audio / Communication / Navigation","5. Comfort","6. Climate","7. Car Body / Exterior",
    "8. Seats","9. Interior","10. Service / Tax", "Indices and interim values"
]
def norm_section_label(s: Optional[str]) -> str:
    """
    Normaliza etiquetas de sección para comparación robusta:
    - lower case
    - colapsa espacios
    - reemplaza '&' por 'and'
    """
    if s is None:
        return ""
    t = re.sub(r"\s+", " ", str(s)).strip().lower()
    t = t.replace("&", "and")
    return t

# Mapeos canónicos (para guardar siempre la etiqueta estándar)
CANONICAL_BY_NORM = {norm_section_label(s): s for s in VALID_SECTIONS}
CANONICAL_SET_NORM = set(CANONICAL_BY_NORM.keys())

CANONICAL_SET = set(VALID_SECTIONS)
EMPTY_LIKE_RE = re.compile(r"^\s*(?:none|null|nan|n/?a|n\.a\.|—|-|–|\.|—\s*—)\s*$", re.IGNORECASE)

def clean_str(x: Optional[str]) -> Optional[str]:
    if pd.isna(x):
        return None
    s = re.sub(r"\s+", " ", str(x)).strip()
    return s if s else None

def is_empty_value(v: Optional[str]) -> bool:
    if v is None:
        return True
    return bool(EMPTY_LIKE_RE.match(str(v)))

def now_ingested_at() -> str:
    return datetime.now().isoformat(timespec="seconds")

def canonical_market(name: str) -> str:
    if not name:
        return "Unknown"
    s = re.sub(r"\s+", " ", name).strip()
    s_cap = " ".join(w.capitalize() for w in s.split())
    MAP = {
        "Ger":"Germany","De":"Germany","Deu":"Germany","Germany":"Germany",
        "Es":"Spain","Esp":"Spain","Spain":"Spain",
        "Fr":"France","Fra":"France","France":"France",
        "It":"Italy","Ita":"Italy","Italy":"Italy",
        "At":"Austria","Aut":"Austria","Austria":"Austria",
        "Pt":"Portugal","Prt":"Portugal","Portugal":"Portugal",
        "Uk":"United Kingdom","Gbr":"United Kingdom","Gb":"United Kingdom","United Kingdom":"United Kingdom",
    }
    return MAP.get(s_cap, s_cap)

def parse_numeric_and_unit(x) -> Tuple[float, Optional[str]]:
    if pd.isna(x) or x is None:
        return (np.nan, None)
    s = str(x).strip()
    if s.lower() in {"yes","y","true","si","sí","1"}: return (1.0, "bool")
    if s.lower() in {"no","n","false","0"}: return (0.0, "bool")
    unit_guess = None
    for up in [r"kw", r"ps", r"hp", r"nm", r"km/h", r"wh/km", r"g/km",
               r"l/100km", r"€", r"eur", r"mm", r"cm", r"kg", r"inch", r"%"]:
        mm = re.search(up, s, flags=re.IGNORECASE)
        if mm:
            unit_guess = mm.group(0).lower()
            break
    m = re.search(r"-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?", s)
    if not m:
        return (np.nan, unit_guess)
    num = m.group(0)
    if "," in num and "." in num:
        num = num.replace(".", "").replace(",", ".")
    elif "," in num and "." not in num:
        num = num.replace(",", ".")
    try:
        return (float(num), unit_guess)
    except Exception:
        return (np.nan, unit_guess)

def split_brand_model(v: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if v is None:
        return (None, None)
    parts = [p.strip() for p in str(v).split("/")]
    if len(parts) >= 2:
        return parts[0], "/".join(parts[1:]).strip()
    toks = str(v).split()
    if len(toks) >= 2:
        return toks[0], " ".join(toks[1:])
    return (None, v)

def find_header_row_anywhere(df: pd.DataFrame) -> int:
    for i in range(df.shape[0]):
        rowvals = df.iloc[i].astype(str).str.strip().str.lower().tolist()
        if "feature/attribute" in rowvals:
            return i
    counts = df.notna().sum(axis=1)
    return int(counts.idxmax())

def extract_market_from_first_row(raw: pd.DataFrame) -> str:
    if raw.shape[0] == 0:
        return "Unknown"
    first_row_vals = raw.iloc[0].astype(str).fillna("").tolist()
    joined = " | ".join(first_row_vals)
    m = re.search(r"Adjustment\s*-\s*([^-–—|]+?)\s*-\s*", joined, flags=re.IGNORECASE)
    if m:
        return canonical_market(m.group(1))
    m2 = re.search(r"\b(germany|spain|france|italy|austria|portugal|united kingdom|uk)\b", joined, flags=re.IGNORECASE)
    return canonical_market(m2.group(1)) if m2 else "Unknown"

# ================== PARSER HOJA / EXCEL → CSV ==================
def parse_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, dtype=str)
    if raw.empty:
        return pd.DataFrame()
    market_sheet = extract_market_from_first_row(raw)
    raw = raw.dropna(how="all").dropna(how="all", axis=1).reset_index(drop=True)
    if raw.empty or raw.shape[1] < 3:
        return pd.DataFrame()
    hrow = find_header_row_anywhere(raw)
    header = list(raw.iloc[hrow, :])
    if len(header) >= 1: header[0] = "Feature/Attribute"
    if len(header) >= 2: header[1] = "Adjustment value"
    for j in range(2, len(header)):
        v = header[j]
        if pd.isna(v) or str(v).strip()=="" or str(v).lower()=="nan":
            header[j] = header[j-1]
    data = raw.iloc[hrow+1:, :len(header)].copy()
    data.columns = header
    data = data.dropna(how="all").reset_index(drop=True)
    fa = data["Feature/Attribute"].apply(clean_str)

    # --- NUEVO: detección robusta y mapeo a etiqueta canónica
    fa_norm = fa.apply(norm_section_label)
    is_sec = fa_norm.isin(CANONICAL_SET_NORM)

    data["Section"] = pd.Series(pd.NA, index=data.index, dtype="object")
    # guardamos SIEMPRE la etiqueta canónica (p. ej. convierte "11. Indices & interim values" a "11. Indices and interim values")
    data.loc[is_sec, "Section"] = fa_norm.loc[is_sec].map(CANONICAL_BY_NORM)

    # propagate / rellenar
    data["Section"] = data["Section"].ffill().fillna("summary")

    # elimina filas que eran cabeceras de sección
    data = data[~is_sec].copy().reset_index(drop=True)

    rows: List[Tuple] = []
    for col_idx in range(2, len(header)):
        base = clean_str(header[col_idx])
        if base is None or base.lower() == "adjustment value":
            continue
        sub_idx = 1
        for prev in range(2, col_idx):
            if clean_str(header[prev]) == base:
                sub_idx += 1
        series = data.iloc[:, col_idx]
        for row_i in range(len(data)):
            feat = clean_str(data.iloc[row_i, 0])
            sect = clean_str(data.loc[row_i, "Section"])
            val  = clean_str(series.iloc[row_i])
            rows.append((market_sheet, sheet_name, sect, feat, base, sub_idx, val))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "market","Sheet","Section","Feature/Attribute","Variant","sub_idx","Value"
    ])
    pu = df["Value"].apply(parse_numeric_and_unit)
    df["value_num"] = [x[0] for x in pu]
    df["unit_guess"] = [x[1] for x in pu]
    bm = df["Variant"].apply(split_brand_model)
    df["variant_brand"]  = [x[0] for x in bm]
    df["variant_model"]  = [x[1] for x in bm]
    df["is_reference"]   = df["variant_brand"].str.upper().isin(["SEAT","CUPRA"])
    df["ingested_at"] = now_ingested_at()
    df = df[[
        "market","Sheet","Section","Feature/Attribute","Variant",
        "variant_brand","variant_model","sub_idx","is_reference",
        "Value","value_num","unit_guess","ingested_at"
    ]]
    return df

def build_csv_from_excel(xlsx_path: str, out_csv_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    def norm_simple(s: str) -> str:
        return re.sub(r"[\s_-]+", "", str(s).lower())
    skip = {s for s in xls.sheet_names if norm_simple(s) in {"indexoverview","additions"}}
    frames: List[pd.DataFrame] = []
    for s in xls.sheet_names:
        if s in skip:
            continue
        try:
            df = parse_sheet(xls, s)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            frames.append(pd.DataFrame({
                "market":["Unknown"],"Sheet":[s],"Section":["summary"],
                "Feature/Attribute":[f"ERROR: {e}"],
                "Variant":[None],"variant_brand":[None],"variant_model":[None],
                "sub_idx":[np.nan],"is_reference":[False],
                "Value":[None],"value_num":[np.nan],"unit_guess":[None],
                "ingested_at":[now_ingested_at()],
            }))
    out = (pd.concat(frames, ignore_index=True)
           if frames else
           pd.DataFrame(columns=[
               "market","Sheet","Section","Feature/Attribute","Variant",
               "variant_brand","variant_model","sub_idx","is_reference",
               "Value","value_num","unit_guess","ingested_at"
           ]))
    out.to_csv(out_csv_path, index=False)
    return out

# ==================== COMPARADOR / INFORME ====================
def list_variants(df: pd.DataFrame) -> List[str]:
    return sorted([v for v in df["Variant"].dropna().unique()]) if "Variant" in df.columns else []

def sections_in_df(df: pd.DataFrame) -> List[str]:
    secs = [s for s in df["Section"].dropna().unique()] if "Section" in df.columns else []
    order = {s:i for i,s in enumerate(VALID_SECTIONS)}
    return sorted(secs, key=lambda s: order.get(s, 999))

def collapse_two_values(g: pd.DataFrame) -> Tuple[Optional[str],Optional[str]]:
    g = g.sort_values("sub_idx", kind="mergesort")
    v1 = clean_str(g["Value"].iloc[0]) if len(g) >= 1 else None
    v2 = clean_str(g["Value"].iloc[1]) if len(g) >= 2 else None
    return (None if is_empty_value(v1) else v1,
            None if is_empty_value(v2) else v2)

def build_variant_matrix(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    sub = df[df["Variant"] == variant][["Section","Feature/Attribute","sub_idx","Value"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Section","Feature/Attribute","value_1","value_2"])
    rows = []
    for (sec, feat), g in sub.groupby(["Section","Feature/Attribute"], dropna=True):
        v1, v2 = collapse_two_values(g)
        if v1 is None and v2 is not None:
            v1, v2 = v2, None
        if v1 is None and v2 is None:
            continue
        rows.append((sec, feat, v1, v2))
    out = pd.DataFrame(rows, columns=["Section","Feature/Attribute","value_1","value_2"]).drop_duplicates()
    return out

def compare_variants_by_section(df: pd.DataFrame, v1: str, v2: str, sections: Optional[List[str]]=None) -> Dict[str, List[Tuple[str, str, str]]]:
    secs = sections if sections else sections_in_df(df)
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    m1 = build_variant_matrix(df, v1).set_index(["Section","Feature/Attribute"])
    m2 = build_variant_matrix(df, v2).set_index(["Section","Feature/Attribute"])
    if m1.empty or m2.empty: return {}
    common_idx = m1.index.intersection(m2.index)
    for sec in secs:
        feats = [idx for idx in common_idx if idx[0] == sec]
        if not feats:
            continue
        diffs = []
        for _, feat in feats:
            a = " | ".join([x for x in [m1.at[(sec, feat),"value_1"], m1.at[(sec, feat),"value_2"]] if x])
            b = " | ".join([x for x in [m2.at[(sec, feat),"value_1"], m2.at[(sec, feat),"value_2"]] if x])
            if a != b:
                diffs.append((feat, a if a else "—", b if b else "—"))
        if diffs:
            out[sec] = diffs
    return out

def render_differences_md(diffs: Dict[str, List[Tuple[str,str,str]]], v1: str, v2: str, sheet: Optional[str]=None) -> str:
    if not diffs:
        t = f"# Informe de diferencias\n\nNo se han encontrado diferencias entre {v1} y {v2}."
        if sheet: t += f"\n\nHoja: {sheet}"
        return t
    lines = [f"# Informe de diferencias\n\nA: {v1}\n\nB: {v2}\n"]
    if sheet: lines.append(f"Hoja: {sheet}")
    for sec, items in diffs.items():
        lines.append(f"\n## {sec}")
        for feat, a, b in items:
            lines.append(f"- {feat}\n  - A: {a}\n  - B: {b}")
    return "\n".join(lines)

def diffs_to_df(diffs: Dict[str, List[Tuple[str,str,str]]], v1: str, v2: str) -> pd.DataFrame:
    rows = []
    for sec, items in diffs.items():
        for feat, a, b in items:
            rows.append({
                "Section": sec,
                "Feature/Attribute": feat,
                "A_Variant": v1,
                "A_Value": a,
                "B_Variant": v2,
                "B_Value": b
            })
    return pd.DataFrame(rows)

# ============== RAG + LLM para chatbot ==============
def build_chat_context(df: pd.DataFrame, v1: str, v2: str, sections: Optional[List[str]], user_q: str, top_k: int = 160) -> pd.DataFrame:
    ctx = df[(df["Variant"] == v1) | (df["Variant"] == v2)].copy()
    if sections:
        ctx = ctx[ctx["Section"].isin(sections)]
    if ctx.empty:
        return ctx
    ctx = ctx[~ctx["Value"].apply(is_empty_value)].copy()
    toks = [t for t in re.split(r"[^\w%/\.]+", user_q.lower()) if t]
    def score_row(r):
        text = " ".join([str(r.get(c) or "") for c in ["Section","Feature/Attribute","Variant","Value"]]).lower()
        return sum(1 for t in toks if t in text)
    ctx["__score"] = ctx.apply(score_row, axis=1)
    return ctx.sort_values(["__score","Section","Feature/Attribute","Variant","sub_idx"],
                           ascending=[False, True, True, True, True]).head(top_k)

def render_context_as_text(df_ctx: pd.DataFrame) -> str:
    lines = []
    for _, r in df_ctx.iterrows():
        sub = int(r["sub_idx"]) if pd.notna(r["sub_idx"]) else 1
        lines.append(f"[{r['Section']}] {r['Feature/Attribute']} — {r['Variant']} (sub{sub}): {r['Value']}")
    return "\n".join(lines)

def llm_chat_answer(context_text: str, user_q: str, filtered_csv_text: str, report_md: str) -> str:
    prompt = f"""
Eres un experto en automoción. Responde con estilo sobrio y directo.
Usa SOLO la información que te paso (no traigas datos externos).

PREGUNTA:
{user_q}

CSV FILTRADO (A/B):
{filtered_csv_text}
- Cuando la columna sub_idx = 1 los valores de la columna "value" corresponden a los valores de ajuste que incrementan o disminuyen al precio del coche. La suma total equivale de esos valores descartando las secciones de "General","0. Basic Data" y "Indices and interim values" corresponde al adjusted price del coche.
- Cuando la columna sub_idx = 2 los valores de la columna "value" corresponden a valores de las caracteristicas o features del coche.

INFORME COMPARATIVO:
{report_md}

EXTRACTO RAG:
{context_text}

Instrucciones:
- Busca y responde lo que te piden, si no lo encuentras dilo y propon una pregunta que te sirva.
- Sintetiza, no listes filas crudas.
- Solo cita cifras presentes en el contexto.
- Si no hay cifra, usa comparativa cualitativa.
- Sé directo y evita enrollarte.
- Usa bullet points si enumeras listas, características o cualquier cosa que ayude al análisis.

"""
    agent = Agent(
        role="Comparador de automoción",
        goal="Respuesta clara usando solo el contexto (CSV filtrado + informe + extracto)",
        backstory="Especialista en fichas técnicas.",
        allow_delegation=False,
        verbose=False,
    )
    task = Task(description=prompt, agent=agent, expected_output="Respuesta clara, útil y directa.")
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return str(crew.kickoff())

# ========= util: nombre de archivo dinámico =========
def _get_df_for_chat() -> pd.DataFrame:
    """
    Devuelve un dataframe con las filas que el chatbot debe usar como contexto,
    funcionando en modo 1 excel (df_struct_main) o 2 excels (df_struct_a/b).
    """
    mode = st.session_state.get("mode_compare", "same")

    # 1) Modo 1 excel
    if mode == "same":
        df_single = st.session_state.get("df_struct_main")
        if df_single is not None:
            sel_sheet = st.session_state.get("sel_sheet_main", "(todas)")
            if sel_sheet and sel_sheet != "(todas)":
                df_single = df_single[df_single["Sheet"] == sel_sheet]
            return df_single

    # 2) Modo 2 excels
    dfa = st.session_state.get("df_struct_a")
    dfb = st.session_state.get("df_struct_b")
    if dfa is not None and dfb is not None:
        sa = st.session_state.get("sel_sheet_a", "(todas)")
        sb = st.session_state.get("sel_sheet_b", "(todas)")
        if sa and sa != "(todas)":
            dfa = dfa[dfa["Sheet"] == sa]
        if sb and sb != "(todas)":
            dfb = dfb[dfb["Sheet"] == sb]
        return pd.concat([dfa, dfb], ignore_index=True)

    # 3) Fallback vacío
    return pd.DataFrame()

def stream_openai_answer(context_text: str,
                         user_q: str,
                         filtered_csv_text: str,
                         report_md: str,
                         stream_pl,
                         model: str = "gpt-4o",
                         temperature: float = 0.2) -> str:
    """
    Genera una respuesta con OpenAI en streaming y la va dibujando en 'stream_pl' (st.empty()).
    Devuelve el texto final para guardarlo en el historial.
    """
    client = OpenAI()  # usa OPENAI_API_KEY del entorno

    system_msg = (
        "Eres un experto en automoción. Responde sobrio y directo. "
        "Usa SOLO la información proporcionada (CSV filtrado, informe y extracto). "
        "Si falta un dato, dilo y sugiere qué revisar. "
        "Usa bullets cuando ayuden al análisis."
    )

    user_content = f"""
PREGUNTA:
{user_q}

CSV FILTRADO (A/B):
{filtered_csv_text}

INFORME COMPARATIVO:
{report_md}

EXTRACTO RAG:
{context_text}
"""

    rendered = ""
    stream_pl.markdown("<div class='msg'>Escribiendo…</div>", unsafe_allow_html=True)

    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                rendered += piece
                stream_pl.markdown(f"<div class='msg'>{rendered}</div>", unsafe_allow_html=True)
                time.sleep(0.005)

        return rendered.strip() if rendered.strip() else "No hay datos relevantes para responder."
    except Exception as e:
        stream_pl.markdown(f"<div class='msg'>[Error de streaming: {e}]</div>", unsafe_allow_html=True)
        return "Ha habido un problema generando la respuesta en streaming."

def stream_markdown(answer: str, placeholder, role_class: str = "msg", delay: float = 0.012, chunk_size: int = 10):
    rendered = ""
    for i in range(0, len(answer), chunk_size):
        rendered += answer[i:i+chunk_size]
        placeholder.markdown(f"<div class='{role_class}'>{rendered}</div>", unsafe_allow_html=True)
        time.sleep(delay)
    return rendered

def build_export_filename(df: pd.DataFrame) -> str:
    ref_df = df[df["is_reference"] == True]
    model = None
    if not ref_df.empty:
        model = ref_df["variant_model"].dropna().iloc[0] or ref_df["Variant"].dropna().iloc[0]
    else:
        if "variant_model" in df.columns and df["variant_model"].notna().any():
            model = df["variant_model"].dropna().iloc[0]
        elif "Variant" in df.columns and df["Variant"].notna().any():
            model = df["Variant"].dropna().iloc[0]
    model_slug = re.sub(r"[^a-z0-9]+", "_", (model or "model").lower()).strip("_")
    market = (df["market"].dropna().iloc[0] if "market" in df.columns and df["market"].notna().any() else "unknown")
    market_slug = re.sub(r"[^a-z0-9]+", "_", str(market).lower()).strip("_")
    ts = (df["ingested_at"].dropna().iloc[0] if "ingested_at" in df.columns and df["ingested_at"].notna().any()
          else datetime.now().isoformat(timespec="seconds"))
    ts_clean = ts
    return f"{model_slug}_{market_slug}_{ts_clean}.csv"

# ========= helpers UI/session =========
def _file_signature(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    try:
        size = uploaded_file.size
    except Exception:
        size = len(uploaded_file.getbuffer())
    return f"{uploaded_file.name}:{size}"

def _reset_state():
    st.session_state["chat_history_agent"] = []
    st.session_state["last_report_md"] = ""
    st.session_state["last_diffs"] = None
    st.session_state["last_highlights"] = ""
    st.session_state["filtered_csv_text"] = ""
    st.session_state["v1_label_display"] = ""
    st.session_state["v2_label_display"] = ""

# ============================ UI ============================
st.markdown("## Price Online Comparator (POC)")
st.caption("Digitaliza el Price Online, compara variantes y realiza preguntas.")

# —— Session state base ——
for key, default in [
    ("mode_compare", "same"),  # 'same' | 'cross'
    ("last_upload_sig_main", None),
    ("last_upload_sig_a", None),
    ("last_upload_sig_b", None),
    ("chat_history_agent", []),
    ("last_report_md", ""),
    ("last_diffs", None),
    ("last_highlights", ""),
    ("sel_sheet_main", "(todas)"),
    ("sel_sheet_a", "(todas)"),
    ("sel_sheet_b", "(todas)"),
    ("sel_sections", []),
    ("v1_main", None),
    ("v2_main", None),
    ("v1_a", None),
    ("v2_b", None),
    ("filtered_csv_text", ""),
    ("chat_open", True),
    ("v1_label_display", ""),
    ("v2_label_display", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ===== Sidebar izquierdo: modo y carga =====
with st.sidebar:
    st.subheader("Modo de comparación")
    mode = st.radio("Elige modo", ["Un solo Excel", "Dos Excels"], horizontal=False,
                    index=0 if st.session_state["mode_compare"] == "same" else 1)
    st.session_state["mode_compare"] = "same" if mode == "Un solo Excel" else "cross"

    st.subheader("Cargar Excel")
    if st.session_state["mode_compare"] == "same":
        xlsx_main = st.file_uploader("Excel (único)", type=["xlsx"], key="uploader_main")
        current_sig_main = _file_signature(xlsx_main)
        if current_sig_main != st.session_state["last_upload_sig_main"]:
            st.session_state["last_upload_sig_main"] = current_sig_main
            _reset_state()
    else:
        xlsx_a = st.file_uploader("Excel A", type=["xlsx"], key="uploader_a")
        xlsx_b = st.file_uploader("Excel B", type=["xlsx"], key="uploader_b")
        current_sig_a = _file_signature(xlsx_a)
        current_sig_b = _file_signature(xlsx_b)
        if current_sig_a != st.session_state["last_upload_sig_a"] or current_sig_b != st.session_state["last_upload_sig_b"]:
            st.session_state["last_upload_sig_a"] = current_sig_a
            st.session_state["last_upload_sig_b"] = current_sig_b
            _reset_state()

# ====== Carga/parsing según modo ======
df_struct_main = None
df_struct_a = None
df_struct_b = None

if st.session_state["mode_compare"] == "same":
    if not xlsx_main:
        st.info("Sube el Excel para continuar.")
        st.stop()
    with st.spinner("Procesando Excel…"):
        tmp_xlsx_path = Path("uploaded_main.xlsx")
        with open(tmp_xlsx_path, "wb") as f:
            f.write(xlsx_main.getbuffer())
        out_csv_path = Path("specs_basic_flat_main.csv")
        df_struct_main = build_csv_from_excel(str(tmp_xlsx_path), str(out_csv_path))
    if df_struct_main.empty:
        st.error("No se pudo extraer información del Excel.")
        st.stop()
    export_name_main = build_export_filename(df_struct_main)
    # 🔑 Guarda en sesión para el chat/helper
    st.session_state["df_struct_main"] = df_struct_main
else:
    if not (xlsx_a and xlsx_b):
        st.info("Sube ambos Excels (A y B) para continuar.")
        st.stop()
    with st.spinner("Procesando Excel A…"):
        tmp_xlsx_path_a = Path("uploaded_a.xlsx")
        with open(tmp_xlsx_path_a, "wb") as f:
            f.write(xlsx_a.getbuffer())
        out_csv_path_a = Path("specs_basic_flat_a.csv")
        df_struct_a = build_csv_from_excel(str(tmp_xlsx_path_a), str(out_csv_path_a))
    with st.spinner("Procesando Excel B…"):
        tmp_xlsx_path_b = Path("uploaded_b.xlsx")
        with open(tmp_xlsx_path_b, "wb") as f:
            f.write(xlsx_b.getbuffer())
        out_csv_path_b = Path("specs_basic_flat_b.csv")
        df_struct_b = build_csv_from_excel(str(tmp_xlsx_path_b), str(out_csv_path_b))
    if df_struct_a.empty or df_struct_b.empty:
        st.error("No se pudo extraer información de alguno de los Excels.")
        st.stop()
    export_name_a = build_export_filename(df_struct_a)
    export_name_b = build_export_filename(df_struct_b)
    # 🔑 Guarda en sesión para el chat/helper
    st.session_state["df_struct_a"] = df_struct_a
    st.session_state["df_struct_b"] = df_struct_b

# ===== Sidebar: Configuración y acciones =====
with st.sidebar:
    st.divider()
    with st.expander("Configuración", expanded=True):
        if st.session_state["mode_compare"] == "same":
            # Sheets
            sheets_main = sorted(df_struct_main["Sheet"].dropna().unique().tolist())
            sel_sheet_options_main = ["(todas)"] + sheets_main
            idx_sheet_main = sel_sheet_options_main.index(st.session_state["sel_sheet_main"]) if st.session_state["sel_sheet_main"] in sel_sheet_options_main else 0
            sel_sheet_main = st.selectbox("Hoja (único Excel)", sel_sheet_options_main, index=idx_sheet_main)
            st.session_state["sel_sheet_main"] = sel_sheet_main

            # Secciones
            sections_avail_main = sections_in_df(df_struct_main)
            default_sections = sections_avail_main if not st.session_state["sel_sections"] else st.session_state["sel_sections"]
            sel_sections = st.multiselect("Secciones", sections_avail_main, default=default_sections)
            st.session_state["sel_sections"] = sel_sections

            # Variantes
            df_view_for_variants = df_struct_main if sel_sheet_main == "(todas)" else df_struct_main[df_struct_main["Sheet"] == sel_sheet_main]
            variants_main = list_variants(df_view_for_variants)
            if len(variants_main) < 2:
                st.warning("No hay suficientes variantes en la hoja seleccionada.")
            def _safe_index(opts, value, fallback_idx):
                return opts.index(value) if (value in opts) else min(fallback_idx, max(0, len(opts)-1)) if opts else 0
            idx_v1 = _safe_index(variants_main, st.session_state["v1_main"], 0)
            idx_v2 = _safe_index(variants_main, st.session_state["v2_main"], 1 if len(variants_main) > 1 else 0)
            v1_main = st.selectbox("Variante A", variants_main, index=idx_v1 if variants_main else 0)
            v2_main = st.selectbox("Variante B", variants_main, index=idx_v2 if variants_main else 0)
            st.session_state["v1_main"] = v1_main
            st.session_state["v2_main"] = v2_main

            # CSV filtrado (A/B)
            df_filtered_sidebar = df_view_for_variants[(df_view_for_variants["Variant"] == v1_main) | (df_view_for_variants["Variant"] == v2_main)].copy()
            df_filtered_sidebar = df_filtered_sidebar[~df_filtered_sidebar["Value"].apply(is_empty_value)]
            st.session_state["filtered_csv_text"] = df_filtered_sidebar.to_csv(index=False) if not df_filtered_sidebar.empty else ""

            st.markdown("---")
            gen = st.button("Comparar y redactar informe", use_container_width=True)

            # Descargas
            st.download_button(
                "Descargar CSV estructurado",
                data=df_struct_main.to_csv(index=False).encode("utf-8"),
                file_name=export_name_main,
                mime="text/csv",
                use_container_width=True,
                key="dl_structured_main"
            )
            if not df_filtered_sidebar.empty:
                st.download_button(
                    "Descargar CSV filtrado (A/B)",
                    data=df_filtered_sidebar.to_csv(index=False).encode("utf-8"),
                    file_name=f"ab_{export_name_main}",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_filtered_main_{sel_sheet_main}_{v1_main}_{v2_main}"
                )
            if st.session_state.get("last_report_md"):
                st.download_button(
                    "Descargar informe (.md)",
                    data=st.session_state["last_report_md"].encode("utf-8"),
                    file_name=f"informe_{export_name_main.replace('.csv','.md')}",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"dl_report_main_{sel_sheet_main}_{v1_main}_{v2_main}"
                )

        else:
            # Sheets por cada Excel
            sheets_a = sorted(df_struct_a["Sheet"].dropna().unique().tolist())
            sheets_b = sorted(df_struct_b["Sheet"].dropna().unique().tolist())
            opt_a = ["(todas)"] + sheets_a
            opt_b = ["(todas)"] + sheets_b
            idx_a = opt_a.index(st.session_state["sel_sheet_a"]) if st.session_state["sel_sheet_a"] in opt_a else 0
            idx_b = opt_b.index(st.session_state["sel_sheet_b"]) if st.session_state["sel_sheet_b"] in opt_b else 0
            sel_sheet_a = st.selectbox("Hoja Excel A", opt_a, index=idx_a)
            sel_sheet_b = st.selectbox("Hoja Excel B", opt_b, index=idx_b)
            st.session_state["sel_sheet_a"] = sel_sheet_a
            st.session_state["sel_sheet_b"] = sel_sheet_b

            # Secciones comunes (intersección; si no hay, unión)
            sections_a = set(sections_in_df(df_struct_a))
            sections_b = set(sections_in_df(df_struct_b))
            common_sections = sorted(list(sections_a & sections_b)) or sorted(list(sections_a | sections_b))
            default_sections = common_sections if not st.session_state["sel_sections"] else st.session_state["sel_sections"]
            sel_sections = st.multiselect("Secciones (comunes)", common_sections, default=default_sections)
            st.session_state["sel_sections"] = sel_sections

            # Variantes independientes
            df_view_a = df_struct_a if sel_sheet_a == "(todas)" else df_struct_a[df_struct_a["Sheet"] == sel_sheet_a]
            df_view_b = df_struct_b if sel_sheet_b == "(todas)" else df_struct_b[df_struct_b["Sheet"] == sel_sheet_b]
            variants_a = list_variants(df_view_a)
            variants_b = list_variants(df_view_b)

            def _safe_index(opts, value, fallback_idx):
                return opts.index(value) if (value in opts) else min(fallback_idx, max(0, len(opts)-1)) if opts else 0
            idx_a_v = _safe_index(variants_a, st.session_state["v1_a"], 0)
            idx_b_v = _safe_index(variants_b, st.session_state["v2_b"], 0 if len(variants_b) else 0)
            v1_a = st.selectbox("Variante A (Excel A)", variants_a, index=idx_a_v if variants_a else 0)
            v2_b = st.selectbox("Variante B (Excel B)", variants_b, index=idx_b_v if variants_b else 0)
            st.session_state["v1_a"] = v1_a
            st.session_state["v2_b"] = v2_b

            # CSV filtrado concatenado A+B (solo filas relevantes)
            df_a_sel = df_view_a[df_view_a["Variant"] == v1_a].copy()
            df_b_sel = df_view_b[df_view_b["Variant"] == v2_b].copy()
            df_ab = pd.concat([df_a_sel, df_b_sel], ignore_index=True)
            df_ab = df_ab[~df_ab["Value"].apply(is_empty_value)]
            st.session_state["filtered_csv_text"] = df_ab.to_csv(index=False) if not df_ab.empty else ""

            st.markdown("---")
            gen = st.button("Comparar y redactar informe", use_container_width=True)

            # Descargas
            st.download_button(
                "Descargar CSV A (estructurado)",
                data=df_struct_a.to_csv(index=False).encode("utf-8"),
                file_name=export_name_a,
                mime="text/csv",
                use_container_width=True,
                key="dl_structured_a"
            )
            st.download_button(
                "Descargar CSV B (estructurado)",
                data=df_struct_b.to_csv(index=False).encode("utf-8"),
                file_name=export_name_b,
                mime="text/csv",
                use_container_width=True,
                key="dl_structured_b"
            )
            if not df_ab.empty:
                st.download_button(
                    "Descargar CSV filtrado (A+B)",
                    data=df_ab.to_csv(index=False).encode("utf-8"),
                    file_name=f"ab_{export_name_a.replace('.csv','')}__{export_name_b}",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_filtered_ab_{sel_sheet_a}_{sel_sheet_b}_{v1_a}_{v2_b}"
                )

# ===== Ejecuta comparación (informe + highlights) =====
if 'gen' in locals() and gen:
    sel_sections = st.session_state["sel_sections"] or None

    if st.session_state["mode_compare"] == "same":
        sel_sheet_main = st.session_state["sel_sheet_main"]
        df_view_for_compare = df_struct_main if sel_sheet_main == "(todas)" else df_struct_main[df_struct_main["Sheet"] == sel_sheet_main]
        v1 = st.session_state["v1_main"]
        v2 = st.session_state["v2_main"]

        diffs = compare_variants_by_section(df_view_for_compare, v1, v2, sections=sel_sections)
        report_md = render_differences_md(diffs, v1, v2, None if sel_sheet_main == "(todas)" else sel_sheet_main)
        st.session_state["last_diffs"] = diffs
        st.session_state["last_report_md"] = report_md
        st.session_state["v1_label_display"] = v1
        st.session_state["v2_label_display"] = v2

        # Highlights
        if diffs:
            hl_prompt = f"""
Eres un experto en automoción. Lee este informe de diferencias y genera un resumen breve y directo,
resaltando los puntos más importantes que un usuario debería conocer.

INFORME DE DIFERENCIAS:
{report_md}

Instrucciones:
- Responde en formato conciso pero redactado.
- Usa bullet points para destacar lo más diferenciador.
- Devuelve: resumen, diferencias clave y recomendación final.
"""
            hl_agent = Agent(role="Analista de highlights", goal="Extraer diferencias clave del informe",
                             backstory="Especialista en comparar fichas técnicas.", allow_delegation=False, verbose=False)
            hl_task = Task(description=hl_prompt, agent=hl_agent,
                           expected_output="Reflexión breve con puntos clave y recomendación.")
            hl_crew = Crew(agents=[hl_agent], tasks=[hl_task], verbose=False)
            st.session_state["last_highlights"] = str(hl_crew.kickoff())
        else:
            st.session_state["last_highlights"] = ""

    else:
        # Modo cross (dos Excels)
        sel_sheet_a = st.session_state["sel_sheet_a"]
        sel_sheet_b = st.session_state["sel_sheet_b"]
        v1_a = st.session_state["v1_a"]
        v2_b = st.session_state["v2_b"]

        df_view_a = df_struct_a if sel_sheet_a == "(todas)" else df_struct_a[df_struct_a["Sheet"] == sel_sheet_a]
        df_view_b = df_struct_b if sel_sheet_b == "(todas)" else df_struct_b[df_struct_b["Sheet"] == sel_sheet_b]

        df_a_sel = df_view_a[df_view_a["Variant"] == v1_a].copy()
        df_b_sel = df_view_b[df_view_b["Variant"] == v2_b].copy()

        # Renombrar variantes para el informe (claridad de origen)
        v1_label = f"{v1_a} [A]"
        v2_label = f"{v2_b} [B]"
        df_a_sel["Variant"] = v1_label
        df_b_sel["Variant"] = v2_label

        combined = pd.concat([df_a_sel, df_b_sel], ignore_index=True)
        diffs = compare_variants_by_section(combined, v1_label, v2_label, sections=sel_sections)
        sheet_label = f"A:{sel_sheet_a} / B:{sel_sheet_b}" if (sel_sheet_a != "(todas)" or sel_sheet_b != "(todas)") else None
        report_md = render_differences_md(diffs, v1_label, v2_label, sheet_label)
        st.session_state["last_diffs"] = diffs
        st.session_state["last_report_md"] = report_md
        st.session_state["v1_label_display"] = v1_label
        st.session_state["v2_label_display"] = v2_label

        # CSV filtrado de contexto para el chat (solo filas con valor)
        combined_ctx = combined[~combined["Value"].apply(is_empty_value)]
        st.session_state["filtered_csv_text"] = combined_ctx.to_csv(index=False) if not combined_ctx.empty else ""

        # Highlights
        if diffs:
            hl_prompt = f"""
Eres un experto en automoción. Lee este informe de diferencias y genera un resumen breve y directo,
resaltando los puntos más importantes que un usuario debería conocer.

INFORME DE DIFERENCIAS:
{report_md}

Instrucciones:
- Responde en formato conciso pero redactado.
- Usa bullet points para destacar lo más diferenciador.
- Devuelve: resumen, diferencias clave y recomendación final.
"""
            hl_agent = Agent(role="Analista de highlights", goal="Extraer diferencias clave del informe",
                             backstory="Especialista en comparar fichas técnicas.", allow_delegation=False, verbose=False)
            hl_task = Task(description=hl_prompt, agent=hl_agent,
                           expected_output="Reflexión breve con puntos clave y recomendación.")
            hl_crew = Crew(agents=[hl_agent], tasks=[hl_task], verbose=False)
            st.session_state["last_highlights"] = str(hl_crew.kickoff())
        else:
            st.session_state["last_highlights"] = ""

# ===== Toggle de visibilidad del chat (tipo "drawer") =====
st.markdown('<div class="header-row">', unsafe_allow_html=True)
chat_toggle = st.toggle("Mostrar AI Agent", value=st.session_state["chat_open"], help="Oculta o muestra el panel de agente IA")
st.markdown('</div>', unsafe_allow_html=True)
st.session_state["chat_open"] = chat_toggle

# ===== Layout principal: centro (resultados) + derecha (chat opcional) =====
if st.session_state["chat_open"]:
    main_col, right_col = st.columns([0.90, 0.30], gap="large")
else:
    main_col = st.container()
    right_col = None

with main_col:
    # Cabecera
    if st.session_state["mode_compare"] == "same":
        export_name = build_export_filename(df_struct_main)
        st.markdown(f"<b>Archivo generado:</b> {export_name} &nbsp; <span class='pill pill-navy'>{len(df_struct_main)} filas</span>", unsafe_allow_html=True)
        sel_sheet = st.session_state["sel_sheet_main"]
        df_view = df_struct_main if sel_sheet == "(todas)" else df_struct_main[df_struct_main["Sheet"] == sel_sheet]
        v1 = st.session_state["v1_main"]
        v2 = st.session_state["v2_main"]
    else:
        st.markdown(f"<b>Excel A:</b> {export_name_a} &nbsp; <span class='pill pill-navy'>{len(df_struct_a)} filas</span>", unsafe_allow_html=True)
        st.markdown(f"<b>Excel B:</b> {export_name_b} &nbsp; <span class='pill pill-navy'>{len(df_struct_b)} filas</span>", unsafe_allow_html=True)
        df_view_a = df_struct_a if st.session_state["sel_sheet_a"] == "(todas)" else df_struct_a[df_struct_a["Sheet"] == st.session_state["sel_sheet_a"]]
        df_view_b = df_struct_b if st.session_state["sel_sheet_b"] == "(todas)" else df_struct_b[df_struct_b["Sheet"] == st.session_state["sel_sheet_b"]]
        v1 = st.session_state["v1_a"]
        v2 = st.session_state["v2_b"]

    # Tarjetas variantes
    c1, c2 = st.columns(2)
    if st.session_state["mode_compare"] == "same":
        for i, v in enumerate([v1, v2]):
            v_df = df_view[df_view["Variant"] == v]
            market = v_df["market"].dropna().iloc[0] if not v_df.empty else "—"
            ingested = v_df["ingested_at"].dropna().iloc[0] if not v_df.empty else "—"
            brand = v_df["variant_brand"].dropna().iloc[0] if not v_df.empty else ""
            model = v_df["variant_model"].dropna().iloc[0] if not v_df.empty else ""
            with (c1 if i==0 else c2):
                st.markdown(f"<b>{'A' if i==0 else 'B'} · {v}</b>", unsafe_allow_html=True)
                st.caption(f"{brand or ''} {model or ''}".strip())
                st.markdown(f"<span class='pill pill-green'>{market}</span> <span class='pill'>{ingested}</span>", unsafe_allow_html=True)
    else:
        for i, (v, dfv) in enumerate([(v1, df_view_a), (v2, df_view_b)]):
            v_df = dfv[dfv["Variant"] == v]
            market = v_df["market"].dropna().iloc[0] if not v_df.empty else "—"
            ingested = v_df["ingested_at"].dropna().iloc[0] if not v_df.empty else "—"
            brand = v_df["variant_brand"].dropna().iloc[0] if not v_df.empty else ""
            model = v_df["variant_model"].dropna().iloc[0] if not v_df.empty else ""
            with (c1 if i==0 else c2):
                st.markdown(f"<b>{'A' if i==0 else 'B'} · {v}</b>", unsafe_allow_html=True)
                st.caption(f"{brand or ''} {model or ''}".strip())
                st.markdown(f"<span class='pill pill-green'>{market}</span> <span class='pill'>{ingested}</span>", unsafe_allow_html=True)

    # Datos filtrados para pestaña 3
    if st.session_state["mode_compare"] == "same":
        df_filtered = df_view[(df_view["Variant"] == v1) | (df_view["Variant"] == v2)].copy()
    else:
        df_filtered = pd.concat([
            df_view_a[df_view_a["Variant"] == v1],
            df_view_b[df_view_b["Variant"] == v2]
        ], ignore_index=True)
    df_filtered = df_filtered[~df_filtered["Value"].apply(is_empty_value)]

    tab1, tab2, tab3 = st.tabs(["Informe", "Diferencias (CSV)", "Datos A/B"])

    with tab1:
        if st.session_state.get("last_report_md"):
            st.markdown(st.session_state["last_report_md"])
            if st.session_state.get("last_highlights"):
                st.subheader("Highlights")
                st.write(st.session_state["last_highlights"])
        else:
            st.info("Pulsa “Comparar y redactar informe” en el panel izquierdo (Configuración).")

    with tab2:
        diffs = st.session_state.get("last_diffs", None)
        if diffs:
            df_diffs = diffs_to_df(diffs, st.session_state.get("v1_label_display","A"), st.session_state.get("v2_label_display","B"))
            st.dataframe(df_diffs, use_container_width=True)
        else:
            st.info("Genera el informe para ver las diferencias.")

    with tab3:
        if st.session_state.get("filtered_csv_text", ""):
            df_show = pd.read_csv(StringIO(st.session_state["filtered_csv_text"]))  # usa io.StringIO
            st.dataframe(df_show, use_container_width=True)
        else:
            st.dataframe(df_filtered, use_container_width=True)

# ===== Chat sencillo en panel derecho con scroll propio =====
if st.session_state["chat_open"]:
    right_col = right_col if 'right_col' in locals() else None
else:
    right_col = None

if right_col is not None:
    with right_col:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chat-head">Chatbot</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-body">', unsafe_allow_html=True)

        # Historial ya existente
        for role, content in st.session_state.chat_history_agent:
            cls = "msg user" if role == "user" else "msg"
            st.markdown(f"<div class='{cls}'>{content}</div>", unsafe_allow_html=True)

        # Placeholder para el streaming del siguiente mensaje del asistente
        stream_pl = st.empty()

        st.markdown('</div>', unsafe_allow_html=True)  # end chat-body
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            user_msg = st.text_area("AI Agent", height=120, placeholder="Ej.: ¿Quiero saber los valores de ajuste que hacen subir o bajar el precio del coche B y cual es el valor de la suma total?")
            submitted_chat = st.form_submit_button("Enviar")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Procesa envío del chat
        if submitted_chat and user_msg and user_msg.strip():
            st.session_state.chat_history_agent.append(("user", user_msg.strip()))

            # 1) Construye la fuente para el contexto (soporta 1 o 2 excels)
            df_source = _get_df_for_chat()

            # 2) Filtros de secciones si corresponde
            sel_sections = st.session_state.get("sel_sections", None)
            if sel_sections:
                df_source = df_source[df_source["Section"].isin(sel_sections)]

            # 3) Variantes seleccionadas según modo
            if st.session_state.get("mode_compare", "same") == "same":
                v1 = st.session_state.get("v1_main")
                v2 = st.session_state.get("v2_main")
            else:
                v1 = st.session_state.get("v1_a")
                v2 = st.session_state.get("v2_b")

            # 4) Prepara CSV filtrado para el prompt (si falta en sesión lo generamos aquí)
            filtered_csv_text = st.session_state.get("filtered_csv_text", "")
            if not filtered_csv_text and not df_source.empty:
                sub = df_source[df_source["Variant"].isin([v1, v2])].copy()
                sub = sub[~sub["Value"].apply(is_empty_value)]
                filtered_csv_text = sub.to_csv(index=False)

            # 5) Build RAG y responde
            report_md = st.session_state.get("last_report_md", "")

            if df_source.empty or not filtered_csv_text or v1 is None or v2 is None:
                answer = "No hay datos relevantes para responder."
                stream_pl.markdown(f"<div class='msg'>{answer}</div>", unsafe_allow_html=True)
            else:
                df_rag = build_chat_context(
                    df_source,
                    v1,
                    v2,
                    sel_sections,
                    user_msg,
                    top_k=160
                )
                if df_rag.empty:
                    answer = "No hay datos suficientes en el contexto actual (revisa hoja/secciones/variantes)."
                    stream_pl.markdown(f"<div class='msg'>{answer}</div>", unsafe_allow_html=True)
                else:
                    ctx_text = render_context_as_text(df_rag)

                    # Streaming OpenAI
                    try:
                        answer = stream_openai_answer(
                            context_text=ctx_text,
                            user_q=user_msg.strip(),
                            filtered_csv_text=filtered_csv_text,
                            report_md=report_md,
                            stream_pl=stream_pl,
                            model="gpt-4o",
                            temperature=0.2
                        )
                    except Exception as e:
                        # Fallback con CrewAI si algo falla
                        answer = llm_chat_answer(ctx_text, user_msg, filtered_csv_text, report_md)
                        stream_pl.markdown(f"<div class='msg'>{answer}</div>", unsafe_allow_html=True)

            # 6) Guarda respuesta final en el historial y rerun
            st.session_state.chat_history_agent.append(("assistant", answer))
            st.experimental_rerun()
