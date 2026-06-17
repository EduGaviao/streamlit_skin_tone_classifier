import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===============================================================
#                     CONFIG
# ===============================================================

st.set_page_config(
    page_title="ISIC — Avaliação de Métodos",
    layout="wide",
    page_icon="🔬",
)

CSV_PATH = "normal_skin_dataset_processed.csv"

FITZ_ORDER = [f"FITZPATRICK {i}" for i in range(1, 7)]
MONK_ORDER = [f"MONK {i}"        for i in range(1, 11)]

FITZ_PALETTE = ["#FBE5D6", "#F7C39F", "#F4B183", "#C9925B", "#BC7D3E", "#5C2E00"]
MONK_PALETTE = ["#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
                "#a07e56", "#825c43", "#604134", "#3a312a", "#292420"]

FITZ_COLOR_MAP = {label: color for label, color in zip(FITZ_ORDER, FITZ_PALETTE)}
MONK_COLOR_MAP = {label: color for label, color in zip(MONK_ORDER, MONK_PALETTE)}

EMB_PATH = "embeddings_real_blender/WinKawaks_vit-small-patch16-224/embeddings.npz"

VIT_MSKCC_CSV  = "vit_predictions_mskcc.csv"
VIT_ESFERA_CSV = "vit_predictions_esferas.csv"

# ===============================================================
#                     HELPERS — MAPEAMENTO ITA
# ===============================================================

def ita_to_fitz(ita):
    if   ita >  55: return "FITZPATRICK 1"
    elif ita >  41: return "FITZPATRICK 2"
    elif ita >  28: return "FITZPATRICK 3"
    elif ita >  10: return "FITZPATRICK 4"
    elif ita > -30: return "FITZPATRICK 5"
    else:           return "FITZPATRICK 6"

_MONK_RANGES = [
    ("MONK 1",   81.62,  100),
    ("MONK 2",   75.99,  81.62),
    ("MONK 3",   68.24,  75.99),
    ("MONK 4",   57.53,  68.24),
    ("MONK 5",   30.61,  57.53),
    ("MONK 6",   -4.63,  30.61),
    ("MONK 7",  -37.77,  -4.63),
    ("MONK 8",  -66.87, -37.77),
    ("MONK 9",  -81.33, -66.87),
    ("MONK 10", -100,   -81.33),
]

def ita_to_monk(ita):
    for label, lo, hi in _MONK_RANGES:
        if lo <= ita <= hi:
            return label
    return None

# ===============================================================
#                     MÉTRICAS ORDINAIS
# ===============================================================

def ordinal_mae(y_true, y_pred, order):
    ranks = {label: i for i, label in enumerate(order)}
    diffs = [abs(ranks[t] - ranks[p])
             for t, p in zip(y_true, y_pred)
             if t in ranks and p in ranks]
    return round(float(np.mean(diffs)), 3) if diffs else float("nan")

def ordinal_acc1(y_true, y_pred, order):
    ranks = {label: i for i, label in enumerate(order)}
    hits = [abs(ranks[t] - ranks[p]) <= 1
            for t, p in zip(y_true, y_pred)
            if t in ranks and p in ranks]
    return round(float(np.mean(hits)), 4) if hits else float("nan")

# ===============================================================
#                     DATA
# ===============================================================

_MST_BASE = os.path.dirname(os.path.abspath(__file__))

_PAPER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_blue_orange", ["#169BFF", "#F2F2F2", "#FF9800"]
)

_COMP_METHODS = {
    "ITA Raw":    "ita_raw_monk",
    "ITA Smooth": "ita_smooth_monk",
    "ITA MSKCC":  "ita_mskcc_monk",
    "CASCo":      "stone_monk",
    "AIDA":       "aida_monk",
}


@st.cache_data
def load_mst_data():
    with open(os.path.join(_MST_BASE, "MST_resultados_esferas_100_enriched.json"), encoding="utf-8") as f:
        df_sunny = pd.DataFrame(json.load(f))
    with open(os.path.join(_MST_BASE, "MST_resultados_final_enriched.json"), encoding="utf-8") as f:
        df_full = pd.DataFrame(json.load(f))
    df_full["mask"] = pd.to_numeric(df_full["mask"], errors="coerce")
    df_full = df_full.dropna(subset=["MST"])
    filtro_ouro = (
        df_full["lighting"].isin({"well_lit"})
        & df_full["pose"].isin({"facing_camera", "frontal"})
        & df_full["mask"].isin({0})
    )
    df_gold = df_full[filtro_ouro].copy()
    return df_sunny, df_gold, df_full


@st.cache_data
def load_vit_data():
    m = pd.read_csv(VIT_MSKCC_CSV)  if os.path.exists(VIT_MSKCC_CSV)  else None
    e = pd.read_csv(VIT_ESFERA_CSV) if os.path.exists(VIT_ESFERA_CSV) else None
    return m, e


@st.cache_data
def load_embeddings(npz_path: str) -> pd.DataFrame:
    d = np.load(npz_path, allow_pickle=True)
    emb_cols = [f"e{i}" for i in range(d["embeddings"].shape[1])]
    emb_df = pd.DataFrame(d["embeddings"], columns=emb_cols)
    emb_df["image_path"] = [str(p) for p in d["image_paths"]]
    return emb_df


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["fst_label"]    = df["fst"].apply(lambda x: f"FITZPATRICK {int(x)}" if pd.notna(x) else np.nan)
    df["mst_r1_label"] = df["mst_r1"].apply(lambda x: f"MONK {int(x)}" if pd.notna(x) else np.nan)
    df["mst_r2_label"] = df["mst_r2"].apply(lambda x: f"MONK {int(x)}" if pd.notna(x) else np.nan)
    # Colorímetro MSKCC mapeado para as escalas
    df["avg_ita_mskcc_fitz"] = df["average_ita_mskcc"].apply(
        lambda x: ita_to_fitz(x) if pd.notna(x) else np.nan
    )
    df["avg_ita_mskcc_monk"] = df["average_ita_mskcc"].apply(
        lambda x: ita_to_monk(x) if pd.notna(x) else np.nan
    )
    # aliases para comparação cross-dataset
    if "ita_mskcc_repro_monk" in df.columns:
        df["ita_mskcc_monk"] = df["ita_mskcc_repro_monk"]
    if "ita_mskcc_repro_fitz" in df.columns:
        df["ita_mskcc_fitz"] = df["ita_mskcc_repro_fitz"]
    return df

df = load_data()

# ===============================================================
#                     HEADER
# ===============================================================

st.title("🔬 MSKCC — Avaliação de Métodos de Classificação de Tom de Pele")
st.caption(
    f"Dataset: `{CSV_PATH}` · "
    f"{df['patient_id'].nunique() if 'patient_id' in df.columns else '?'} pacientes · "
    f"{df['tag_id'].nunique()} sites · "
    f"{len(df)} fotografias"
)

tab_dist, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Distribuição",
    "📊 ITA: Computado vs MSKCC",
    "🎯 Precisão por Método",
    "🧬 PCA Lab",
    "🗺️ Análise por Local do Corpo",
    "🤖 PCA Embeddings (ViT-Small)",
    "📋 Comparação entre Bases",
    "🧠 ViT Fine-tuned",
])

# ===============================================================
#               TAB DIST — DISTRIBUIÇÃO DA BASE
# ===============================================================

with tab_dist:
    st.header("Distribuição da Base MSKCC — Pele Normal")
    _n_pac  = df["patient_id"].nunique() if "patient_id" in df.columns else "?"
    _n_site = df["tag_id"].nunique()
    _n_foto = len(df)
    st.caption(
        f"Pele normal (sem lesão) · "
        f"**{_n_pac}** pacientes · **{_n_site}** sites anatômicos · **{_n_foto}** fotografias"
    )

    # ── helpers ────────────────────────────────────────────────
    def _bar_monk(data_col, y_label, title):
        counts = (
            data_col.value_counts()
            .reindex(MONK_ORDER, fill_value=0)
            .reset_index()
        )
        counts.columns = ["Tom", y_label]
        fig = px.bar(
            counts, x="Tom", y=y_label,
            color="Tom",
            color_discrete_map=MONK_COLOR_MAP,
            category_orders={"Tom": MONK_ORDER},
            text=y_label, height=340, title=title,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=45, b=10))
        return fig

    def _bar_fitz(data_col, y_label, title):
        counts = (
            data_col.value_counts()
            .reindex(FITZ_ORDER, fill_value=0)
            .reset_index()
        )
        counts.columns = ["Tom", y_label]
        fig = px.bar(
            counts, x="Tom", y=y_label,
            color="Tom",
            color_discrete_map=FITZ_COLOR_MAP,
            category_orders={"Tom": FITZ_ORDER},
            text=y_label, height=340, title=title,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=45, b=10))
        return fig

    # ── MONK ──────────────────────────────────────────────────
    st.subheader("Escala Monk (MST)")
    scale_dist = st.radio(
        "Anotador", ["MST R1", "MST R2"], horizontal=True, key="scale_dist"
    )
    gt_dist = "mst_r1_label" if scale_dist == "MST R1" else "mst_r2_label"

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        if "patient_id" in df.columns:
            pat_tone = (
                df[["patient_id", gt_dist]].dropna()
                .groupby("patient_id")[gt_dist]
                .agg(lambda x: x.mode().iloc[0])
            )
            st.plotly_chart(
                _bar_monk(pat_tone, "Pacientes", "Pacientes por Tom Monk"),
                use_container_width=True,
            )
    with col_d2:
        site_tone = (
            df[["tag_id", gt_dist]].dropna()
            .groupby("tag_id")[gt_dist]
            .agg(lambda x: x.mode().iloc[0])
        )
        st.plotly_chart(
            _bar_monk(site_tone, "Sites", "Sites por Tom Monk"),
            use_container_width=True,
        )
    with col_d3:
        st.plotly_chart(
            _bar_monk(df[gt_dist].dropna(), "Fotografias", "Fotografias por Tom Monk"),
            use_container_width=True,
        )

    st.divider()

    # ── FITZPATRICK ───────────────────────────────────────────
    st.subheader("Escala Fitzpatrick (FST)")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        if "patient_id" in df.columns:
            pat_fitz = (
                df[["patient_id", "fst_label"]].dropna()
                .groupby("patient_id")["fst_label"]
                .agg(lambda x: x.mode().iloc[0])
            )
            st.plotly_chart(
                _bar_fitz(pat_fitz, "Pacientes", "Pacientes por FST"),
                use_container_width=True,
            )
    with col_f2:
        site_fitz = (
            df[["tag_id", "fst_label"]].dropna()
            .groupby("tag_id")["fst_label"]
            .agg(lambda x: x.mode().iloc[0])
        )
        st.plotly_chart(
            _bar_fitz(site_fitz, "Sites", "Sites por FST"),
            use_container_width=True,
        )
    with col_f3:
        st.plotly_chart(
            _bar_fitz(df["fst_label"].dropna(), "Fotografias", "Fotografias por FST"),
            use_container_width=True,
        )

    st.divider()

    # ── Fotografias por parte do corpo ─────────────────────────
    st.subheader("Fotografias por Parte do Corpo")
    site_counts = (
        df["anatomic_site"].dropna()
        .value_counts()
        .reset_index()
    )
    site_counts.columns = ["Parte do Corpo", "Fotografias"]
    fig_site = px.bar(
        site_counts, x="Parte do Corpo", y="Fotografias",
        text="Fotografias", color_discrete_sequence=["steelblue"], height=380,
    )
    fig_site.update_traces(textposition="outside", marker_color="steelblue")
    fig_site.update_layout(showlegend=False, margin=dict(t=30))
    st.plotly_chart(fig_site, use_container_width=True)

    st.divider()

    # ── Tabela cruzada: parte do corpo × tom ───────────────────
    st.subheader("Tabela Cruzada: Parte do Corpo × Tom")
    cross_scale = st.radio("Escala", ["Monk", "Fitzpatrick"], horizontal=True, key="cross_scale")
    if cross_scale == "Monk":
        cross_col   = gt_dist
        cross_order = MONK_ORDER
        cross_label = "Tom Monk"
    else:
        cross_col   = "fst_label"
        cross_order = FITZ_ORDER
        cross_label = "FST"

    cross_df = df[["anatomic_site", cross_col]].dropna()
    cross = (
        cross_df.groupby(["anatomic_site", cross_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[c for c in cross_order if c in cross_df[cross_col].unique()], fill_value=0)
    )
    fig_heat = px.imshow(
        cross,
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True,
        labels={"x": cross_label, "y": "Parte do Corpo", "color": "Fotografias"},
        height=max(350, len(cross) * 45),
    )
    fig_heat.update_layout(xaxis_title=cross_label, yaxis_title="Parte do Corpo")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Resumo numérico ────────────────────────────────────────
    st.subheader("Resumo Numérico")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pacientes", _n_pac)
    c2.metric("Sites anatômicos", _n_site)
    c3.metric("Fotografias", _n_foto)
    c4.metric("Fotos / site", round(_n_foto / _n_site, 1))
    c5.metric("Sites / paciente", round(_n_site / _n_pac, 1) if isinstance(_n_pac, int) else "?")

# ===============================================================
#               TAB 1 — ITA: TODOS OS MÉTODOS
# ===============================================================

with tab1:
    st.header("Comparação de Valores ITA por Método")

    ITA_METHODS = {
        "ita_raw":            "ITA Raw (computado)",
        "ita_smooth":         "ITA Smooth (computado)",
        "img_ita_mskcc":      "ITA Imagem MSKCC",
        "average_ita_mskcc":  "ITA Colorímetro MSKCC (GT)",
    }
    COLORS = {
        "ita_raw":           "#1f77b4",
        "ita_smooth":        "#ff7f0e",
        "img_ita_mskcc":     "#2ca02c",
        "average_ita_mskcc": "#d62728",
    }

    ita_cols_available = [c for c in ITA_METHODS if c in df.columns]
    sub = df[ita_cols_available].dropna()

    # ---- histograma sobreposto -----------------------------------
    st.subheader("Distribuição dos valores ITA por método")

    c1, c2 = st.columns(2)
    nbins   = c1.slider("Número de bins", 20, 100, 50, key="bins_ita")
    opacity = c2.slider("Opacidade", 0.2, 1.0, 0.55, step=0.05, key="opac_ita")

    methods_hist = st.multiselect(
        "Métodos a exibir",
        options=ita_cols_available,
        default=ita_cols_available,
        format_func=lambda x: ITA_METHODS[x],
        key="hist_methods",
    )

    fig_hist = go.Figure()
    for col in methods_hist:
        fig_hist.add_trace(go.Histogram(
            x=sub[col],
            name=ITA_METHODS[col],
            nbinsx=nbins,
            opacity=opacity,
            marker_color=COLORS[col],
        ))
    for col in methods_hist:
        mean_val = sub[col].mean()
        fig_hist.add_vline(
            x=mean_val, line_dash="dash", line_color=COLORS[col],
            annotation_text=f"μ={mean_val:.1f}°",
            annotation_position="top",
        )
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="ITA (°)",
        yaxis_title="Frequência",
        height=430,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---- tabela estatísticas -------------------------------------
    st.subheader("Estatísticas descritivas")
    stats_rows = []
    for col in ita_cols_available:
        s = sub[col]
        stats_rows.append({
            "Método": ITA_METHODS[col],
            "Média": round(s.mean(), 2),
            "Mediana": round(s.median(), 2),
            "DP": round(s.std(), 2),
            "Mín": round(s.min(), 2),
            "Máx": round(s.max(), 2),
        })
    st.dataframe(pd.DataFrame(stats_rows).set_index("Método"), use_container_width=True)

    st.divider()

    # ---- boxplot por grupo ---------------------------------------
    st.subheader("Distribuição ITA por grupo")

    group_col = st.selectbox(
        "Agrupar por",
        ["fst", "mst_r1", "mst_r2", "race", "anatomic_site", "dermoscopic_type", "dermatoscope"],
        key="group_box",
    )
    method_box = st.radio(
        "Método ITA",
        options=ita_cols_available,
        format_func=lambda x: ITA_METHODS[x],
        horizontal=True,
        key="method_box",
    )

    box_df = df[[group_col, method_box]].dropna().copy()
    box_df[group_col] = box_df[group_col].astype(str)
    group_order = sorted(
        box_df[group_col].unique(),
        key=lambda x: float(x) if x.replace(".", "").isdigit() else x,
    )

    fig_box = px.box(
        box_df, x=group_col, y=method_box,
        category_orders={group_col: group_order},
        color=group_col,
        points="outliers",
        labels={method_box: ITA_METHODS[method_box], group_col: group_col},
        title=f"{ITA_METHODS[method_box]} por {group_col}",
        height=430,
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ===============================================================
#               TAB 2 — PRECISÃO POR MÉTODO
# ===============================================================

with tab2:
    st.header("Precisão das Classificações por Método")

    scale = st.radio("Escala", ["Fitzpatrick", "Monk"], horizontal=True)

    if scale == "Fitzpatrick":
        gt_options = {"FST (ground truth)": "fst_label"}
        pred_options = {
            "ITA Raw":              "ita_raw_fitz",
            "ITA Smooth":           "ita_smooth_fitz",
            "ITA MSKCC (imagem)":   "ita_mskcc_fitz",
            "ITA Colorímetro MSKCC":"avg_ita_mskcc_fitz",
            "Stone":                "stone_fitz",
            "AIDA":                 "aida_fitz",
        }
        class_order = FITZ_ORDER
        cmap = "Blues"
    else:
        gt_options = {
            "MST R1 (ground truth)": "mst_r1_label",
            "MST R2 (ground truth)": "mst_r2_label",
        }
        pred_options = {
            "ITA Raw":              "ita_raw_monk",
            "ITA Smooth":           "ita_smooth_monk",
            "ITA MSKCC (imagem)":   "ita_mskcc_monk",
            "ITA Colorímetro MSKCC":"avg_ita_mskcc_monk",
            "Stone":                "stone_monk",
            "AIDA":                 "aida_monk",
        }
        class_order = MONK_ORDER
        cmap = "Purples"

    gt_label   = st.selectbox("Ground truth", list(gt_options.keys()))
    gt_col     = gt_options[gt_label]
    methods_sel = st.multiselect(
        "Métodos a comparar",
        list(pred_options.keys()),
        default=list(pred_options.keys()),
    )

    if not methods_sel:
        st.warning("Selecione ao menos um método.")
    else:
        # ---- tabela resumo ---------------------------------------
        rows = []
        for m in methods_sel:
            col = pred_options[m]
            sub = df[[gt_col, col]].dropna()
            active = [c for c in class_order if c in sub[gt_col].values or c in sub[col].values]
            yt = pd.Categorical(sub[gt_col], categories=active)
            yp = pd.Categorical(sub[col],    categories=active)
            rows.append({
                "Método": m,
                "n": len(sub),
                "Acurácia": round(accuracy_score(yt, yp), 4),
                "Kappa (quad)": round(cohen_kappa_score(yt, yp, weights="quadratic"), 4),
                "MAE ordinal": ordinal_mae(sub[gt_col], sub[col], class_order),
                "Acurácia ±1": ordinal_acc1(sub[gt_col], sub[col], class_order),
            })
        st.dataframe(
            pd.DataFrame(rows).set_index("Método").sort_values("Kappa (quad)", ascending=False),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Matrizes de Confusão")

        n_cols  = min(len(methods_sel), 2)
        cols_cm = st.columns(n_cols)

        for idx, m in enumerate(methods_sel):
            col = pred_options[m]
            sub = df[[gt_col, col]].dropna()
            active = [c for c in class_order if c in sub[gt_col].values or c in sub[col].values]
            short  = [a.split()[-1] for a in active]  # "1", "2", ..

            yt = pd.Categorical(sub[gt_col], categories=active)
            yp = pd.Categorical(sub[col],    categories=active)

            cm_counts = confusion_matrix(yt, yp, labels=active)
            # normaliza por linha (verdadeiro) para obter % por classe real
            row_sums  = cm_counts.sum(axis=1, keepdims=True)
            cm_pct    = np.where(row_sums > 0, cm_counts / row_sums * 100, 0)

            # anotações: "N\n(X%)"
            annots = np.array([
                [f"{cm_counts[i,j]}\n({cm_pct[i,j]:.1f}%)"
                 for j in range(len(active))]
                for i in range(len(active))
            ])

            n_classes  = len(active)
            cell_font  = max(5, 9 - n_classes)   # 6 classes→7pt, 9 classes→5pt
            fig_size   = max(5.5, 0.6 * n_classes)
            fig_cm, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
            sns.heatmap(
                cm_pct, annot=annots, fmt="", cmap=cmap,
                xticklabels=short, yticklabels=short,
                ax=ax, cbar=True, linewidths=0.4,
                vmin=0, vmax=100,
                annot_kws={"size": cell_font, "linespacing": 1.2},
            )
            ax.set_title(m, fontsize=11)
            ax.set_xlabel("Predito")
            ax.set_ylabel(gt_label.split(" (")[0])
            ax.tick_params(labelsize=8)
            plt.tight_layout()

            with cols_cm[idx % n_cols]:
                st.pyplot(fig_cm)
                st.caption(
                    f"Acurácia: **{accuracy_score(yt, yp):.4f}** · "
                    f"Kappa (quad): **{cohen_kappa_score(yt, yp, weights='quadratic'):.4f}**"
                )
            plt.close(fig_cm)

# ===============================================================
#               TAB 3 — PCA
# ===============================================================

with tab3:
    st.header("PCA das Features Lab")

    # ---- escolha do método de features --------------------------
    feat_source = st.radio(
        "Features de cor",
        ["Nosso método (stats completas)", "MSKCC (img_l, img_a, img_b)"],
        horizontal=True,
    )

    if feat_source.startswith("Nosso"):
        pca_features_all = {
            "L_mean": "L* média",  "L_std": "L* DP",  "L_max": "L* máx",  "L_min": "L* mín",
            "a_mean": "a* média",  "a_std": "a* DP",  "a_max": "a* máx",  "a_min": "a* mín",
            "b_mean": "b* média",  "b_std": "b* DP",  "b_max": "b* máx",  "b_min": "b* mín",
        }
    else:
        pca_features_all = {
            "img_l_mskcc": "L* (MSKCC img)",
            "img_a_mskcc": "a* (MSKCC img)",
            "img_b_mskcc": "b* (MSKCC img)",
        }

    c_left, c_right = st.columns([2, 1])
    with c_left:
        feats_sel = st.multiselect(
            "Features para o PCA",
            options=list(pca_features_all.keys()),
            default=list(pca_features_all.keys()),
            format_func=lambda x: pca_features_all[x],
        )
    with c_right:
        color_pca = st.selectbox(
            "Colorir por",
            ["fst_label", "mst_r1_label", "race", "anatomic_site",
             "dermoscopic_type", "ita_raw_fitz", "ita_mskcc_fitz",
             "stone_fitz", "aida_fitz",
             "ita_raw_monk", "ita_mskcc_monk", "stone_monk", "aida_monk"],
        )
        n_components = st.slider(
            "Componentes PCA",
            min_value=2,
            max_value=min(len(feats_sel) if feats_sel else 2, 5),
            value=2,
        )

    if len(feats_sel) < 2:
        st.warning("Selecione ao menos 2 features.")
    else:
        pca_df = df[feats_sel + [color_pca]].dropna().copy()

        X      = StandardScaler().fit_transform(pca_df[feats_sel])
        pca    = PCA(n_components=n_components)
        coords = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_

        pca_plot = pd.DataFrame(
            coords,
            columns=[f"PC{i+1} ({var_exp[i]*100:.1f}%)" for i in range(n_components)],
        )
        pca_plot[color_pca] = pca_df[color_pca].values

        # ---- define paleta e ordenação ---------------------------
        is_fitz = color_pca in ("fst_label", "ita_raw_fitz", "ita_smooth_fitz",
                                 "ita_mskcc_fitz", "stone_fitz", "aida_fitz")
        is_monk = color_pca in ("mst_r1_label", "mst_r2_label", "ita_raw_monk",
                                 "ita_smooth_monk", "ita_mskcc_monk", "stone_monk", "aida_monk")

        if is_fitz:
            present = [c for c in FITZ_ORDER if c in pca_plot[color_pca].values]
            cat_order      = {color_pca: present}
            color_disc_map = {c: FITZ_COLOR_MAP[c] for c in present}
        elif is_monk:
            present = [c for c in MONK_ORDER if c in pca_plot[color_pca].values]
            cat_order      = {color_pca: present}
            color_disc_map = {c: MONK_COLOR_MAP[c] for c in present}
        else:
            vals = sorted(pca_plot[color_pca].dropna().unique(),
                          key=lambda x: float(x) if str(x).replace(".", "").isdigit() else str(x))
            cat_order      = {color_pca: [str(v) for v in vals]}
            pca_plot[color_pca] = pca_plot[color_pca].astype(str)
            color_disc_map = None

        st.divider()

        # ---- variância explicada (sem acumulada) -----------------
        col_var, col_load = st.columns(2)

        with col_var:
            st.subheader("Variância Explicada")
            fig_var = go.Figure()
            fig_var.add_bar(
                x=[f"PC{i+1}" for i in range(n_components)],
                y=var_exp * 100,
                marker_color="steelblue",
            )
            fig_var.update_layout(
                yaxis_title="%",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_var, use_container_width=True)

        with col_load:
            st.subheader("Loadings (PC1 e PC2)")
            load_df = pd.DataFrame(
                pca.components_[:2].T,
                index=[pca_features_all[f] for f in feats_sel],
                columns=["PC1", "PC2"],
            ).round(3)
            st.dataframe(
                load_df.style.background_gradient(cmap="RdBu", axis=None),
                height=280,
            )

        st.divider()

        # ---- scatter 2D ------------------------------------------
        st.subheader("PCA 2D")
        pc_x = pca_plot.columns[0]
        pc_y = pca_plot.columns[1]

        scatter_kwargs = dict(
            data_frame=pca_plot, x=pc_x, y=pc_y,
            color=color_pca,
            opacity=0.65,
            title=f"PCA — colorido por {color_pca}",
            category_orders=cat_order,
            height=550,
        )
        if color_disc_map:
            scatter_kwargs["color_discrete_map"] = color_disc_map

        fig_pca = px.scatter(**scatter_kwargs)
        fig_pca.update_traces(marker_size=5)
        st.plotly_chart(fig_pca, use_container_width=True)

        # ---- PC1 vs PC3 se houver --------------------------------
        if n_components >= 3:
            st.subheader("PCA: PC1 vs PC3")
            pc_z = pca_plot.columns[2]
            scatter_kwargs2 = {**scatter_kwargs,
                               "y": pc_z, "title": f"PC1 vs PC3 — colorido por {color_pca}",
                               "height": 450}
            fig_pca2 = px.scatter(**scatter_kwargs2)
            fig_pca2.update_traces(marker_size=5)
            st.plotly_chart(fig_pca2, use_container_width=True)

# ===============================================================
#               TAB 4 — ANÁLISE POR LOCAL DO CORPO
# ===============================================================

with tab4:
    st.header("Análise por Local do Corpo (anatomic_site)")

    scale_site = st.radio("Escala", ["Fitzpatrick", "Monk"], horizontal=True, key="scale_site")

    if scale_site == "Fitzpatrick":
        gt_opts_site = {"FST (ground truth)": "fst_label"}
        pred_opts_site = {
            "ITA Raw":               "ita_raw_fitz",
            "ITA Smooth":            "ita_smooth_fitz",
            "ITA MSKCC (imagem)":    "ita_mskcc_fitz",
            "ITA Colorímetro MSKCC": "avg_ita_mskcc_fitz",
            "Stone":                 "stone_fitz",
            "AIDA":                  "aida_fitz",
        }
        class_order_site = FITZ_ORDER
        cmap_site = "Blues"
    else:
        gt_opts_site = {
            "MST R1 (ground truth)": "mst_r1_label",
            "MST R2 (ground truth)": "mst_r2_label",
        }
        pred_opts_site = {
            "ITA Raw":               "ita_raw_monk",
            "ITA Smooth":            "ita_smooth_monk",
            "ITA MSKCC (imagem)":    "ita_mskcc_monk",
            "ITA Colorímetro MSKCC": "avg_ita_mskcc_monk",
            "Stone":                 "stone_monk",
            "AIDA":                  "aida_monk",
        }
        class_order_site = MONK_ORDER
        cmap_site = "Purples"

    gt_label_site = st.selectbox("Ground truth", list(gt_opts_site.keys()), key="gt_site")
    gt_col_site   = gt_opts_site[gt_label_site]

    methods_site = st.multiselect(
        "Métodos a comparar",
        list(pred_opts_site.keys()),
        default=list(pred_opts_site.keys()),
        key="methods_site",
    )

    if not methods_site:
        st.warning("Selecione ao menos um método.")
    else:
        sites = sorted(df["anatomic_site"].dropna().unique())

        # ---- matriz kappa por site × método ----------------------
        kappa_rows, n_rows = [], []
        for site in sites:
            sub_site = df[df["anatomic_site"] == site]
            row_k = {"Local": site}
            row_n = {"Local": site}
            for m in methods_site:
                col = pred_opts_site[m]
                sub_s = sub_site[[gt_col_site, col]].dropna()
                row_n[m] = len(sub_s)
                if len(sub_s) < 5:
                    row_k[m] = np.nan
                else:
                    active = [c for c in class_order_site
                              if c in sub_s[gt_col_site].values or c in sub_s[col].values]
                    if len(active) < 2:
                        row_k[m] = np.nan
                    else:
                        yt = pd.Categorical(sub_s[gt_col_site], categories=active)
                        yp = pd.Categorical(sub_s[col],         categories=active)
                        try:
                            row_k[m] = round(cohen_kappa_score(yt, yp, weights="quadratic"), 3)
                        except Exception:
                            row_k[m] = np.nan
            kappa_rows.append(row_k)
            n_rows.append(row_n)

        kappa_df_site = pd.DataFrame(kappa_rows).set_index("Local")
        n_df_site     = pd.DataFrame(n_rows).set_index("Local")

        # heatmap kappa
        st.subheader("Kappa (quadrático) por Local do Corpo")
        h = max(4, len(sites) * 0.55)
        w = max(6, len(methods_site) * 1.8)
        fig_hm, ax_hm = plt.subplots(figsize=(w, h))
        sns.heatmap(
            kappa_df_site, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-0.2, vmax=1.0, center=0.4,
            linewidths=0.4, ax=ax_hm,
            annot_kws={"size": 9},
        )
        ax_hm.set_xlabel("Método", fontsize=10)
        ax_hm.set_ylabel("Local do corpo", fontsize=10)
        ax_hm.tick_params(labelsize=8)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig_hm)
        plt.close(fig_hm)

        with st.expander("Ver contagem de amostras por local"):
            st.dataframe(n_df_site, use_container_width=True)

        st.divider()

        # ---- matrizes de confusão por site selecionado -----------
        st.subheader("Matrizes de Confusão por Local do Corpo")
        site_sel = st.selectbox("Selecione o local do corpo", sites, key="site_cm")
        sub_site_sel = df[df["anatomic_site"] == site_sel]

        n_cols_cm = min(len(methods_site), 2)
        cols_cm_site = st.columns(n_cols_cm)

        for idx, m in enumerate(methods_site):
            col = pred_opts_site[m]
            sub_s = sub_site_sel[[gt_col_site, col]].dropna()

            with cols_cm_site[idx % n_cols_cm]:
                if len(sub_s) < 3:
                    st.warning(f"**{m}**: poucos dados (n={len(sub_s)})")
                    continue

                active = [c for c in class_order_site
                          if c in sub_s[gt_col_site].values or c in sub_s[col].values]
                if len(active) < 2:
                    st.warning(f"**{m}**: classes insuficientes")
                    continue

                short = [a.split()[-1] for a in active]
                yt = pd.Categorical(sub_s[gt_col_site], categories=active)
                yp = pd.Categorical(sub_s[col],         categories=active)

                cm_counts = confusion_matrix(yt, yp, labels=active)
                row_sums  = cm_counts.sum(axis=1, keepdims=True)
                cm_pct    = np.where(row_sums > 0, cm_counts / row_sums * 100, 0)
                annots    = np.array([
                    [f"{cm_counts[i,j]}\n({cm_pct[i,j]:.1f}%)"
                     for j in range(len(active))]
                    for i in range(len(active))
                ])

                n_cls     = len(active)
                cell_font = max(5, 9 - n_cls)
                fig_size  = max(5.5, 0.6 * n_cls)
                fig_cm, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
                sns.heatmap(
                    cm_pct, annot=annots, fmt="", cmap=cmap_site,
                    xticklabels=short, yticklabels=short,
                    ax=ax, cbar=True, linewidths=0.4,
                    vmin=0, vmax=100,
                    annot_kws={"size": cell_font, "linespacing": 1.2},
                )
                ax.set_title(f"{m}", fontsize=10)
                ax.set_xlabel("Predito")
                ax.set_ylabel(gt_label_site.split(" (")[0])
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_cm)
                try:
                    kappa_v = cohen_kappa_score(yt, yp, weights="quadratic")
                    acc_v   = accuracy_score(yt, yp)
                    st.caption(
                        f"n={len(sub_s)} · Acurácia: **{acc_v:.4f}** · Kappa: **{kappa_v:.4f}**"
                    )
                except Exception:
                    st.caption(f"n={len(sub_s)}")

# ===============================================================
#               TAB 5 — PCA EMBEDDINGS ViT / CLIP
# ===============================================================

_COLOR_COLS = [
    "fst_label", "mst_r1_label", "mst_r2_label",
    "race", "anatomic_site", "dermoscopic_type", "dermatoscope",
    "ita_raw_fitz", "ita_smooth_fitz", "ita_mskcc_fitz", "avg_ita_mskcc_fitz",
    "stone_fitz", "aida_fitz",
    "ita_raw_monk", "ita_smooth_monk", "ita_mskcc_monk", "avg_ita_mskcc_monk",
    "stone_monk", "aida_monk",
]

with tab5:
    st.header("PCA — Embeddings ViT-Small")
    st.caption("Modelo: `WinKawaks/vit-small-patch16-224`")

    if not os.path.exists(EMB_PATH):
        st.warning(
            "Arquivo de embeddings não encontrado. "
            "Execute `inference.py --model WinKawaks/vit-small-patch16-224` para gerar."
        )
    else:
        with st.spinner("Carregando embeddings..."):
            emb_df = load_embeddings(EMB_PATH)

        # join metadata
        meta_cols = [c for c in _COLOR_COLS if c in df.columns]
        merged = emb_df.merge(df[["image_path"] + meta_cols], on="image_path", how="left")

        emb_dim = emb_df.shape[1] - 1
        st.caption(f"{len(merged)} imagens · dimensão {emb_dim}")

        c_left, c_right = st.columns([2, 1])
        with c_left:
            color_emb = st.selectbox("Colorir por", meta_cols, key="color_emb")
        with c_right:
            n_comp_emb = st.radio("Dimensões PCA", [2, 3], horizontal=True, key="n_comp_emb")

        feat_cols = [c for c in emb_df.columns if c != "image_path"]
        pca_input = merged[feat_cols + [color_emb]].dropna(subset=[color_emb])

        if len(pca_input) < 10:
            st.warning("Poucos dados para o PCA após filtrar NaN.")
        else:
            X_emb   = StandardScaler().fit_transform(pca_input[feat_cols])
            pca_emb = PCA(n_components=n_comp_emb)
            coords  = pca_emb.fit_transform(X_emb)
            var_exp = pca_emb.explained_variance_ratio_

            col_names = [f"PC{i+1} ({var_exp[i]*100:.1f}%)" for i in range(n_comp_emb)]
            pca_plot  = pd.DataFrame(coords, columns=col_names)
            pca_plot[color_emb] = pca_input[color_emb].values

            # paleta
            _is_fitz = color_emb in ("fst_label", "ita_raw_fitz", "ita_smooth_fitz",
                                      "ita_mskcc_fitz", "avg_ita_mskcc_fitz",
                                      "stone_fitz", "aida_fitz")
            _is_monk = color_emb in ("mst_r1_label", "mst_r2_label", "ita_raw_monk",
                                      "ita_smooth_monk", "ita_mskcc_monk", "avg_ita_mskcc_monk",
                                      "stone_monk", "aida_monk")

            if _is_fitz:
                present       = [c for c in FITZ_ORDER if c in pca_plot[color_emb].values]
                cat_order_emb = {color_emb: present}
                cmap_emb      = {c: FITZ_COLOR_MAP[c] for c in present}
            elif _is_monk:
                present       = [c for c in MONK_ORDER if c in pca_plot[color_emb].values]
                cat_order_emb = {color_emb: present}
                cmap_emb      = {c: MONK_COLOR_MAP[c] for c in present}
            else:
                vals = sorted(
                    pca_plot[color_emb].dropna().unique(),
                    key=lambda x: float(x) if str(x).replace(".", "").isdigit() else str(x),
                )
                pca_plot[color_emb] = pca_plot[color_emb].astype(str)
                cat_order_emb = {color_emb: [str(v) for v in vals]}
                cmap_emb      = None

            # variância explicada
            fig_var = go.Figure()
            fig_var.add_bar(
                x=[f"PC{i+1}" for i in range(n_comp_emb)],
                y=var_exp * 100,
                marker_color="steelblue",
            )
            fig_var.update_layout(yaxis_title="%", height=250, showlegend=False,
                                  title="Variância Explicada")
            st.plotly_chart(fig_var, use_container_width=True)

            st.divider()

            # 2D
            st.subheader("PCA 2D")
            kw2d = dict(
                data_frame=pca_plot, x=col_names[0], y=col_names[1],
                color=color_emb, opacity=0.65,
                title=f"ViT-Small · PCA 2D — {color_emb}",
                category_orders=cat_order_emb,
                height=560,
            )
            if cmap_emb:
                kw2d["color_discrete_map"] = cmap_emb
            fig2d = px.scatter(**kw2d)
            fig2d.update_traces(marker_size=5)
            st.plotly_chart(fig2d, use_container_width=True)

            # 3D
            if n_comp_emb == 3:
                st.divider()
                st.subheader("PCA 3D")
                kw3d = dict(
                    data_frame=pca_plot, x=col_names[0], y=col_names[1], z=col_names[2],
                    color=color_emb, opacity=0.7,
                    title=f"ViT-Small · PCA 3D — {color_emb}",
                    category_orders=cat_order_emb,
                    height=620,
                )
                if cmap_emb:
                    kw3d["color_discrete_map"] = cmap_emb
                fig3d = px.scatter_3d(**kw3d)
                fig3d.update_traces(marker_size=3)
                st.plotly_chart(fig3d, use_container_width=True)

# ===============================================================
#               TAB 6 — COMPARAÇÃO ENTRE BASES
# ===============================================================

with tab6:
    st.header("Comparação entre Bases de Dados")

    df_sunny, df_gold, df_mste_all = load_mst_data()

    mskcc_gt_choice = st.radio(
        "GT para MSKCC", ["MST R1", "MST R2"], horizontal=True, key="mskcc_gt_comp"
    )
    gt_col_mskcc = "mst_r1_label" if mskcc_gt_choice == "MST R1" else "mst_r2_label"

    datasets_comp = {
        "SunnyDay":   (df_sunny,    "true_monk"),
        "MST-E Ouro": (df_gold,     "MST"),
        "MST-E All":  (df_mste_all, "MST"),
        "MSKCC":      (df,          gt_col_mskcc),
    }
    ds_names  = list(datasets_comp.keys())
    met_names = list(_COMP_METHODS.keys())

    # ---- tabelas de resumo (kappa e acurácia) --------------------
    kappa_rows, acc_rows, mae_rows, acc1_rows, n_rows = [], [], [], [], []
    for m_label, m_col in _COMP_METHODS.items():
        kr  = {"Método": m_label}
        ar  = {"Método": m_label}
        mr  = {"Método": m_label}
        a1r = {"Método": m_label}
        nr  = {"Método": m_label}
        for ds_name, (ds_df, gt_col) in datasets_comp.items():
            if m_col not in ds_df.columns:
                kr[ds_name] = ar[ds_name] = mr[ds_name] = a1r[ds_name] = np.nan
                nr[ds_name] = 0
                continue
            sub = ds_df[[gt_col, m_col]].dropna()
            sub = sub[sub[gt_col].isin(MONK_ORDER) & sub[m_col].isin(MONK_ORDER)]
            nr[ds_name] = len(sub)
            if len(sub) < 5:
                kr[ds_name] = ar[ds_name] = mr[ds_name] = a1r[ds_name] = np.nan
            else:
                yt = pd.Categorical(sub[gt_col], categories=MONK_ORDER)
                yp = pd.Categorical(sub[m_col],  categories=MONK_ORDER)
                try:
                    kr[ds_name]  = round(cohen_kappa_score(yt, yp, weights="quadratic"), 3)
                    ar[ds_name]  = round(accuracy_score(yt, yp), 3)
                    mr[ds_name]  = ordinal_mae(sub[gt_col], sub[m_col], MONK_ORDER)
                    a1r[ds_name] = ordinal_acc1(sub[gt_col], sub[m_col], MONK_ORDER)
                except Exception:
                    kr[ds_name] = ar[ds_name] = mr[ds_name] = a1r[ds_name] = np.nan
        kappa_rows.append(kr)
        acc_rows.append(ar)
        mae_rows.append(mr)
        acc1_rows.append(a1r)
        n_rows.append(nr)

    kappa_tbl = pd.DataFrame(kappa_rows).set_index("Método")
    acc_tbl   = pd.DataFrame(acc_rows).set_index("Método")
    mae_tbl   = pd.DataFrame(mae_rows).set_index("Método")
    acc1_tbl  = pd.DataFrame(acc1_rows).set_index("Método")
    n_tbl     = pd.DataFrame(n_rows).set_index("Método")

    col_k, col_a = st.columns(2)
    with col_k:
        st.subheader("Kappa Quadrático")
        st.dataframe(
            kappa_tbl.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None)
                           .format("{:.3f}"),
            use_container_width=True,
        )
    with col_a:
        st.subheader("Acurácia")
        st.dataframe(
            acc_tbl.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None)
                         .format("{:.3f}"),
            use_container_width=True,
        )

    col_m, col_a1 = st.columns(2)
    with col_m:
        st.subheader("MAE Ordinal ↓")
        st.caption("Menor = melhor. Erro médio em posições na escala Monk.")
        st.dataframe(
            mae_tbl.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=5, axis=None)
                         .format("{:.3f}"),
            use_container_width=True,
        )
    with col_a1:
        st.subheader("Acurácia ±1 ↑")
        st.caption("Proporção de predições a no máximo 1 classe de distância do GT.")
        st.dataframe(
            acc1_tbl.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None)
                          .format("{:.3f}"),
            use_container_width=True,
        )

    with st.expander("Ver n de amostras por base e método"):
        st.dataframe(n_tbl, use_container_width=True)

    st.divider()

    # ---- grade de matrizes de confusão: métodos × bases ----------
    st.subheader("Matrizes de Confusão — Monk Scale")
    st.caption("Linhas = métodos · Colunas = bases · Normalizado por linha real · κ no canto inferior direito")

    n_cls  = len(MONK_ORDER)
    cell_w = 3.2
    fig_grid, axes_grid = plt.subplots(
        len(met_names), len(ds_names),
        figsize=(cell_w * len(ds_names), cell_w * len(met_names)),
        squeeze=False,
    )

    for r, (m_label, m_col) in enumerate(_COMP_METHODS.items()):
        for c, (ds_name, (ds_df, gt_col)) in enumerate(datasets_comp.items()):
            ax = axes_grid[r][c]
            sub = ds_df[[gt_col, m_col]].dropna()
            sub = sub[sub[gt_col].isin(MONK_ORDER) & sub[m_col].isin(MONK_ORDER)]

            if len(sub) < 3:
                ax.set_visible(False)
                continue

            yt = pd.Categorical(sub[gt_col], categories=MONK_ORDER)
            yp = pd.Categorical(sub[m_col],  categories=MONK_ORDER)
            cm_norm = confusion_matrix(yt, yp, labels=MONK_ORDER, normalize="true")

            ax.imshow(cm_norm, cmap=_PAPER_CMAP, vmin=0, vmax=1, aspect="auto")

            short = [str(i + 1) for i in range(n_cls)]
            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))
            ax.set_xticklabels(short if r == len(met_names) - 1 else [], fontsize=5)
            ax.set_yticklabels(short if c == 0 else [], fontsize=5)

            # borda diagonal
            for i in range(n_cls):
                ax.add_patch(plt.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=1.5
                ))

            # cabeçalhos
            if r == 0:
                ax.set_title(ds_name, fontsize=9, pad=4, fontweight="bold")
            if c == 0:
                ax.set_ylabel(m_label, fontsize=9, labelpad=4, fontweight="bold")
            if r == len(met_names) - 1:
                ax.set_xlabel("Pred", fontsize=7)

            # kappa no canto
            try:
                kv = cohen_kappa_score(yt, yp, weights="quadratic")
                ax.text(
                    0.98, 0.02, f"κ={kv:.2f}",
                    transform=ax.transAxes, fontsize=6,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
                )
            except Exception:
                pass

    # colorbar global
    fig_grid.subplots_adjust(right=0.88, wspace=0.05, hspace=0.1)
    cbar_ax = fig_grid.add_axes([0.90, 0.15, 0.012, 0.70])
    sm = plt.cm.ScalarMappable(cmap=_PAPER_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig_grid.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Proporção", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    st.pyplot(fig_grid)
    plt.close(fig_grid)

# ===============================================================
#               TAB COLOR GT — GT COLORIMÉTRICO
# ===============================================================

# ===============================================================
#               TAB 7 — ViT FINE-TUNED
# ===============================================================

# config por GT: (coluna pred, coluna split, coluna true no df principal, ordem)
_VIT_GTS = {
    "ViT MST R1": ("vit_mst_r1_pred", "split_mst_r1", "mst_r1_label", MONK_ORDER, "Purples"),
    "ViT MST R2": ("vit_mst_r2_pred", "split_mst_r2", "mst_r2_label", MONK_ORDER, "Purples"),
    "ViT FST":    ("vit_fst_pred",    "split_fst",    "fst_label",    FITZ_ORDER,  "Blues"),
}


def _cm_plot(ax, sub, true_col, pred_col, order, cmap, title):
    active = [c for c in order if c in sub[true_col].values or c in sub[pred_col].values]
    short  = [c.split()[-1] for c in active]
    yt = pd.Categorical(sub[true_col], categories=active)
    yp = pd.Categorical(sub[pred_col], categories=active)
    cm_c = confusion_matrix(yt, yp, labels=active)
    rs   = cm_c.sum(axis=1, keepdims=True)
    cm_p = np.where(rs > 0, cm_c / rs * 100, 0)
    ann  = np.array([[f"{cm_c[i,j]}\n({cm_p[i,j]:.0f}%)"
                      for j in range(len(active))] for i in range(len(active))])
    n_cls = len(active)
    sns.heatmap(cm_p, annot=ann, fmt="", cmap=cmap,
                xticklabels=short, yticklabels=short,
                ax=ax, vmin=0, vmax=100, linewidths=0.4,
                annot_kws={"size": max(5, 9 - n_cls), "linespacing": 1.2})
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    try:
        kv = cohen_kappa_score(yt, yp, weights="quadratic")
        av = accuracy_score(yt, yp)
        return len(sub), kv, av
    except Exception:
        return len(sub), float("nan"), float("nan")


with tab7:
    st.header("ViT Fine-tuned — Classificação de Tom de Pele")
    st.caption(
        "Backbone: ViT-Small · Loss: Ordinal Cross-Entropy · "
        "Split: IND por `tag_id` (70/15/15) · Máx. 2 imgs/indivíduo no treino"
    )

    vit_mskcc_raw, vit_esfera = load_vit_data()

    if vit_mskcc_raw is None:
        st.warning("Execute `finetune_vit.py` para gerar as predições.")
        st.stop()

    # join com df principal para ter os rótulos verdadeiros em string
    vit_df = df.merge(vit_mskcc_raw, on="isic_id", how="inner")

    # ── Seção 1: Métricas resumo ──────────────────────────────
    st.subheader("Métricas por Ground Truth")

    col_mskcc, col_esf = st.columns(2)

    # MSKCC test
    with col_mskcc:
        st.markdown("**MSKCC (test set, split IND)**")
        rows_m = []
        for gt_name, (pred_col, split_col, true_col, order, _) in _VIT_GTS.items():
            sub = vit_df[vit_df[split_col] == "test"][[true_col, pred_col]].dropna()
            sub = sub[sub[true_col].isin(order) & sub[pred_col].isin(order)]
            if len(sub) < 3:
                continue
            yt = pd.Categorical(sub[true_col], categories=order)
            yp = pd.Categorical(sub[pred_col], categories=order)
            try:
                rows_m.append({
                    "GT": gt_name, "n": len(sub),
                    "κ (quad)":    round(cohen_kappa_score(yt, yp, weights="quadratic"), 3),
                    "Acurácia":    round(accuracy_score(yt, yp), 3),
                    "MAE":         ordinal_mae(sub[true_col], sub[pred_col], order),
                    "Acc ±1":      ordinal_acc1(sub[true_col], sub[pred_col], order),
                })
            except Exception:
                pass
        if rows_m:
            tbl = pd.DataFrame(rows_m).set_index("GT")
            st.dataframe(
                tbl.style
                   .background_gradient(cmap="RdYlGn", subset=["κ (quad)", "Acurácia", "Acc ±1"],
                                        vmin=0, vmax=1, axis=None)
                   .background_gradient(cmap="RdYlGn_r", subset=["MAE"], vmin=0, vmax=5, axis=None)
                   .format("{:.3f}", subset=["κ (quad)", "Acurácia", "MAE", "Acc ±1"]),
                use_container_width=True,
            )

    # Esferas — modelo treinado nas próprias esferas
    with col_esf:
        st.markdown("**Esferas sintéticas (modelo treinado nas esferas)**")
        if vit_esfera is not None and "vit_esfera_pred" in vit_esfera.columns:
            st.caption("GT = `true_monk` · Split 70/15/15 estratificado por classe")
            rows_e = []
            for split_name in ["train", "val", "test"]:
                esf_sub = vit_esfera[vit_esfera["split"] == split_name]
                sub = esf_sub[["true_monk", "vit_esfera_pred"]].dropna()
                sub = sub[sub["true_monk"].isin(MONK_ORDER) & sub["vit_esfera_pred"].isin(MONK_ORDER)]
                n_split = len(esf_sub)
                if len(sub) < 3:
                    rows_e.append({"Split": split_name, "n total": n_split, "n pred": 0,
                                   "κ (quad)": None, "Acurácia": None, "MAE": None, "Acc ±1": None})
                    continue
                yt = pd.Categorical(sub["true_monk"],       categories=MONK_ORDER)
                yp = pd.Categorical(sub["vit_esfera_pred"], categories=MONK_ORDER)
                try:
                    rows_e.append({
                        "Split":    split_name,
                        "n total":  n_split,
                        "n pred":   len(sub),
                        "κ (quad)": round(cohen_kappa_score(yt, yp, weights="quadratic"), 3),
                        "Acurácia": round(accuracy_score(yt, yp), 3),
                        "MAE":      ordinal_mae(sub["true_monk"], sub["vit_esfera_pred"], MONK_ORDER),
                        "Acc ±1":   ordinal_acc1(sub["true_monk"], sub["vit_esfera_pred"], MONK_ORDER),
                    })
                except Exception:
                    pass
            if rows_e:
                tbl_e = pd.DataFrame(rows_e).set_index("Split")
                metric_cols = [c for c in ["κ (quad)", "Acurácia", "MAE", "Acc ±1"] if tbl_e[c].notna().any()]
                st.dataframe(
                    tbl_e.style
                         .background_gradient(cmap="RdYlGn",
                                              subset=[c for c in ["κ (quad)", "Acurácia", "Acc ±1"] if c in metric_cols],
                                              vmin=0, vmax=1, axis=None)
                         .background_gradient(cmap="RdYlGn_r",
                                              subset=[c for c in ["MAE"] if c in metric_cols],
                                              vmin=0, vmax=5, axis=None)
                         .format("{:.3f}", subset=metric_cols, na_rep="—"),
                    use_container_width=True,
                )
        else:
            st.info("Rode `finetune_vit.py --dataset esferas` para gerar predições.")

    st.divider()

    # ── Seção 2: Matrizes de confusão ─────────────────────────
    st.subheader("Matrizes de Confusão")
    cm_src = st.radio("Dataset", ["MSKCC (test)", "Esferas (test)"],
                      horizontal=True, key="vit_cm_src")

    if cm_src == "MSKCC (test)":
        cols_cm = st.columns(3)
        for i, (gt_name, (pred_col, split_col, true_col, order, cmap)) in enumerate(_VIT_GTS.items()):
            sub = vit_df[vit_df[split_col] == "test"][[true_col, pred_col]].dropna()
            sub = sub[sub[true_col].isin(order) & sub[pred_col].isin(order)]
            with cols_cm[i]:
                if len(sub) < 3:
                    st.warning(f"{gt_name}: dados insuficientes")
                    continue
                n_cls = len([c for c in order if c in sub[true_col].values or c in sub[pred_col].values])
                fsz = max(4.5, 0.55 * n_cls)
                fig, ax = plt.subplots(figsize=(fsz, fsz * 0.9))
                n, kv, av = _cm_plot(ax, sub, true_col, pred_col, order, cmap, gt_name)
                plt.tight_layout()
                st.pyplot(fig)
                st.caption(f"n={n} · κ={kv:.3f} · acc={av:.3f}")
                plt.close(fig)

    elif (vit_esfera is not None
          and "vit_esfera_pred" in vit_esfera.columns
          and "split" in vit_esfera.columns):
        esf_test = vit_esfera[vit_esfera["split"] == "test"]
        sub = esf_test[["true_monk", "vit_esfera_pred"]].dropna()
        sub = sub[sub["true_monk"].isin(MONK_ORDER) & sub["vit_esfera_pred"].isin(MONK_ORDER)]
        st.caption("GT = `true_monk` das esferas (test set) · Modelo treinado nas esferas")
        if len(sub) >= 3:
            n_cls = len([c for c in MONK_ORDER if c in sub["true_monk"].values or c in sub["vit_esfera_pred"].values])
            fsz = max(4.5, 0.55 * n_cls)
            fig, ax = plt.subplots(figsize=(fsz, fsz * 0.9))
            n, kv, av = _cm_plot(ax, sub, "true_monk", "vit_esfera_pred",
                                 MONK_ORDER, "Purples", "ViT Esferas (test)")
            plt.tight_layout()
            col_esf_cm, _ = st.columns([1, 1])
            with col_esf_cm:
                st.pyplot(fig)
                st.caption(f"n={n} · κ={kv:.3f} · acc={av:.3f}")
            plt.close(fig)
        else:
            st.warning("Poucos dados no split de teste das esferas.")
    else:
        st.info("Rode `finetune_vit.py --dataset esferas` para gerar predições das esferas.")

    st.divider()

    # ── Seção 3: ViT vs métodos clássicos no mesmo test set ───
    st.subheader("ViT vs Métodos Clássicos — MSKCC test (escala Monk, GT = MST R1)")
    st.caption("Todos os métodos avaliados nos mesmos isic_ids do test set IND.")

    test_ids   = set(vit_df[vit_df["split_mst_r1"] == "test"]["isic_id"])
    df_test_vit = df[df["isic_id"].isin(test_ids)].merge(
        vit_mskcc_raw[["isic_id", "vit_mst_r1_pred"]], on="isic_id", how="left"
    )

    _COMP_VIT = {
        "ViT MST R1": "vit_mst_r1_pred",
        "ITA Raw":    "ita_raw_monk",
        "ITA Smooth": "ita_smooth_monk",
        "ITA MSKCC":  "ita_mskcc_monk",
        "CASCo":      "stone_monk",
        "AIDA":       "aida_monk",
    }

    comp_rows = []
    for m_label, m_col in _COMP_VIT.items():
        if m_col not in df_test_vit.columns:
            continue
        sub = df_test_vit[["mst_r1_label", m_col]].dropna()
        sub = sub[sub["mst_r1_label"].isin(MONK_ORDER) & sub[m_col].isin(MONK_ORDER)]
        if len(sub) < 3:
            continue
        yt = pd.Categorical(sub["mst_r1_label"], categories=MONK_ORDER)
        yp = pd.Categorical(sub[m_col],          categories=MONK_ORDER)
        try:
            comp_rows.append({
                "Método":   m_label,
                "n":        len(sub),
                "κ (quad)": round(cohen_kappa_score(yt, yp, weights="quadratic"), 3),
                "Acurácia": round(accuracy_score(yt, yp), 3),
                "MAE":      ordinal_mae(sub["mst_r1_label"], sub[m_col], MONK_ORDER),
                "Acc ±1":   ordinal_acc1(sub["mst_r1_label"], sub[m_col], MONK_ORDER),
            })
        except Exception:
            pass

    if comp_rows:
        comp_tbl = pd.DataFrame(comp_rows).set_index("Método").sort_values("κ (quad)", ascending=False)
        st.dataframe(
            comp_tbl.style
                    .background_gradient(cmap="RdYlGn", subset=["κ (quad)", "Acurácia", "Acc ±1"],
                                         vmin=0, vmax=1, axis=None)
                    .background_gradient(cmap="RdYlGn_r", subset=["MAE"], vmin=0, vmax=5, axis=None)
                    .format("{:.3f}", subset=["κ (quad)", "Acurácia", "MAE", "Acc ±1"]),
            use_container_width=True,
        )
