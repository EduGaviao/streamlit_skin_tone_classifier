import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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

# ===============================================================
#                     DATA
# ===============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["fst_label"]    = df["fst"].apply(lambda x: f"FITZPATRICK {int(x)}" if pd.notna(x) else np.nan)
    df["mst_r1_label"] = df["mst_r1"].apply(lambda x: f"MONK {int(x)}" if pd.notna(x) else np.nan)
    df["mst_r2_label"] = df["mst_r2"].apply(lambda x: f"MONK {int(x)}" if pd.notna(x) else np.nan)
    return df

df = load_data()

# ===============================================================
#                     HEADER
# ===============================================================

st.title("🔬 MSKCC — Avaliação de Métodos de Classificação de Tom de Pele")
st.caption(f"Dataset: `{CSV_PATH}` · {len(df)} imagens")

tab1, tab2, tab3 = st.tabs([
    "📊 ITA: Computado vs MSKCC",
    "🎯 Precisão por Método",
    "🧬 PCA Lab",
])

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
            "ITA Raw":    "ita_raw_fitz",
            "ITA Smooth": "ita_smooth_fitz",
            "ITA MSKCC":  "ita_mskcc_fitz",
            "Stone":      "stone_fitz",
            "AIDA":       "aida_fitz",
        }
        class_order = FITZ_ORDER
        cmap = "Blues"
    else:
        gt_options = {
            "MST R1 (ground truth)": "mst_r1_label",
            "MST R2 (ground truth)": "mst_r2_label",
        }
        pred_options = {
            "ITA Raw":    "ita_raw_monk",
            "ITA Smooth": "ita_smooth_monk",
            "ITA MSKCC":  "ita_mskcc_monk",
            "Stone":      "stone_monk",
            "AIDA":       "aida_monk",
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
