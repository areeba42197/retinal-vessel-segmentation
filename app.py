"""Retinal Vessel Segmentation: full project dashboard + live demo.   Run:  streamlit run app.py
Everything except the "Live demo" tab reads saved results (no dataset or TensorFlow needed).
Research/educational project. Not a medical diagnostic system."""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(ROOT, *a)

st.set_page_config(page_title="Retinal Vessel Segmentation", layout="wide")

PAL = ["#7C5CFF", "#00D1FF", "#FF5C8A", "#FFC145", "#00E0A4", "#FF8A3D"]
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif;}
.stApp {background: radial-gradient(1200px 600px at 10% -10%, #1d1659 0%, transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, #06364a 0%, transparent 55%), #0A0F24;}
.block-container {padding-top:3.6rem; max-width:1400px;}
#MainMenu, footer {visibility:hidden;}
.hero {background: linear-gradient(120deg,#7C5CFF 0%,#00A3FF 55%,#00E0A4 100%); border-radius:22px; padding:30px 34px; margin-bottom:18px;
       box-shadow:0 18px 50px rgba(124,92,255,.35);}
.hero h1 {color:#fff; font-size:2.1rem; font-weight:800; margin:0 0 6px 0; letter-spacing:-.5px;}
.hero p {color:rgba(255,255,255,.92); margin:0; font-size:1.02rem;}
.badge {display:inline-block; background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.35); color:#fff; padding:4px 12px;
        border-radius:999px; font-size:.78rem; font-weight:600; margin:12px 8px 0 0; backdrop-filter:blur(4px);}
.stTabs [data-baseweb="tab-list"] {gap:6px; background:#101735; padding:6px; border-radius:14px; flex-wrap:wrap;}
.stTabs [data-baseweb="tab"] {height:42px; border-radius:10px; padding:0 16px; color:#AAB4DD; font-weight:600;}
.stTabs [aria-selected="true"] {background:linear-gradient(120deg,#7C5CFF,#00A3FF); color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#171F45,#0F1632); border:1px solid #28336B; border-radius:16px; padding:16px 18px;
        box-shadow:0 8px 24px rgba(0,0,0,.35); border-top:3px solid #7C5CFF;}
[data-testid="stMetricValue"] {font-size:1.75rem; font-weight:800; background:linear-gradient(90deg,#fff,#9fe8ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
[data-testid="stMetricLabel"] {color:#9FB0E8; font-weight:600;}
h2, h3 {font-weight:800 !important; letter-spacing:-.3px;}
[data-testid="stImage"] img {border-radius:14px; border:1px solid #28336B; box-shadow:0 8px 22px rgba(0,0,0,.4);}
[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden; border:1px solid #28336B;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#111846,#0A0F24); border-right:1px solid #28336B;}
.stAlert {border-radius:14px;}
.sect {font-size:.78rem; text-transform:uppercase; letter-spacing:2px; color:#00D1FF; font-weight:700; margin-bottom:-6px;}

.brand {font-size:1.35rem; font-weight:800; letter-spacing:2px; color:#fff; padding:6px 4px 14px 4px; line-height:1.1;}
.brand span {background:linear-gradient(90deg,#7C5CFF,#00D1FF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-left:6px;}
.brand-sub {font-size:.72rem; letter-spacing:.5px; font-weight:500; color:#8FA0D8; margin-top:6px;}
.navlabel {font-size:.7rem; letter-spacing:2px; color:#6F80C0; font-weight:700; margin:4px 0 6px 4px;}
[data-testid="stSidebar"] .stButton button {justify-content:flex-start; text-align:left; border:0; border-radius:10px; padding:9px 14px; font-weight:600; color:#B5C0EA; background:transparent;}
[data-testid="stSidebar"] .stButton button:hover {background:rgba(124,92,255,.18); color:#fff;}
[data-testid="stSidebar"] .stButton button[kind="primary"], [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {background:linear-gradient(120deg,#7C5CFF,#00A3FF); color:#fff; box-shadow:0 6px 18px rgba(124,92,255,.4);}
[data-testid="stSidebar"] .stButton button > div, [data-testid="stSidebar"] .stButton button p {justify-content:flex-start !important; text-align:left !important; width:100%;}
[data-testid="stSidebar"] .stButton {margin-bottom:-10px;}
</style>""", unsafe_allow_html=True)

import plotly.express as px


def _long(data):
    d = data.to_frame() if isinstance(data, pd.Series) else data.copy()
    d.index = d.index.astype(str)
    xn = d.index.name or "x"
    d = d.rename_axis(xn).reset_index()
    return d.melt(id_vars=xn, var_name="_series", value_name="_val"), xn, d.shape[1] - 1


def _finish(fig, ylab=None, legend=True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,26,56,.55)", height=340,
                      margin=dict(l=10, r=10, t=20, b=10), font=dict(family="Inter", size=13), showlegend=legend,
                      legend=dict(orientation="h", y=1.12, title=None), xaxis_title=None, yaxis_title=ylab,
                      hoverlabel=dict(bgcolor="#131A38"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)"); fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")
    return fig


def _bar(data, color=None, y_label=None, **kw):
    L, xn, n = _long(data)
    if n == 1 and color is None:
        fig = px.bar(L, x=xn, y="_val", color=xn, color_discrete_sequence=PAL)
        legend = False
    elif n == 1:
        fig = px.bar(L, x=xn, y="_val", color_discrete_sequence=[color]); legend = False
    else:
        fig = px.bar(L, x=xn, y="_val", color="_series", barmode="group", color_discrete_sequence=PAL); legend = True
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}: %{y:.4f}<extra></extra>" if n == 1 else None)
    st.plotly_chart(_finish(fig, y_label, legend), use_container_width=True)


def _line(data, **kw):
    L, xn, n = _long(data)
    fig = px.line(L, x=xn, y="_val", color="_series", color_discrete_sequence=PAL, markers=len(L) < 80)
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(_finish(fig, None, n > 1), use_container_width=True)


st.bar_chart, st.line_chart = _bar, _line


@st.cache_data
def csv(*a):
    return pd.read_csv(P(*a))


@st.cache_data
def js(*a):
    return json.load(open(P(*a)))


def img(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing file: {os.path.relpath(path, ROOT)}")


NAMES = {"E1": "U-Net · BCE · no aug", "E2": "U-Net · BCE+Dice · no aug", "E3": "U-Net · BCE+Dice · aug",
         "E4": "U-Net · BCE+Dice · aug · CLAHE", "E5": "EfficientNet-B0 U-Net · aug", "E6": "EfficientNet-B0 U-Net · aug · CLAHE"}
res = csv("experiments", "results.csv")
sel = js("experiments", "final_selection.json")
test = js("experiments", "final_test_metrics.json")
M = test["metrics"]
frangi = js("experiments", "frangi_baseline.json")

st.markdown(f"""<div class="hero"><h1>Retinal Blood Vessel Segmentation</h1>
<p>Binary semantic segmentation of the retinal vasculature in colour fundus images using U-Net and EfficientNet-B0 encoder-decoder CNNs (DRIVE dataset)</p>
<span class="badge">Test Dice {M['dice']:.3f}</span><span class="badge">ROC-AUC {M['roc_auc']:.3f}</span>
<span class="badge">Final model {sel['final']}</span><span class="badge">Research / educational — not a diagnostic tool</span></div>""", unsafe_allow_html=True)

PAGES = ['Overview', 'Dataset', 'Pipeline', 'Training', 'Ablation', 'Test Results', 'Error Analysis', 'Baseline', 'Gallery', 'Live Demo', 'Limitations']
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
with st.sidebar:
    st.markdown('<div class="brand">RETINAL<span>VESSEL</span><div class="brand-sub">Deep-learning segmentation on DRIVE</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="navlabel">NAVIGATION</div>', unsafe_allow_html=True)
    for _p in PAGES:
        if st.button(_p, key=f"nav_{_p}", use_container_width=True, type="primary" if st.session_state.page == _p else "secondary"):
            st.session_state.page = _p
            st.rerun()
    st.markdown('<div class="navlabel" style="margin-top:22px">PROJECT SUMMARY</div>', unsafe_allow_html=True)
    st.markdown(f"""
**Task** binary semantic segmentation  
**Dataset** DRIVE, 20 annotated images  
**Partition** 12 train / 4 val / 4 test  
**Selected model** {NAMES[sel['final']]}  
**Threshold** {sel['threshold']}  
**Primary metric** Dice (DSC)
""")
    st.caption("Research and educational use only. Not a medical diagnostic system.")

page_idx = PAGES.index(st.session_state.page)

# ------------------------------------------------------------------ Overview
if page_idx == 0:
    vd = res.set_index("experiment_id").loc[sel["final"], "val_dice"]
    e4 = res.set_index("experiment_id").loc["E4", "val_dice"]
    ea = csv("results", "tables", "test_error_analysis.csv")
    st.markdown('<div class="sect">Project overview</div>', unsafe_allow_html=True)
    st.header("Problem statement and objective")
    st.markdown("""
The morphology of the retinal vasculature (vessel calibre, tortuosity and branching) is an important indicator in ophthalmic and systemic disease screening.
Manual annotation of vessels in **colour fundus photographs (CFP)** is slow and varies between observers.

This project develops and evaluates **deep convolutional neural networks (CNNs)** for **automatic binary semantic segmentation of the retinal vasculature**:
every pixel of a fundus image is classified as *vessel* or *background*, producing a vessel mask.

**Objectives**
1. Build a leakage-free, reproducible pipeline on the DRIVE dataset (image-level train / validation / test partition).
2. Compare a **U-Net** with an **EfficientNet-B0 encoder U-Net** in a controlled ablation (loss function, data augmentation, CLAHE, encoder).
3. Select the model and decision threshold on the validation set only, then evaluate **once** on a held-out test set.
4. Analyse failure modes and compare against a classical baseline (multiscale **Frangi vesselness filter**).

> **Scope:** research prototype for vessel segmentation only. It is not a diagnostic system and has not been clinically validated.
""")
    gi = js("results", "gallery", "index.json")
    st.markdown("#### Qualitative example (held-out test image)")
    i0 = gi["test_ids"][0]
    c = st.columns(3)
    for col, (n, cap) in zip(c, [("original", "Input: colour fundus photograph"), ("gt", "Ground truth: manual annotation"), ("pred", "Prediction: segmentation mask")]):
        with col:
            img(P("results", "gallery", f"test_{i0}_{n}.png"), cap)
    st.caption(f"Test image {i0}, Dice similarity coefficient (DSC) = {test['per_image_dice'][str(i0)]:.3f}. The closer the prediction is to the ground truth, the higher the DSC (maximum 1.0).")

    st.markdown("#### Methodology")
    s1, s2, s3, s4 = st.columns(4)
    s1.info("**1. Dataset**\n\nDRIVE: 20 annotated fundus images (584 × 565 RGB) with manual vessel masks and field-of-view (FOV) masks.")
    s2.info("**2. Training**\n\n12 training images. Random 128 × 128 patches with geometric and photometric augmentation. BCE + Dice loss, AdamW optimiser.")
    s3.info("**3. Model selection**\n\n4 validation images. Six experiments (E1 to E6) are ranked by validation Dice. The decision threshold is tuned here.")
    s4.info("**4. Test evaluation**\n\n4 held-out test images, evaluated once with the selected model and fixed threshold. Metrics are computed inside the FOV.")

    st.markdown("#### Main results (held-out test set)")
    c = st.columns(4)
    c[0].metric("Dice (DSC)", f"{M['dice']:.3f}", help="Overlap between predicted and ground-truth vessel masks: 2TP / (2TP + FP + FN). 1.0 is a perfect match.")
    c[1].metric("Sensitivity", f"{M['recall']:.3f}", help="Recall / true positive rate: the fraction of true vessel pixels that were detected.")
    c[2].metric("Precision", f"{M['precision']:.3f}", help="Positive predictive value: the fraction of predicted vessel pixels that are truly vessel.")
    c[3].metric("Specificity", f"{M['specificity']:.3f}", help="True negative rate: the fraction of background pixels correctly classified as background.")
    st.markdown(f"""
The selected model (**{sel['final']}**) reaches a Dice similarity coefficient of **{M['dice']:.3f}** (IoU {M['iou']:.3f}, ROC-AUC {M['roc_auc']:.3f}) on the held-out test set.
The classical Frangi vesselness baseline reaches a Dice of only **{frangi['test']['dice']:.3f}**, so the CNN improves substantially on the hand-designed filter.

**Main limitation:** thin vessels. Skeleton recall is **{ea.recall_thick_vessels.min():.2f} to {ea.recall_thick_vessels.max():.2f}** for thick vessels but only
**{ea.recall_thin_vessels.min():.2f} to {ea.recall_thin_vessels.max():.2f}** for thin vessels (local half-width of 1.5 px or less), where fine branches are missed or broken.
""")

    st.markdown("#### Model selection")
    st.markdown(f"""
Six configurations (E1 to E6) were compared by varying the loss function (BCE vs BCE + Dice), augmentation, CLAHE contrast enhancement and the encoder (U-Net vs EfficientNet-B0 U-Net).
**{sel['final']} ({NAMES[sel['final']]})** was selected on validation Dice ({vd:.3f}). E4, which adds CLAHE, is statistically indistinguishable ({e4:.3f}), so the simpler model was kept.
Test Dice ({M['dice']:.3f}) is consistent with validation Dice ({vd:.3f}), which indicates no meaningful selection bias.
""")

    with st.expander("Additional metrics (IoU, F1, MCC, ROC-AUC, PR-AUC, clDice)"):
        c = st.columns(6)
        for col, (lab, k) in zip(c, [("IoU", "iou"), ("F1", "f1"), ("MCC", "mcc"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("clDice", "cldice")]):
            col.metric(lab, f"{M[k]:.3f}")
    with st.expander("Terminology"):
        st.markdown("""
- **Semantic segmentation**: assigning a class label (vessel / background) to every pixel.
- **CNN**: convolutional neural network, a deep model that learns image features from data.
- **U-Net**: encoder-decoder CNN with skip connections, widely used for biomedical segmentation.
- **EfficientNet-B0**: a compact CNN architecture (MBConv blocks) used here as the U-Net encoder.
- **Dice (DSC) / IoU**: overlap measures between prediction and ground truth (higher is better, maximum 1.0).
- **Sensitivity / Specificity / Precision**: detected fraction of vessels / correctly rejected background / correctness of predicted vessels.
- **Decision threshold**: probability above which a pixel is labelled vessel (0.45, selected on validation).
- **Data augmentation**: random flips, rotations, scaling and brightness changes applied to training patches.
- **CLAHE**: contrast-limited adaptive histogram equalisation, an image contrast enhancement.
- **FOV**: field of view, the circular retinal region of a fundus image. Metrics ignore pixels outside it.
- **Validation / test set**: images used for model selection / images reserved for the single final evaluation.
- **Ablation study**: removing or changing one component at a time to measure its effect.
- **Frangi filter**: classical multiscale filter that enhances tubular structures. Used as a non-learning baseline.
""")
    st.warning("**Scope and limitations:** only 20 annotated images are available, so validation and test sets contain 4 images each and results carry high variance. "
               "Each experiment was run once (single seed). Training was CPU-only and several runs peaked near the epoch limit, so the models are likely under-trained. "
               "The EfficientNet-B0 encoder was randomly initialised because ImageNet weights could not be downloaded.")


# ------------------------------------------------------------------ Dataset & EDA
if page_idx == 1:
    st.info('**Content:** dataset description, image-level train / validation / test partition, FOV statistics and the vessel / background class imbalance (about 12.5% vessel pixels).')
    st.subheader("Dataset report (DRIVE)")
    st.code(open(P("results", "tables", "dataset_report.txt")).read(), language="text")
    stats = csv("results", "tables", "mask_and_intensity_stats.csv")
    a, b, c = st.columns(3)
    a.metric("Labelled images", len(stats)); b.metric("Mean vessel share of FOV", f"{stats.vessel_pct.mean():.1f}%")
    c.metric("Image size", "584 × 565 RGB")
    st.markdown("**Split (image level, fixed seed)**")
    sp = stats.groupby("split").id.apply(lambda s: ", ".join(map(str, s))).rename("image IDs").to_frame()
    sp["n"] = stats.groupby("split").size()
    st.dataframe(sp.loc[["train", "val", "test"]], use_container_width=True)
    st.markdown("**Vessel share per image (class imbalance)**")
    st.bar_chart(stats.set_index("id")["vessel_pct"], y_label="% of FOV pixels", color="#3b6fb6")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mean green-channel intensity per image (illumination)**")
        st.bar_chart(stats.set_index("id")["green_mean"], color="#2e8b57")
    with c2:
        st.markdown("**Contrast (p95 − p5 of green channel)**")
        st.bar_chart(stats.set_index("id")["contrast_p95_p5"], color="#c0762d")
    st.dataframe(stats.round(2), use_container_width=True, height=260)
    img(P("results", "eda", "fig1_samples_pairs.png"), "Image / vessel mask / overlay (alignment check)")
    c1, c2 = st.columns(2)
    with c1:
        img(P("results", "eda", "fig3_class_imbalance.png"), "Class imbalance")
    with c2:
        img(P("results", "eda", "intensity_analysis.png"), "Intensity analysis")
    img(P("results", "eda", "difficult_and_green_channel.png"), "Difficult examples, green channel and CLAHE")

# ------------------------------------------------------------------ Preprocessing & augmentation
if page_idx == 2:
    st.info('**Content:** preprocessing (FOV masking, CLAHE) and synchronised geometric and photometric data augmentation applied to image and mask.')
    cfg = js("experiments", "E3_summary.json")["config"]
    st.subheader("Preprocessing")
    st.markdown("""
| Mode | Description | Used in |
|--|--|--|
| basic | RGB / 255, zero outside field of view (FOV) | E1, E2, E3 (final), E5 |
| clahe | CLAHE on LAB lightness (clip 2.0, 8×8 tiles) | E4, E6 |
| green | green channel ×3 | implemented, not trained |
""")
    st.markdown("Loss and all metrics are computed **only inside the FOV**; patches are sampled inside the FOV; patches are cut **after** the image-level split (no leakage).")
    st.subheader("Augmentation (applied identically to image and mask)")
    st.json(cfg["augmentation"])
    img(P("results", "eda", "augmentation_check.png"), "Synchronised augmentation check: patches (top) and masks (bottom)")

# ------------------------------------------------------------------ Models & training
if page_idx == 3:
    st.info('**Content:** loss, Dice and learning-rate curves for each experiment. Use the selector to switch between E1 and E6.')
    st.subheader("Training curves")
    exp = st.selectbox("Experiment", list(NAMES), index=2, format_func=lambda e: f"{e} — {NAMES[e]}")
    h = csv("experiments", f"{exp}_history.csv")
    s = js("experiments", f"{exp}_summary.json")
    a, b, c, d = st.columns(4)
    a.metric("Parameters", f"{s['params']:,}"); b.metric("Epochs run", s["epochs_run"])
    c.metric("Best epoch (val Dice)", s["best_epoch"]); d.metric("Training time", f"{s['training_time'] / 60:.1f} min")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Loss"); st.line_chart(h.set_index("epoch")[["loss", "val_loss"]])
        st.markdown("Learning rate"); st.line_chart(h.set_index("epoch")[["lr"]])
    with c2:
        st.markdown("Dice (train = soft Dice on augmented patches · val = full images, FOV only)")
        st.line_chart(h.set_index("epoch")[["dice_metric", "val_dice"]])
        st.markdown("Validation IoU"); st.line_chart(h.set_index("epoch")[["val_iou"]])
    st.caption("Reading the curves: validation loss follows training loss (no overfitting); the best epochs of several runs are close to the 40-epoch cap, "
               "so models were probably still improving (under-trained).")
    with st.expander("Full configuration of this run"):
        st.json(s["config"])
    st.subheader("Model comparison")
    mt = pd.DataFrame([{"Exp": e, "Model": NAMES[e], "Params": js("experiments", f"{e}_summary.json")["params"],
                        "Trainable": js("experiments", f"{e}_summary.json")["trainable_params"],
                        "Train time (min)": round(js("experiments", f"{e}_summary.json")["training_time"] / 60, 1)} for e in NAMES])
    st.dataframe(mt, use_container_width=True, hide_index=True)
    st.markdown("Optimizer AdamW (lr 1e-3, weight decay 1e-4) · ReduceLROnPlateau on val loss · early stopping + best checkpoint on val Dice · batch 16 × 128×128 patches.")

# ------------------------------------------------------------------ Ablation
if page_idx == 4:
    st.info('**Content:** controlled ablation of loss function, augmentation, CLAHE and encoder. Models are ranked by validation Dice. The test set is not used for selection.')
    st.subheader("Ablation (validation set, 4 images, single seed)")
    t = res[res.experiment_id.str.match(r"^E\d+$")].copy()
    t["Setup"] = t.experiment_id.map(NAMES)
    show = t[["experiment_id", "Setup", "val_dice", "val_iou", "best_epoch", "training_time"]].rename(
        columns={"experiment_id": "Exp", "val_dice": "Val Dice", "val_iou": "Val IoU", "best_epoch": "Best epoch", "training_time": "Train time (s)"})
    st.dataframe(show.round(4), use_container_width=True, hide_index=True)
    st.bar_chart(t.set_index("experiment_id")[["val_dice", "val_iou"]])
    d = t.set_index("experiment_id").val_dice
    st.markdown(f"""
| Change | Δ Val Dice |
|--|--:|
| BCE → BCE+Dice (E1→E2) | {d['E2'] - d['E1']:+.4f} |
| + augmentation (E2→E3) | {d['E3'] - d['E2']:+.4f} |
| + CLAHE (E3→E4) | {d['E4'] - d['E3']:+.4f} |
| U-Net → EffNet-B0 U-Net, random init (E3→E5) | {d['E5'] - d['E3']:+.4f} |
| + CLAHE on EffNet (E5→E6) | {d['E6'] - d['E5']:+.4f} |
""")
    st.info(f"**Selected: {sel['final']}.** {sel['note']}. Differences of ≈0.01 are within plausible seed/split noise (4 validation images, one seed).")
    st.subheader("Threshold selection (validation only)")
    th = csv("results", "tables", "threshold_sweep_val.csv")
    st.line_chart(th.set_index("threshold"))
    st.caption(f"Locked threshold: {sel['threshold']}. Dice is almost flat between 0.30 and 0.70.")

# ------------------------------------------------------------------ Test results
if page_idx == 5:
    st.info('**Content:** single evaluation of the selected model on the held-out test split. Metric definitions are under Overview, Terminology.')
    st.subheader(f"Held-out test — {sel['final']}, threshold {sel['threshold']} (evaluated once)")
    mt = pd.DataFrame({"metric": ["Dice", "IoU", "Precision", "Recall / Sensitivity", "Specificity", "F1", "MCC", "ROC-AUC", "PR-AUC", "clDice", "Pixel accuracy (supplementary)"],
                       "value": [M[k] for k in ["dice", "iou", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "pr_auc", "cldice", "accuracy"]]})
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(mt.round(4), use_container_width=True, hide_index=True)
    with c2:
        st.bar_chart(mt.set_index("metric").iloc[:-1])
    per = pd.Series(test["per_image_dice"], name="Dice")
    st.markdown("**Per-image Dice**")
    st.bar_chart(per)
    st.caption(f"Mean {M['dice_mean_per_image']:.4f} ± {M['dice_std_per_image']:.4f} across the 4 test images. HD95 not computed.")
    st.markdown("**Validation vs test (sanity check for selection bias)**")
    vd = res.set_index("experiment_id").loc[sel["final"], "val_dice"]
    st.dataframe(pd.DataFrame({"Dice": {"Validation (selection)": vd, "Test (held out)": M["dice"]}}).round(4))

# ------------------------------------------------------------------ Error analysis
if page_idx == 6:
    st.info('**Content:** failure analysis. In the error maps: white = true positive, red = false positive, blue = false negative, black = true negative.')
    st.subheader("Where does the model fail?")
    ea = csv("results", "tables", "test_error_analysis.csv")
    st.dataframe(ea.rename(columns={"recall_thin_vessels": "recall thin", "recall_thick_vessels": "recall thick"}), use_container_width=True, hide_index=True)
    st.bar_chart(ea.set_index("id")[["recall_thin_vessels", "recall_thick_vessels"]])
    st.markdown(f"""
- **Thick vessels** are found almost perfectly (skeleton recall {ea.recall_thick_vessels.min():.2f}–{ea.recall_thick_vessels.max():.2f}).
- **Thin vessels** (local half-width ≤ 1.5 px) are the main weakness (recall {ea.recall_thin_vessels.min():.2f}–{ea.recall_thin_vessels.max():.2f}): broken or missed fine branches.
- False positives are mostly short spurs along true vessels and spots near the optic disc.
""")
    st.markdown("**Error maps**  ·  white = TP · red = FP · blue = FN · black = TN")
    img(P("results", "error_maps", "fig5_6_test_predictions_error_maps.png"))

# ------------------------------------------------------------------ Classical baseline
if page_idx == 7:
    st.info('**Content:** comparison of the selected CNN with the classical multiscale Frangi vesselness filter on the same held-out test images.')
    st.subheader("Classical Frangi vesselness filter vs CNN")
    st.caption("Frangi vesselness on the inverted green channel; its single threshold was fitted on train+val. Not part of the deep model.")
    cmp_ = pd.DataFrame({"Frangi (classical)": {k: frangi["test"][k] for k in ["dice", "iou", "precision", "recall", "specificity", "roc_auc", "pr_auc", "mcc", "cldice"]},
                         f"{sel['final']} deep model": {k: M[k] for k in ["dice", "iou", "precision", "recall", "specificity", "roc_auc", "pr_auc", "mcc", "cldice"]}})
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(cmp_.round(4))
    with c2:
        st.bar_chart(cmp_)
    img(P("results", "comparisons", "fig9_frangi_vs_deep.png"), "Test images: Frangi vs final model vs ground truth")

# ------------------------------------------------------------------ Gallery
if page_idx == 8:
    st.info("**Content:** per-image qualitative results. The 20 official DRIVE test images have no public annotation here, so they are shown qualitatively only (no metrics).")
    gi = js("results", "gallery", "index.json")
    st.subheader("Held-out test images (with ground truth)")
    i = st.selectbox("Test image", gi["test_ids"], format_func=lambda x: f"image {x} — Dice {test['per_image_dice'][str(x)]:.3f}")
    G = lambda n: P("results", "gallery", f"test_{i}_{n}.png")
    r1 = st.columns(3); r2 = st.columns(3)
    for col, (n, cap) in zip(list(r1) + list(r2), [("original", "Original"), ("gt", "Ground truth"), ("pred", "Prediction"),
                                                   ("prob", "Probability heat-map"), ("overlay", "Overlay"), ("error", "Error map (red FP · blue FN)")]):
        with col:
            img(G(n), cap)
    st.divider()
    st.subheader("Official DRIVE test images (no ground truth → qualitative only, no metrics)")
    j = st.slider("Official test image", 1, gi["official_test_n"], 1)
    img(P("results", "gallery", "official_test", f"{j:02d}.jpg"), f"Image {j:02d}: original | predicted mask | overlay")

# ------------------------------------------------------------------ Live demo
if page_idx == 9:
    st.info('**Content:** inference on a user-supplied fundus image with the selected model and an adjustable threshold. Requires TensorFlow.')
    st.subheader("Try it on your own fundus image")
    st.caption("Runs the final model locally. Needs TensorFlow installed (see requirements.txt).")
    up = st.file_uploader("Retinal fundus image", type=["png", "jpg", "jpeg", "tif", "tiff", "gif"])
    gt_up = st.file_uploader("Optional ground-truth vessel mask (same size)", type=["png", "gif", "tif", "jpg"])
    if up is not None:
        from PIL import Image
        try:
            from src.inference import load_final, segment

            @st.cache_resource
            def get_model():
                return load_final()

            @st.cache_data(show_spinner="Segmenting…")
            def run(b, name):
                m, c_ = get_model()
                im = np.array(Image.open(up).convert("RGB"))
                p, _, f = segment(m, c_, im)
                return im, p, f
            im, prob, fov = run(up.getvalue(), up.name)
            _, cfg_ = get_model()
            thr = st.slider("Threshold (default = validation-selected)", 0.05, 0.95, float(cfg_["threshold"]), 0.05)
            mask = prob > thr
            ov = im.copy(); ov[mask] = (0, 255, 0)
            c1, c2, c3, c4 = st.columns(4)
            c1.image(im, "Original Image", use_container_width=True)
            c2.image((np.clip(prob, 0, 1) * 255).astype(np.uint8), "Probability map", use_container_width=True)
            c3.image((mask * 255).astype(np.uint8), "Predicted Vessel Mask", use_container_width=True)
            c4.image(ov, "Overlay", use_container_width=True)
            st.caption("FOV is estimated automatically. Values outside the retina are zeroed.")
            if gt_up is None:
                st.info("No ground-truth mask provided. Metrics cannot be calculated.")
            else:
                from src.metrics import evaluate_set
                gt = np.array(Image.open(gt_up).convert("L")) > 127
                if gt.shape != mask.shape:
                    st.error("Ground-truth mask size differs from the image; metrics not computed.")
                else:
                    m_, _ = evaluate_set([prob], [gt], [fov], thr, with_auc=False)
                    cols = st.columns(4)
                    for col, k in zip(cols, ["dice", "iou", "precision", "recall"]):
                        col.metric(k.capitalize(), f"{m_[k]:.3f}")
        except ModuleNotFoundError as e:
            st.error(f"Missing package for the live demo: {e}. Install requirements.txt.")
    else:
        st.info("Upload an image to run the model. The other tabs work without TensorFlow or the dataset.")

# ------------------------------------------------------------------ Limitations
if page_idx == 10:
    st.info('**Content:** limitations of this study and directions for future work.')
    st.subheader("Limitations")
    st.markdown("""
- Only 20 labelled images; validation and test sets contain **4 images each** → wide uncertainty, no confidence intervals.
- **One seed** per experiment; differences of ≈0.01 Dice should not be over-interpreted.
- EfficientNet-B0 encoder is **randomly initialised** (ImageNet weights could not be downloaded) — no pretrained comparison was possible.
- CPU-only training with small 128×128 patches; several runs peaked near the epoch cap → likely **under-trained**.
- Not run: Attention U-Net, 5-fold cross-validation, CHASE_DB1 cross-dataset test, HD95, clDice as a loss. Flip-TTA was checked on validation only (not used).
- Single dataset; domain shift untested. **Not clinically validated; not for diagnosis.**

**Future work:** pretrained encoders, longer multi-seed training / cross-validation, external datasets, topology-aware losses, attention mechanisms.
""")
    st.subheader("Reproducibility")
    st.code(f"seed: 42 · split_seed: 42\ntrain: 12 images · val: 4 · test: 4  (IDs in splits/*.txt)\n"
            f"final model: {sel['final']} · threshold {sel['threshold']}", language="text")
