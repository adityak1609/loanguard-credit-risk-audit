import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from loanguard.features import FeatureSpec  # noqa: E402
from loanguard.config import GRADE_MAP, PROCESSED  # noqa: E402

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="LoanGuard | Credit-Risk Model Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.stApp {
    background: #0f1117;
    color: #e2e8f0;
}

h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #0c1821 0%, #0f1b2d 40%, #112240 100%);
    border: 1px solid rgba(45, 212, 191, 0.15);
    border-radius: 16px;
    padding: 40px 44px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, rgba(45, 212, 191, 0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40%;
    left: -10%;
    width: 50%;
    height: 150%;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(45, 212, 191, 0.1);
    border: 1px solid rgba(45, 212, 191, 0.25);
    color: #2dd4bf;
    padding: 5px 16px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 10px 0;
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(135deg, #2dd4bf 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #64748b;
    font-size: 0.92rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.6;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569;
    margin: 0 0 18px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e293b;
}

/* ── Input Cards ── */
.input-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

/* ── Result Cards ── */
.result-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.result-card.high {
    border-color: rgba(251, 146, 60, 0.4);
    background: linear-gradient(145deg, #111827 0%, #1c1310 100%);
}
.result-card.medium {
    border-color: rgba(250, 204, 21, 0.3);
    background: linear-gradient(145deg, #111827 0%, #1a1812 100%);
}
.result-card.low {
    border-color: rgba(45, 212, 191, 0.3);
    background: linear-gradient(145deg, #111827 0%, #0f1d1a 100%);
}

.prob-display {
    font-family: 'Inter', sans-serif;
    font-size: 4.2rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
    letter-spacing: -0.03em;
}
.prob-display.high { color: #fb923c; }
.prob-display.medium { color: #facc15; }
.prob-display.low { color: #2dd4bf; }

.risk-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 8px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.risk-label.high { color: #fb923c; }
.risk-label.medium { color: #facc15; }
.risk-label.low { color: #2dd4bf; }

.risk-desc {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 10px;
    line-height: 1.55;
}

/* ── Progress Bar ── */
.prob-bar-container {
    background: #1e293b;
    border-radius: 100px;
    height: 5px;
    margin: 20px 0;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}
.prob-bar.high { background: linear-gradient(90deg, #fb923c, #f97316); }
.prob-bar.medium { background: linear-gradient(90deg, #facc15, #eab308); }
.prob-bar.low { background: linear-gradient(90deg, #2dd4bf, #14b8a6); }

/* ── Stat Pills ── */
.stat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 20px;
}
.stat-pill {
    background: #0f1117;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 10px 14px;
    flex: 1;
    min-width: 90px;
}
.stat-pill-label {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 4px;
    font-weight: 500;
}
.stat-pill-value {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #e2e8f0;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0d9488, #14b8a6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(45, 212, 191, 0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(45, 212, 191, 0.35) !important;
    background: linear-gradient(135deg, #14b8a6, #2dd4bf) !important;
}

/* ── Streamlit element overrides ── */
.stSelectbox > div > div {
    background: #111827 !important;
    border-color: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div {
    background: #14b8a6 !important;
}
label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stNumberInput > div > div > input {
    background: #111827 !important;
    border-color: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── SHAP section ── */
.shap-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 16px;
}
.shap-header {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.shap-sub {
    color: #475569;
    font-size: 0.78rem;
    margin-bottom: 20px;
    line-height: 1.5;
}

/* ── Await card ── */
.await-card {
    background: #111827;
    border: 1px dashed #1e293b;
    border-radius: 16px;
    padding: 56px 32px;
    text-align: center;
}
.await-icon {
    font-size: 2.5rem;
    margin-bottom: 14px;
    opacity: 0.5;
}
.await-title {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #334155;
    margin: 0 0 6px 0;
}
.await-sub {
    color: #334155;
    font-size: 0.78rem;
    margin: 0;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────
# The feature spec is loaded from the same artefact the model was fitted with,
# so the column set cannot drift from training. It used to be retyped by hand
# below, which silently zero-filled any name that did not match.
@st.cache_resource
def load_artifacts():
    model = joblib.load(PROCESSED / 'lgbm_model.pkl')
    calibrator = joblib.load(PROCESSED / 'calibrator.pkl')
    spec = FeatureSpec.load(PROCESSED / 'feature_spec.json')
    cfg = json.loads((PROCESSED / 'serving_config.json').read_text())
    return model, calibrator, spec, cfg

model, calibrator, spec, cfg = load_artifacts()
THRESHOLD = cfg['threshold']

# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Audit-First Credit Risk</div>
    <p class="hero-title">Loan<span>Guard</span></p>
    <p class="hero-sub">Calibrated default probability with per-application explanations.<br>
    Trained on 1.35M resolved LendingClub loans (2007–2018) · test AUC 0.724 · ECE 0.002.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<p class="section-header">Borrower Profile</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        loan_amnt  = st.number_input("Loan Amount ($)", 1000, 40000, 10000, step=500)
        annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
        fico       = st.slider("FICO Credit Score", 580, 850, 700)
        emp_length = st.selectbox("Employment (years)", [0,1,2,3,4,5,6,7,8,9,10])
        open_acc   = st.number_input("Open Accounts", 1, 40, 10)
        mort_acc   = st.number_input("Mortgage Accounts", 0, 20, 1)

    with c2:
        int_rate   = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, step=0.1)
        dti        = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0, step=0.5)
        revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0)
        revol_bal  = st.number_input("Revolving Balance ($)", 0, 100000, 15000, step=500)
        pub_rec    = st.number_input("Public Records", 0, 10, 0)
        grade      = st.selectbox("Loan Grade", ['A','B','C','D','E','F','G'])

    home_ownership = st.selectbox("Home Ownership", ['RENT','OWN','MORTGAGE','OTHER'])
    purpose = st.selectbox("Loan Purpose", [
        'debt_consolidation','credit_card','home_improvement','other',
        'major_purchase','small_business','car','medical','moving',
        'vacation','house','wedding','renewable_energy','educational'])

    st.markdown('<p class="section-header">Credit File</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        term_months  = st.selectbox("Term (months)", [36, 60])
        sub_grade_n  = st.selectbox("Sub-grade", [1, 2, 3, 4, 5])
        total_acc    = st.number_input("Total Accounts", 1, 100, 25)
    with c4:
        delinq_2yrs      = st.number_input("Delinquencies (2 yrs)", 0, 20, 0)
        inq_last_6mths   = st.number_input("Inquiries (6 mths)", 0, 20, 1)
        credit_hist_yrs  = st.slider("Credit History (years)", 0.0, 50.0, 15.0, step=0.5)
    verification_status = st.selectbox(
        "Income Verification", ['Verified', 'Source Verified', 'Not Verified'])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analyze Default Risk →", type="primary")

# ── Right Panel ──────────────────────────────────────────────
with right:
    st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div class="await-card">
            <div class="await-icon">🛡️</div>
            <p class="await-title">Awaiting Analysis</p>
            <p class="await-sub">Fill in borrower details and click analyze</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Scheduled monthly payment, from the standard amortisation formula.
        r = int_rate / 100 / 12
        installment = (
            loan_amnt * r / (1 - (1 + r) ** -term_months) if r > 0
            else loan_amnt / term_months
        )

        issue_d = pd.Timestamp.today().normalize()
        raw = pd.DataFrame([{
            'loan_amnt': loan_amnt, 'int_rate': int_rate,
            'annual_inc': annual_inc, 'dti': dti,
            'grade': GRADE_MAP[grade],
            'sub_grade': GRADE_MAP[grade] * 5 + sub_grade_n,
            'emp_length': emp_length, 'fico_range_low': fico,
            'open_acc': open_acc, 'revol_util': revol_util,
            'revol_bal': revol_bal, 'mort_acc': mort_acc, 'pub_rec': pub_rec,
            'total_acc': total_acc, 'delinq_2yrs': delinq_2yrs,
            'inq_last_6mths': inq_last_6mths, 'installment': installment,
            'term_months': term_months,
            'home_ownership': home_ownership, 'purpose': purpose,
            'verification_status': verification_status,
            'issue_d': issue_d,
            'earliest_cr_line': issue_d - pd.Timedelta(days=credit_hist_yrs * 365.25),
        }])

        # One code path shared with training: engineering, encoding, and
        # column alignment all come from the saved FeatureSpec.
        input_df = spec.transform(raw)

        # Isotonic calibration maps the raw score onto an actual probability.
        # The previous build trained with scale_pos_weight and displayed the
        # raw output as a percentage, which overstated risk by roughly 3x.
        raw_score = model.predict_proba(input_df)[0][1]
        prob = float(np.clip(calibrator.predict([raw_score])[0], 0, 1))
        pct  = prob * 100

        if prob >= THRESHOLD:
            tier = "high"
            desc = (f"Above the {THRESHOLD:.0%} decline threshold, which was set by "
                    "minimising expected loss on held-out data.")
        elif prob >= THRESHOLD * 0.6:
            tier = "medium"
            desc = "Below the decline threshold but in the upper risk band. Manual review recommended."
        else:
            tier = "low"
            desc = "Well inside the approve band for this portfolio's loss economics."

        tier_icons = {"high": "⚠️", "medium": "◉", "low": "✓"}
        tier_labels = {"high": "HIGH RISK", "medium": "ELEVATED", "low": "LOW RISK"}

        # Result Card
        st.markdown(f"""
        <div class="result-card {tier}">
            <p class="prob-display {tier}">{pct:.1f}%</p>
            <p class="risk-label {tier}">
                {tier_icons[tier]} {tier_labels[tier]}
            </p>
            <p class="risk-desc">{desc}</p>
            <div class="prob-bar-container">
                <div class="prob-bar {tier}" style="width: {pct}%"></div>
            </div>
            <div class="stat-row">
                <div class="stat-pill">
                    <div class="stat-pill-label">vs Portfolio Base</div>
                    <div class="stat-pill-value">{pct / 19.97:.2f}x</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Decline At</div>
                    <div class="stat-pill-value">{THRESHOLD:.0%}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">FICO Score</div>
                    <div class="stat-pill-value">{fico}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-label">Int. Rate</div>
                    <div class="stat-pill-value">{int_rate}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP
        st.markdown("""
        <div class="shap-card">
            <div class="shap-header">Why this prediction?</div>
            <div class="shap-sub">SHAP contributions to the raw model score (log-odds),
            before calibration — directionally consistent with the probability above,
            but not on the same scale.</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Computing explanations..."):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # For binary LightGBM, shap may return a list of two arrays (one
            # per class) or a single array. The previous build took index 0 in
            # both cases, which explained the *repayment* class -- every
            # contribution was shown with the wrong sign.
            if isinstance(shap_values, list):
                sv, base = shap_values[1][0], explainer.expected_value[1]
            else:
                sv, base = shap_values[0], explainer.expected_value
            base = float(np.ravel(base)[0])

            # ── Custom horizontal bar chart (replaces SHAP waterfall) ──
            feature_names = input_df.columns.tolist()
            contributions = sv

            # Sort by absolute value, take top 12
            indices = np.argsort(np.abs(contributions))[::-1][:12]
            top_names = [feature_names[i] for i in indices]
            top_vals  = [contributions[i] for i in indices]

            # Reverse for bottom-to-top display
            top_names = top_names[::-1]
            top_vals  = top_vals[::-1]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')

            colors = ['#2dd4bf' if v > 0 else '#fb923c' for v in top_vals]
            bars = ax.barh(range(len(top_names)), top_vals, color=colors,
                           height=0.6, edgecolor='none', alpha=0.9)

            ax.set_yticks(range(len(top_names)))
            ax.set_yticklabels(top_names, fontsize=9, color='#94a3b8',
                               fontfamily='Inter')
            ax.set_xlabel('SHAP value (impact on log-odds)', fontsize=9,
                          color='#64748b', fontfamily='Inter', labelpad=10)
            ax.tick_params(axis='x', colors='#475569', labelsize=8)

            # Add value labels
            for bar, val in zip(bars, top_vals):
                x_pos = bar.get_width()
                ha = 'left' if val >= 0 else 'right'
                offset = 0.01 if val >= 0 else -0.01
                ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                        f'{val:+.3f}', va='center', ha=ha,
                        fontsize=7.5, color='#94a3b8', fontfamily='Inter')

            ax.axvline(x=0, color='#334155', linewidth=0.8, linestyle='-')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#1e293b')
            ax.spines['left'].set_color('#1e293b')

            # Title
            ax.set_title(f'f(x) = {base + sum(contributions):.3f}   (base = {base:.3f})',
                         fontsize=9, color='#64748b', fontfamily='Inter',
                         pad=12, loc='left')

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2dd4bf', label='Increases risk'),
                Patch(facecolor='#fb923c', label='Decreases risk')
            ]
            leg = ax.legend(handles=legend_elements, loc='lower right',
                           fontsize=7.5, frameon=True, facecolor='#111827',
                           edgecolor='#1e293b', labelcolor='#94a3b8')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
