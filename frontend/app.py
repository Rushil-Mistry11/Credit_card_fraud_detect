import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ml.config import DATA_PATH, DISPLAY_NAMES, FEATURES  # noqa: E402

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
RISK_COLORS = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c"}

st.set_page_config(page_title="Fraud Risk Scoring", page_icon="💳", layout="wide")


# ---------- helpers ----------
def api(method, path, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Start it with "
                 "`uvicorn backend.main:app --reload`.")
    except Exception as e:  # noqa: BLE001
        st.error(f"API error: {e}")
    return None


@st.cache_data(show_spinner=False)
def load_samples():
    """All fraud rows + a random slice of legit rows, for the 'load sample' buttons."""
    if not Path(DATA_PATH).exists():
        return None
    df = pd.read_csv(DATA_PATH)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(min(3000, (df["Class"] == 0).sum()), random_state=1)
    return pd.concat([fraud, legit])


def set_sample(kind: int):
    df = load_samples()
    if df is None:
        return
    row = df[df["Class"] == kind].sample(1).iloc[0]
    for f in FEATURES:
        st.session_state[f"in_{f}"] = float(row[f])
    st.session_state["sample_label"] = "Fraud" if kind == 1 else "Legitimate"


def gauge(score: float, level: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": f"Fraud Risk: {level}", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": RISK_COLORS[level]},
            "steps": [{"range": [0, 30], "color": "#d5f5e3"},
                      {"range": [30, 70], "color": "#fcf3cf"},
                      {"range": [70, 100], "color": "#fadbd8"}],
        }))
    fig.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
    return fig


for f in FEATURES:
    st.session_state.setdefault(f"in_{f}", 0.0)

st.title("💳 Real-Time Credit Card Fraud Risk Scoring")
health = api("GET", "/health")
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Real-time Evaluator", "📊 Model Comparison","🗂️ Admin Audit Logs", "💳 Payment Simulator"])

# ---------- TAB 1 ----------
with tab1:
    c_top = st.columns([2, 1, 1])
    algo = c_top[0].selectbox(
        "Algorithm", list(DISPLAY_NAMES), index=3,
        format_func=lambda k: DISPLAY_NAMES[k])
    c_top[1].button("Load legitimate sample", on_click=set_sample, args=(0,),
                    width="stretch", disabled=load_samples() is None)
    c_top[2].button("Load fraud sample", on_click=set_sample, args=(1,),
                    width="stretch", disabled=load_samples() is None)
    if load_samples() is None:
        st.caption("Dataset not found, so sample buttons are disabled. Enter values manually.")
    elif "sample_label" in st.session_state:
        st.caption(f"Loaded a real **{st.session_state['sample_label']}** transaction "
                   "from the dataset (ground truth shown for demo only).")

    left, right = st.columns([1.2, 1])
    with left:
        a, b = st.columns(2)
        a.number_input("Time (seconds)", key="in_Time", step=1.0, format="%.2f")
        b.number_input("Amount ($)", key="in_Amount", step=1.0, format="%.2f")
        with st.expander("PCA features V1 - V28", expanded=False):
            cols = st.columns(4)
            for i in range(1, 29):
                cols[(i - 1) % 4].number_input(f"V{i}", key=f"in_V{i}", step=0.01, format="%.4f")

        if st.button("Evaluate transaction", type="primary", width="stretch"):
            payload = {"algorithm": algo,
                       "features": {f: st.session_state[f"in_{f}"] for f in FEATURES}}
            st.session_state["last_result"] = api("POST", "/predict", json=payload)

    with right:
        res = st.session_state.get("last_result")
        if res:
            st.plotly_chart(gauge(res["risk_score"], res["risk_level"]), width="stretch")
            m1, m2 = st.columns(2)
            m1.metric("Decision", "FLAGGED" if res["is_flagged"] else "APPROVED")
            m2.metric("Log ID", f"#{res['id']}")
            st.caption(f"Scored by {DISPLAY_NAMES[res['algorithm']]}. "
                       "Probabilities from SMOTE-trained models are not calibrated; "
                       "use them as relative risk scores.")
        else:
            st.info("Enter a transaction (or load a sample) and click **Evaluate**.")

# ---------- TAB 2 ----------
with tab2:
    data = api("GET", "/metrics")
    if data:
        models, meta = data["models"], data["meta"]
        st.caption(f"Dataset: {meta['n_rows']:,} transactions, {meta['n_fraud']:,} fraud "
                   f"({meta['fraud_pct']}%). Test set: {meta['test_rows']:,} rows. "
                   f"Threshold: {meta['threshold']}")

        table = pd.DataFrame({
            m["display_name"]: {
                "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
                "Train F1": m["train_f1"], "Train time (s)": m["train_time_sec"]}
            for m in models.values()}).T
        st.subheader("Leaderboard")
        st.dataframe(table.style.format("{:.4f}", subset=table.columns[:-1])
                     .format("{:.1f}", subset=["Train time (s)"])
                     .highlight_max(subset=["F1", "ROC-AUC", "PR-AUC"],props="background-color:#1e6b3c; color:white;"),
                     width="stretch")

        long = table[["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]].reset_index() \
            .melt(id_vars="index", var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(long, x="Metric", y="Score", color="index", barmode="group",
                               labels={"index": "Model"}, height=380),
                        width="stretch")

        c1, c2 = st.columns(2)
        roc = go.Figure()
        for m in models.values():
            roc.add_scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"],
                            mode="lines", name=f"{m['display_name']} ({m['roc_auc']:.3f})")
        roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                        showlegend=False)
        roc.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR", height=380)
        c1.plotly_chart(roc, width="stretch")

        pr = go.Figure()
        for m in models.values():
            pr.add_scatter(x=m["pr_curve"]["recall"], y=m["pr_curve"]["precision"],
                           mode="lines", name=f"{m['display_name']} ({m['pr_auc']:.3f})")
        pr.update_layout(title="Precision-Recall Curves", xaxis_title="Recall",
                         yaxis_title="Precision", height=380)
        c2.plotly_chart(pr, width="stretch")

        st.subheader("Confusion Matrices")
        cols = st.columns(len(models))
        for col, m in zip(cols, models.values()):
            fig = px.imshow(m["confusion_matrix"], text_auto=True, color_continuous_scale="Blues",
                            x=["Legit", "Fraud"], y=["Legit", "Fraud"],
                            labels=dict(x="Predicted", y="Actual"), title=m["display_name"])
            fig.update_layout(coloraxis_showscale=False, height=320, title_font_size=14)
            col.plotly_chart(fig, width="stretch")

        c3, c4 = st.columns(2)
        fi = data["feature_importance"]
        c3.plotly_chart(px.bar(x=list(fi.values())[::-1], y=list(fi.keys())[::-1],
                               orientation="h", title="Feature Importance (Decision Tree)",
                               labels={"x": "Importance", "y": ""}, height=420),
                        width="stretch")
        bv = pd.DataFrame({"Model": [m["display_name"] for m in models.values()],
                           "Train F1": [m["train_f1"] for m in models.values()],
                           "Test F1": [m["f1"] for m in models.values()]}) \
            .melt(id_vars="Model", var_name="Split", value_name="F1")
        c4.plotly_chart(px.bar(bv, x="Model", y="F1", color="Split", barmode="group",
                               title="Bias-Variance: Train vs Test F1", height=420),
                        width="stretch")

# ---------- TAB 3 ----------
with tab3:
    top = st.columns([1, 1, 2])
    risk = top[0].selectbox("Risk filter", ["All", "LOW", "MEDIUM", "HIGH"])
    limit = top[1].number_input("Rows", 10, 1000, 200, step=10)
    top[2].button("Refresh")

    summary = api("GET", "/logs/summary")
    if summary:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total evaluated", summary["total"])
        s2.metric("Flagged as fraud", summary["flagged"])
        s3.metric("High risk", summary["by_risk"].get("HIGH", 0))
        s4.metric("Low risk", summary["by_risk"].get("LOW", 0))

    params = {"limit": int(limit)}
    if risk != "All":
        params["risk"] = risk
    logs = api("GET", "/logs", params=params)
    if logs:
        df = pd.DataFrame(logs)
        df["algorithm"] = df["algorithm"].map(DISPLAY_NAMES)
        df["fraud_probability"] = (df["fraud_probability"] * 100).round(2)
        df = df.rename(columns={"fraud_probability": "risk_%"})
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "audit_logs.csv", "text/csv")
    elif logs == []:
        st.info("No transactions logged yet. Evaluate one in the first tab.")

# ---------- TAB 4 ----------
import re
from datetime import date

PROFILES = {"Regular spending behaviour": 0,
            "Unusual / suspicious behaviour": 1,
            "Borderline behaviour (triggers OTP)": "borderline"}
BORDERLINE_PATH = Path(DATA_PATH).parent / "borderline_samples.csv"
ENGINE_ACTION = {"LOW": "APPROVE",
                 "MEDIUM": "STEP-UP VERIFICATION (OTP)",
                 "HIGH": "DECLINE & FLAG FOR REVIEW"}
DEMO_OTP = "123456"


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) != 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def expiry_ok(text: str) -> bool:
    m = re.fullmatch(r"(0[1-9]|1[0-2])/(\d{2})", text.strip())
    if not m:
        return False
    today = date.today()
    return (2000 + int(m.group(2)), int(m.group(1))) >= (today.year, today.month)


def reset_payment():
    st.session_state["pay_stage"] = "form"
    st.session_state.pop("pay_ctx", None)


def pick_transaction(kind, algo):
    """Return (row, problem). Picks a real dataset row for the chosen behaviour."""
    if kind == "borderline":
        if not BORDERLINE_PATH.exists():
            return None, ("Borderline samples not created yet. Run "
                          "`python -m ml.make_borderline_samples` once, then try again.")
        pool = pd.read_csv(BORDERLINE_PATH)
        pool = pool[pool[f"medium_{algo}"] == 1]
        if pool.empty:
            return None, (f"No borderline transactions exist for {DISPLAY_NAMES[algo]}. "
                          "Choose another risk engine (MLP or SVC usually have some).")
        return pool.sample(1).iloc[0], None
    samples = load_samples()
    if samples is None:
        return None, "Dataset not found, so no transaction can be picked. Check data/creditcard.csv."
    return samples[samples["Class"] == kind].sample(1).iloc[0], None


with tab4:
    st.session_state.setdefault("pay_stage", "form")
    stage = st.session_state["pay_stage"]
    ctx = st.session_state.get("pay_ctx")
    left, right = st.columns([1, 1], gap="large")

    # ---------------- LEFT: what the customer sees ----------------
    with left:
        st.subheader("🔒 Secure Checkout")

        if stage == "form":
            with st.form("pay_form"):
                amount = st.number_input("Amount (€)", min_value=0.01, value=49.99,
                                         step=1.0, format="%.2f", key="pay_amount")
                name = st.text_input("Cardholder name", "Demo Customer", key="pay_name")
                card = st.text_input("Card number (demo card, never stored)",
                                     "4242 4242 4242 4242", max_chars=19, key="pay_card")
                c1, c2 = st.columns(2)
                expiry = c1.text_input("Expiry (MM/YY)", "12/30", max_chars=5, key="pay_expiry")
                cvv = c2.text_input("CVV", "123", type="password", max_chars=4, key="pay_cvv")

                st.markdown("**Demo controls**")
                profile = st.radio("Customer behaviour to simulate", list(PROFILES),
                                   key="pay_profile")
                st.caption("Borderline = a real held-out transaction that the chosen model "
                           "scores as medium risk. It uses that transaction's own amount, "
                           "so the engine reliably asks for an OTP.")
                algo = st.selectbox("Risk engine model", list(DISPLAY_NAMES), index=3,
                                    format_func=lambda k: DISPLAY_NAMES[k], key="pay_algo")
                submitted = st.form_submit_button("Pay now", type="primary", width="stretch")

            if submitted:
                errors = []
                if not name.strip():
                    errors.append("Enter the cardholder name.")
                if not luhn_ok(card):
                    errors.append("Card number is invalid (16 digits, valid checksum).")
                if not expiry_ok(expiry):
                    errors.append("Expiry must be MM/YY and not in the past.")
                if not re.fullmatch(r"\d{3,4}", cvv):
                    errors.append("CVV must be 3 or 4 digits.")
                for e in errors:
                    st.error(e)

                if not errors:
                    kind = PROFILES[profile]
                    row, problem = pick_transaction(kind, algo)
                    if problem:
                        st.error(problem)
                    else:
                        feats = {f: float(row[f]) for f in FEATURES}
                        # Borderline rows keep their own amount (that is what was scored
                        # as medium); other profiles use the amount typed in the form.
                        used_amount = float(row["Amount"]) if kind == "borderline" else float(amount)
                        feats["Amount"] = used_amount
                        res = api("POST", "/predict", json={"algorithm": algo, "features": feats})
                        if res:
                            outcome = {"LOW": "approved", "MEDIUM": "pending",
                                       "HIGH": "declined"}[res["risk_level"]]
                            st.session_state["pay_ctx"] = {
                                "amount": used_amount, "typed_amount": float(amount),
                                "own_amount": kind == "borderline",
                                "last4": "".join(ch for ch in card if ch.isdigit())[-4:],
                                "algo": algo, "profile": profile,
                                "result": res, "outcome": outcome}
                            st.session_state["pay_stage"] = "otp" if outcome == "pending" else "done"
                            st.rerun()

        elif stage == "otp" and ctx:
            st.warning("⚠️ **Extra verification required.** For your security, enter the "
                       "6-digit code we sent to your registered phone.")
            with st.form("otp_form"):
                otp = st.text_input("One-time code", max_chars=6, key="pay_otp")
                verify = st.form_submit_button("Verify & pay", type="primary", width="stretch")
            st.caption(f"Demo code: {DEMO_OTP}")
            if verify:
                ctx["outcome"] = "approved_verified" if otp == DEMO_OTP else "declined_otp"
                st.session_state["pay_stage"] = "done"
                st.rerun()
            st.button("Cancel payment", on_click=reset_payment)

        elif stage == "done" and ctx:
            out, r = ctx["outcome"], ctx["result"]
            if out in ("approved", "approved_verified"):
                st.success("✅ Payment successful")
                with st.container(border=True):
                    st.markdown(f"**Amount:** €{ctx['amount']:,.2f}  \n"
                                f"**Card:** •••• •••• •••• {ctx['last4']}  \n"
                                f"**Reference:** TXN-{r['id']:06d}")
                if out == "approved_verified":
                    st.caption("Approved after one-time-code verification.")
            elif out == "declined":
                st.error("⛔ Payment declined")
                st.write("For your security we couldn't complete this payment. "
                         "Our fraud team has been notified and will review it.")
            else:  # declined_otp
                st.error("⛔ Verification failed. The payment was cancelled.")
            st.button("Make another payment", on_click=reset_payment, type="primary")

    # ---------------- RIGHT: what the bank's risk engine sees ----------------
    with right:
        st.subheader("🛡️ Risk engine view")
        if stage == "form" or not ctx:
            st.info("Customers never see this panel. After you click **Pay now**, it shows "
                    "how the model scored the payment and what action the engine took.")
            st.markdown("**Risk bands**  \n🟢 LOW (< 30%) → approve  \n"
                        "🟡 MEDIUM (30–70%) → ask for OTP  \n🔴 HIGH (≥ 70%) → decline & flag")
        else:
            r = ctx["result"]
            st.plotly_chart(gauge(r["risk_score"], r["risk_level"]), width="stretch")
            st.markdown(f"**Engine action:** {ENGINE_ACTION[r['risk_level']]}  \n"
                        f"**Model:** {DISPLAY_NAMES[ctx['algo']]}  \n"
                        f"**Log ID:** #{r['id']} (see the Audit Logs tab)")
            st.caption(f"Behaviour profile simulated: {ctx['profile']}.")
            if ctx.get("own_amount"):
                st.caption(f"The borderline profile scored the recorded transaction amount "
                           f"(€{ctx['amount']:,.2f}), not the €{ctx['typed_amount']:,.2f} typed.")

    st.caption("Simulation notice: card details are checked locally and are never sent to the "
               "API or stored. The model's 28 behavioural features (V1–V28) are taken from a "
               "real transaction in the dataset that matches the behaviour you pick; for the "
               "regular and suspicious profiles only the amount comes from the form. A real "
               "bank would compute these features from card history.")
