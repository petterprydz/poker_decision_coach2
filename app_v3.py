import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from poker_engine import (
    monte_carlo_equity,
    parse_cards,
    best_hand_rank,
    format_cards,
    RANGE_PROFILES,
    get_range_description,
)
from llm_coach import structured_hand_analysis, ChatbotCoach
from card_picker import card_picker_ui


# ---------------------------------------------------------------------------
# Styled table helper
# ---------------------------------------------------------------------------
def _styled_table(rows: list, headers: list) -> str:
    header_html = "".join(
        f'<th style="padding:10px 14px;text-align:left;color:#c7d2fe;'
        f'font-weight:700;font-size:0.85rem;letter-spacing:0.06em;'
        f'text-transform:uppercase;border-bottom:2px solid #4f46e5">'
        f'{h}</th>' for h in headers
    )
    row_html = ""
    for i, row in enumerate(rows):
        bg = "#1e293b" if i % 2 == 0 else "#162032"
        cells = "".join(
            f'<td style="padding:9px 14px;color:#f0f0f0;font-weight:500;'
            f'font-size:0.9rem;border-bottom:1px solid #334155">{v}</td>'
            for v in row.values()
        )
        row_html += f'<tr style="background:{bg}">{cells}</tr>'
    return (
        '<div style="border-radius:10px;overflow:hidden;'
        'box-shadow:0 4px 15px rgba(0,0,0,0.4);margin-bottom:14px">'
        '<table style="width:100%;border-collapse:collapse;'
        'background:#1e293b;font-family:Inter,sans-serif">'
        f'<thead><tr style="background:#1e1b4b">{header_html}</tr></thead>'
        f'<tbody>{row_html}</tbody>'
        '</table></div>'
    )


# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Poker Decision Coach v2", layout="wide", page_icon="🃏")

st.markdown("""
<style>
  /* ── Base ── */
  .stApp { background-color: #1a1f2e; }
  h1,h2,h3 { color:#f0f0f0 !important; font-family: Inter, sans-serif; font-weight:700; }
  p, span, label, div { color:#e2e8f0 !important; font-family: Inter, sans-serif; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background-color: #0f1523 !important; }
  [data-testid="stSidebar"] * { color:#f0f0f0 !important; font-weight:500; }

  /* ── All text inputs ── */
  input[type="text"], input[type="password"], input[type="number"], textarea {
    color: #f0f0f0 !important;
    font-weight: 500 !important;
    background-color: #1e293b !important;
    border: 1.5px solid #334155 !important;
  }
  input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
  }
  [data-testid="stTextInput"] label,
  [data-testid="stNumberInput"] label,
  [data-testid="stSelectbox"] label,
  [data-testid="stFileUploader"] label {
    color: #f0f0f0 !important;
    font-weight: 600 !important;
  }

  /* ── Selectbox / dropdown ── */
  [data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #f0f0f0 !important;
    font-weight: 500 !important;
  }
  [data-baseweb="select"] * { color: #f0f0f0 !important; font-weight:500 !important; }
  div[data-baseweb="popover"] * { color: #111 !important; font-weight:600 !important; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] * { color: #f0f0f0 !important; }
  [data-testid="stFileUploaderDropzone"] { border-color: #475569 !important; }

  /* ── Buttons ── */
  button[kind="secondary"] {
    color: #f0f0f0 !important;
    font-weight: 600 !important;
    background-color: #1e293b !important;
    border: 1.5px solid #475569 !important;
  }
  button[kind="secondary"]:hover {
    border-color: #6366f1 !important;
    color: #fff !important;
  }
  button[kind="primary"] {
    color: #fff !important;
    font-weight: 700 !important;
  }

  /* ── Radio buttons ── */
  [data-testid="stRadio"] label { color: #f0f0f0 !important; font-weight:500 !important; }
  [data-testid="stRadio"] span  { color: #f0f0f0 !important; }

  /* ── Captions ── */
  [data-testid="stCaptionContainer"] p { color: #94a3b8 !important; font-weight:400 !important; }

  /* ── Metrics ── */
  div[data-testid="stMetricValue"]  { color:#818cf8 !important; font-weight:800 !important; }
  div[data-testid="stMetricLabel"]  { color:#cbd5e1 !important; font-weight:600 !important; }
  div[data-testid="stMetricDelta"]  { font-weight:600 !important; }

  /* ── Tab labels ── */
  button[data-baseweb="tab"] { color: #cbd5e1 !important; font-weight:600 !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #f0f0f0 !important; }

  /* ── Coach cards ── */
  .coach-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-left: 5px solid #6366f1;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    width: 100%;
    box-sizing: border-box;
  }
  .coach-card * { color:#f0f0f0 !important; font-weight:500; }
  .coach-card b { font-weight:700 !important; color:#fff !important; }
  .mistake-card {
    background: linear-gradient(135deg, #2d1515, #1a0a0a);
    border-left: 5px solid #ef4444;
  }
  .correct-card {
    background: linear-gradient(135deg, #0f2d1a, #0a1a0f);
    border-left: 5px solid #22c55e;
  }

  /* ── Pills ── */
  .pill {
    display:inline-block; padding:5px 12px; border-radius:999px;
    font-weight:700; font-size:0.85em; margin:2px 2px 4px 0;
  }
  .pill-green  { background:#166534; color:#86efac !important; }
  .pill-red    { background:#7f1d1d; color:#fca5a5 !important; }
  .pill-blue   { background:#1e3a5f; color:#93c5fd !important; }
  .pill-orange { background:#7c2d12; color:#fdba74 !important; }
  .pill-purple { background:#3b0764; color:#d8b4fe !important; }

  /* ── Severity ── */
  .severity-major    { color: #ef4444 !important; font-weight:700; }
  .severity-moderate { color: #f97316 !important; font-weight:700; }
  .severity-minor    { color: #eab308 !important; font-weight:700; }
  .severity-none     { color: #22c55e !important; font-weight:700; }

  /* ── Chat bubbles ── */
  .chat-user { background:#1e3a5f; border-radius:12px 12px 4px 12px; padding:10px 14px; margin:6px 0; }
  .chat-user * { color:#f0f0f0 !important; font-weight:500; }
  .chat-bot  { background:#1e293b; border-radius:12px 12px 12px 4px; padding:10px 14px; margin:6px 0; }
  .chat-bot * { color:#e2e8f0 !important; font-weight:400; }

  /* ── All buttons text ── */
  [data-testid="stButton"] button {
    color: #f0f0f0 !important;
    font-weight: 600 !important;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def reset_session():
    saved_key = st.session_state.get("_api_key_saved", "")
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    if saved_key:
        st.session_state["_api_key_saved"] = saved_key
    st.rerun()


if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = None
if "chat_history_display" not in st.session_state:
    st.session_state["chat_history_display"] = []
if "llm_analyses" not in st.session_state:
    st.session_state["llm_analyses"] = {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_default_df() -> pd.DataFrame:
    candidates = ["demo_spots.csv", os.path.join("data", "demo_spots.csv")]
    for path in candidates:
        if os.path.exists(path):
            st.session_state["data_source"] = f"Demo file: {path}"
            return pd.read_csv(path)
    st.session_state["data_source"] = "Built-in demo (5 hands)"
    data = {
        "spot_id": range(1, 6),
        "hero_hand": ["QhJh", "8h7h", "AdJd", "Ts9s", "AsQs"],
        "board":     ["Ts9d2c7h", "9h6d2cKs", "KhTd4s8c", "Js8d3c6h", "QhTd5c2d"],
        "pot":       [100, 90, 120, 80, 110],
        "bet":       [15, 20, 30, 35, 55],
        "action":    ["call", "call", "call", "call", "call"],
    }
    return pd.DataFrame(data)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"hero_hand", "board", "pot", "bet", "action"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(list(missing))}")
    if "spot_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["spot_id"] = df.index + 1
    df["hero_hand"] = df["hero_hand"].astype(str)
    df["board"]     = df["board"].astype(str)
    df["action"]    = df["action"].astype(str).str.lower()
    df["pot"]       = pd.to_numeric(df["pot"], errors="coerce")
    df["bet"]       = pd.to_numeric(df["bet"], errors="coerce")
    if df["pot"].isna().any() or df["bet"].isna().any():
        raise ValueError("pot/bet must be numeric for all rows.")
    if not df["action"].isin(["call", "fold"]).all():
        raise ValueError("action must be 'call' or 'fold'.")
    return df


def _infer_street(board: str) -> str:
    n = len(str(board).strip()) // 2
    if n <= 3: return "Flop"
    if n == 4: return "Turn"
    return "River"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🃏 Settings")
    st.markdown("---")

    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Anthropic API Key", type="password",
                                 help="Required for LLM features (hand analysis + chatbot)")
    
    if api_key:
        st.session_state["_api_key_saved"] = api_key
        st.success("API key set ✓")
        if st.session_state.get("_last_api_key") != api_key:
            st.session_state["chatbot"] = ChatbotCoach(api_key)
            st.session_state["_last_api_key"] = api_key

    st.markdown("---")
    st.subheader("Opponent Range")
    opp_range = st.selectbox(
        "Villain range profile",
        options=list(RANGE_PROFILES.keys()),
        index=2,
        help="Affects equity calculation — more realistic than random opponent"
    )
    st.caption(f"📋 {get_range_description(opp_range)}")

    st.markdown("---")
    st.subheader("Equity Precision")
    mode = st.radio("Monte Carlo Iterations", ["Fast", "Balanced", "High"], index=1, horizontal=True)
    iters = {"Fast": 1200, "Balanced": 3000, "High": 8000}[mode]
    st.caption(f"Iterations: {iters:,}")

    st.markdown("---")
    st.subheader("Data Source")
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                  help="Columns: hero_hand, board, pot, bet, action")

    st.markdown("---")
    if st.button("🔄 Reset Session"):
        reset_session()


# ---------------------------------------------------------------------------
# Load & compute
# ---------------------------------------------------------------------------
try:
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.session_state["data_source"] = "Uploaded CSV"
    else:
        df_raw = load_default_df()

    if "manual_hands" in st.session_state and len(st.session_state["manual_hands"]) > 0:
        df_raw = pd.concat([df_raw, st.session_state["manual_hands"]], ignore_index=True)
        df_raw = df_raw.reset_index(drop=True)
        df_raw["spot_id"] = df_raw.index + 1

    df_raw = ensure_columns(df_raw)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()


@st.cache_data(show_spinner=False)
def compute_metrics(df_in: pd.DataFrame, iters: int, opp_range: str) -> pd.DataFrame:
    df = df_in.copy()
    equities, made_names, errors, streets = [], [], [], []

    for _, r in df.iterrows():
        try:
            eq = monte_carlo_equity(r["hero_hand"], r["board"], iters=iters, seed=42,
                                    opp_range_profile=opp_range)
            hero  = parse_cards(r["hero_hand"])
            board = parse_cards(r["board"])
            _, _, name = best_hand_rank(hero + board)
            equities.append(eq)
            made_names.append(name)
            errors.append("")
        except Exception as e:
            equities.append(np.nan)
            made_names.append("Invalid")
            errors.append(str(e))
        streets.append(_infer_street(r["board"]))

    df["equity"]    = equities
    df["made_hand"] = made_names
    df["error"]     = errors
    df["street"]    = streets
    df["pot_odds"]  = df["bet"] / (df["pot"] + df["bet"])
    df["ev_call"]   = (df["equity"] * (df["pot"] + df["bet"])) - ((1 - df["equity"]) * df["bet"])
    df.loc[df["equity"].isna(), ["pot_odds", "ev_call"]] = np.nan
    df["ev_realized"]     = np.where(df["action"] == "call", df["ev_call"], 0.0)
    df["model_reco"]      = np.where(df["ev_call"] >= 0, "call", "fold")
    df.loc[df["ev_call"].isna(), "model_reco"] = "n/a"
    df["decision_correct"] = False
    df.loc[(df["action"] == "call") & (df["ev_call"] >= 0), "decision_correct"] = True
    df.loc[(df["action"] == "fold") & (df["ev_call"] < 0), "decision_correct"] = True
    df.loc[df["ev_call"].isna(), "decision_correct"] = False
    df["margin"] = df["equity"] - df["pot_odds"]
    df.loc[df["equity"].isna(), "margin"] = np.nan
    return df


with st.spinner("Computing range-aware equity..."):
    df = compute_metrics(df_raw, iters, opp_range)

valid_df = df[df["error"] == ""].copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🃏 Poker Decision Coach")
st.caption(f"Range-aware equity · LLM hand analysis · AI coaching chatbot  |  Villain: **{opp_range}** — {get_range_description(opp_range)}")
st.write("---")

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🃏 New Hand",
    "📊 Session Overview",
    "🔎 Spot Review + LLM Analysis",
    "📈 Leak Detection",
    "🤖 AI Coaching Chat",
])


# ===========================================================================
# TAB 0: New Hand — Interactive Card Picker
# ===========================================================================
with tab0:
    st.subheader("Enter a New Hand")
    st.caption("Pick cards below → fill in the details → click Add Hand.")

    col_picker, col_settings = st.columns([3, 1])

    with col_picker:
        nh_hero, nh_board = card_picker_ui(key="newhnd_picker")

    with col_settings:
        st.markdown("#### Hand Details")

        if nh_hero or nh_board:
            st.success(
                f"**Cards confirmed ✓**\n\n"
                f"Hero: `{format_cards(nh_hero) if nh_hero else '—'}`\n\n"
                f"Board: `{format_cards(nh_board) if nh_board else '—'}`"
            )
        else:
            st.info("No cards selected yet. Use the picker on the left.")

        nh_pot = st.number_input("Pot Size ($)", min_value=1, value=100, key="nh_pot")
        nh_bet = st.number_input("Bet You Face ($)", min_value=1, value=30, key="nh_bet")
        nh_act = st.selectbox("Your Action", ["call", "fold"], index=0, key="nh_act")

        if nh_hero and nh_board:
            try:
                pot_odds = nh_bet / (nh_pot + nh_bet)
                st.markdown(f"**Pot odds required:** `{pot_odds:.1%}`")
                st.caption("You need this much equity to break even on a call.")
            except Exception:
                pass

        can_submit = len(nh_hero) == 4 and len(nh_board) >= 6
        submitted = st.button(
            "Add Hand ✓" if can_submit else "Add Hand (select cards first)",
            type="primary" if can_submit else "secondary",
            disabled=not can_submit,
            use_container_width=True,
        )

    if submitted and can_submit:
        new_row = pd.DataFrame([{
            "hero_hand": nh_hero,
            "board": nh_board,
            "pot": nh_pot,
            "bet": nh_bet,
            "action": nh_act,
        }])
        if "manual_hands" not in st.session_state:
            st.session_state["manual_hands"] = new_row
        else:
            st.session_state["manual_hands"] = pd.concat(
                [st.session_state["manual_hands"], new_row], ignore_index=True)
        st.success(f"Hand added — {format_cards(nh_hero)} on {format_cards(nh_board)}. Switch to Session Overview or Spot Review to analyse.")
        st.rerun()


# ===========================================================================
# TAB 1: Session Overview
# ===========================================================================
with tab1:
    total   = len(df)
    valid_n = len(valid_df)
    correct = int(valid_df["decision_correct"].sum())
    quality = correct / valid_n * 100 if valid_n else 0
    net_ev  = float(valid_df["ev_realized"].sum(skipna=True))
    neg_calls = int(((valid_df["action"] == "call") & (valid_df["ev_call"] < 0)).sum())
    pos_folds = int(((valid_df["action"] == "fold") & (valid_df["ev_call"] > 0)).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Hands", total)
    c2.metric("Decision Quality", f"{quality:.0f}%")
    c3.metric("Net Session EV", f"${net_ev:.2f}")
    c4.metric("−EV Calls", neg_calls, delta=f"-{neg_calls} leaks" if neg_calls else "Clean", delta_color="inverse")
    c5.metric("+EV Folds", pos_folds, delta=f"-{pos_folds} missed" if pos_folds else "None missed", delta_color="inverse")

    st.markdown(f"**Data:** {st.session_state.get('data_source','')}  |  **Villain range:** {opp_range}  |  **MC iters:** {iters:,}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### EV per Hand")
        plot_df = valid_df.copy()
        plot_df["sign"] = np.where(plot_df["ev_call"] >= 0, "Positive EV", "Negative EV")
        fig = px.bar(plot_df, x="spot_id", y="ev_call", color="sign",
                     color_discrete_map={"Positive EV": "#22c55e", "Negative EV": "#ef4444"},
                     hover_data=["hero_hand", "board", "action", "equity", "pot_odds"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), xaxis_title="Hand ID",
            yaxis_title="EV if Call ($)", legend_title_text=""
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Equity vs Pot Odds")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=valid_df["spot_id"], y=valid_df["equity"] * 100,
            mode="lines+markers", name="Equity %",
            line=dict(color="#090df5", width=2),
            
            marker=dict(size=7)
        ))
        fig2.add_trace(go.Scatter(
            x=valid_df["spot_id"], y=valid_df["pot_odds"] * 100,
            mode="lines+markers", name="Pot Odds Required %",
            line=dict(color="#f36804", width=2, dash="dash"),
            marker=dict(size=7)
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"), xaxis_title="Hand ID", yaxis_title="%",
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig2, use_container_width=True)

    if "street" in valid_df.columns:
        st.markdown("### Decision Accuracy by Street")
        street_stats = valid_df.groupby("street").agg(
            hands=("spot_id", "count"),
            accuracy=("decision_correct", lambda x: round(x.mean() * 100, 1)),
            avg_ev=("ev_call", lambda x: round(x.mean(), 2)),
        ).reset_index()
        rows = [
            {
                "Street": r["street"].title(),
                "Hands": int(r["hands"]),
                "Accuracy (%)": f'{r["accuracy"]}%',
                "Average EV ($)": f'${r["avg_ev"]:.2f}'
            }
            for _, r in street_stats.iterrows()
        ]
        st.markdown(
            _styled_table(rows, ["Street", "Hands", "Accuracy (%)", "Average EV ($)"]),
            unsafe_allow_html=True
        )

    with st.expander("📋 Full Session Table"):
        show_cols = ["spot_id", "hero_hand", "board", "street", "pot", "bet", "action",
                     "equity", "pot_odds", "ev_call", "model_reco", "decision_correct"]
        show_cols = [c for c in show_cols if c in valid_df.columns]
        tmp = valid_df[show_cols].copy()
        tmp["hero_hand"] = tmp["hero_hand"].apply(format_cards)
        tmp["board"]     = tmp["board"].apply(format_cards)
        st.dataframe(tmp, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2: Spot Review + LLM Analysis
# ===========================================================================
with tab2:
    sid = st.select_slider("Select hand:", options=df["spot_id"].tolist(),
                            value=int(df["spot_id"].min()))
    h = df[df["spot_id"] == sid].iloc[0]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader(f"Hand #{int(sid)}  —  {h['street']}")

        hero_pretty  = format_cards(h["hero_hand"]) if h["error"] == "" else h["hero_hand"]
        board_pretty = format_cards(h["board"])     if h["error"] == "" else h["board"]

        st.write(f"**Hero:** {hero_pretty}  |  **Board:** {board_pretty}")
        st.write(f"**Made hand:** `{h['made_hand']}`")

        if h["error"]:
            st.error(h["error"])
        else:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Equity", f"{h['equity']:.1%}", help=f"vs {opp_range} range")
            mc2.metric("Pot Odds Required", f"{h['pot_odds']:.1%}")
            mc3.metric("EV (Call)", f"${h['ev_call']:.2f}")

            st.markdown(f"**Model Recommendation:** `{str(h['model_reco']).upper()}`  |  **Your Action:** `{str(h['action']).upper()}`")

            if h["decision_correct"]:
                st.success("✅ Your decision matches the EV model.")
            else:
                st.error("❌ Your decision does not match the EV model.")

    with col2:
        st.info(f"**Context**\n\nPot: ${float(h['pot']):.0f}  |  Bet: ${float(h['bet']):.0f}\nVillain Range: {opp_range}")

        if not pd.isna(h["equity"]):
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(h["equity"]) * 100,
                number={"suffix": "%", "font": {"color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#e2e8f0"},
                    "bar": {"color": "#6366f1"},
                    "threshold": {
                        "line": {"color": "#f97316", "width": 3},
                        "thickness": 0.8,
                        "value": float(h["pot_odds"]) * 100,
                    },
                    "bgcolor": "#1e293b",
                    "steps": [
                        {"range": [0, float(h["pot_odds"]) * 100], "color": "#7f1d1d"},
                        {"range": [float(h["pot_odds"]) * 100, 100], "color": "#166534"},
                    ],
                },
                title={"text": "Equity (orange = breakeven)", "font": {"color": "#e2e8f0"}},
            ))
            fig_g.update_layout(
                height=200, margin=dict(t=40, b=0, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 LLM Hand Analysis")

    cache_key = f"analysis_{sid}_{opp_range}"

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to unlock LLM hand analysis.")
    elif h["error"] != "":
        st.error("Cannot analyse an invalid hand.")
    else:
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            run_analysis = st.button("🔍 Analyse This Hand", key=f"btn_{sid}")
        with col_btn2:
            if cache_key in st.session_state["llm_analyses"]:
                st.caption("✓ Analysis cached — click to refresh")

        if run_analysis:
            ctx = {
                "hero_hand": h["hero_hand"],
                "hero_hand_pretty": hero_pretty,
                "board": h["board"],
                "board_pretty": board_pretty,
                "made_hand": h["made_hand"],
                "pot": float(h["pot"]),
                "bet": float(h["bet"]),
                "pot_odds": float(h["pot_odds"]),
                "equity": float(h["equity"]),
                "ev_call": float(h["ev_call"]),
                "action": h["action"],
                "model_reco": h["model_reco"],
                "decision_correct": bool(h["decision_correct"]),
                "opp_range": opp_range,
                "opp_range_desc": get_range_description(opp_range),
            }
            with st.spinner("Analysing hand..."):
                analysis = structured_hand_analysis(ctx, api_key)
            st.session_state["llm_analyses"][cache_key] = analysis

        if cache_key in st.session_state["llm_analyses"]:
            a = st.session_state["llm_analyses"][cache_key]

            sev = a.get("mistake_severity", "none").lower()
            card_class = "mistake-card" if sev in ("major", "moderate") else \
                         "correct-card" if sev == "none" else "coach-card"

            st.markdown(f"""
            <div class="coach-card {card_class}">
              <b>📍 Situation</b><br>{a.get('situation_summary','')}<br><br>
              <span class="pill pill-purple">Street: {a.get('street','').title()}</span>
              <span class="pill pill-blue">Hand Strength: {a.get('hero_hand_strength','').replace('_',' ').title()}</span>
              <span class="pill {'pill-red' if sev in ('major','moderate') else 'pill-green'}">
                Mistake: {sev.title()}</span>
            </div>
            """, unsafe_allow_html=True)

            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown(f"""
                <div class="coach-card" style="height:100%">
                  <b>🎯 Opponent Range Analysis</b><br><br>
                  {a.get('opponent_range_analysis','')}
                </div>
                """, unsafe_allow_html=True)

            with col_r:
                rec_color = "pill-green" if a.get("recommended_action") == "call" else "pill-red"
                st.markdown(f"""
                <div class="coach-card" style="height:100%">
                  <b>✅ Recommended Action</b>&nbsp;
                  <span class="pill {rec_color}" style="font-size:1em">
                    {a.get('recommended_action','').upper()}</span><br><br>
                  {a.get('recommended_action_reasoning','')}
                </div>
                """, unsafe_allow_html=True)

            col_l2, col_r2 = st.columns(2)

            with col_l2:
                eq_assess = a.get("equity_assessment", "")
                eq_color  = "pill-orange" if eq_assess in ("overestimated","underestimated") else "pill-green"
                st.markdown(f"""
                <div class="coach-card" style="height:100%">
                  <b>📊 Equity Assessment</b>&nbsp;
                  <span class="pill {eq_color}">{eq_assess.title()}</span><br><br>
                  {a.get('equity_comment','')}
                </div>
                """, unsafe_allow_html=True)

            with col_r2:
                if sev not in ("none", ""):
                    st.markdown(f"""
                    <div class="coach-card mistake-card" style="height:100%">
                      <b>⚠️ Mistake Analysis</b><br><br>
                      Type: <b>{a.get('mistake_type','').replace('_',' ').title()}</b>
                      — <span class="severity-{sev}">{sev.title()}</span><br><br>
                      {a.get('mistake_explanation','')}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="coach-card correct-card" style="height:100%">
                      <b>✅ Decision Quality</b><br><br>
                      No mistake detected. Decision aligns with EV model.
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="coach-card">
              <b>💡 Key Concept:</b>&nbsp;{a.get('key_concept','')}<br><br>
              <b>🏋️ Drill Suggestion:</b>&nbsp;{a.get('drill_suggestion','')}
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# TAB 3: Leak Detection
# ===========================================================================
with tab3:
    st.subheader("Session Leak Detection")

    neg_ev_calls = valid_df[(valid_df["action"] == "call") & (valid_df["ev_call"] < 0)].copy()
    pos_ev_folds = valid_df[(valid_df["action"] == "fold") & (valid_df["ev_call"] > 0)].copy()

    leak_ev   = float(neg_ev_calls["ev_call"].sum()) if len(neg_ev_calls) else 0.0
    missed_ev = float(pos_ev_folds["ev_call"].sum()) if len(pos_ev_folds) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Negative EV Calls",    len(neg_ev_calls))
    c2.metric("EV Lost (Bad Calls)",  f"${abs(leak_ev):.2f}")
    c3.metric("Positive EV Folds",    len(pos_ev_folds))
    c4.metric("Missed Value",         f"${missed_ev:.2f}")

    col_leak, col_fold = st.columns(2)

    with col_leak:
        st.markdown("#### 🔴 Negative EV Calls (Leaks)")
        if len(neg_ev_calls) > 0:
            show = neg_ev_calls.sort_values("ev_call")[
                ["spot_id", "hero_hand", "board", "street", "equity", "pot_odds", "ev_call"]
            ].copy()
            show["hero_hand"] = show["hero_hand"].apply(format_cards)
            show["board"]     = show["board"].apply(format_cards)
            show["equity"]    = show["equity"].map("{:.1%}".format)
            show["pot_odds"]  = show["pot_odds"].map("{:.1%}".format)
            show["ev_call"]   = show["ev_call"].map("${:.2f}".format)
            rows = [
                {
                    "Hand": int(r["spot_id"]),
                    "Hero": r["hero_hand"],
                    "Board": r["board"],
                    "Street": r["street"].title(),
                    "Equity": r["equity"],
                    "Pot Odds": r["pot_odds"],
                    "EV (Call)": r["ev_call"],
                }
                for _, r in show.iterrows()
            ]
            st.markdown(
                _styled_table(rows, ["Hand", "Hero", "Board", "Street", "Equity", "Pot Odds", "EV (Call)"]),
                unsafe_allow_html=True
            )
            fig_leak = px.bar(
                neg_ev_calls.sort_values("ev_call"), x="spot_id", y="ev_call",
                color_discrete_sequence=["#ef4444"],
                labels={"ev_call": "EV Lost ($)", "spot_id": "Hand"}
            )
            fig_leak.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig_leak, use_container_width=True)
        else:
            st.success("No negative EV calls — great discipline!")

    with col_fold:
        st.markdown("#### 🟡 Positive EV Folds (Missed Value)")
        if len(pos_ev_folds) > 0:
            show2 = pos_ev_folds.sort_values("ev_call", ascending=False)[
                ["spot_id", "hero_hand", "board", "street", "equity", "pot_odds", "ev_call"]
            ].copy()
            show2["hero_hand"] = show2["hero_hand"].apply(format_cards)
            show2["board"]     = show2["board"].apply(format_cards)
            show2["equity"]    = show2["equity"].map("{:.1%}".format)
            show2["pot_odds"]  = show2["pot_odds"].map("{:.1%}".format)
            show2["ev_call"]   = show2["ev_call"].map("${:.2f}".format)
            rows2 = [
                {
                    "Hand": int(r["spot_id"]),
                    "Hero": r["hero_hand"],
                    "Board": r["board"],
                    "Street": r["street"].title(),
                    "Equity": r["equity"],
                    "Pot Odds": r["pot_odds"],
                    "EV (Call)": r["ev_call"],
                }
                for _, r in show2.iterrows()
            ]
            st.markdown(
                _styled_table(rows2, ["Hand", "Hero", "Board", "Street", "Equity", "Pot Odds", "EV (Call)"]),
                unsafe_allow_html=True
            )
        else:
            st.success("No missed positive EV calls — you are not folding good spots!")

    st.markdown("#### Decision Margin: Equity vs Pot Odds")
    fig_sc = px.scatter(
        valid_df, x="pot_odds", y="equity",
        color="decision_correct",
        color_discrete_map={True: "#22c55e", False: "#ef4444"},
        hover_data=["spot_id", "hero_hand", "action", "ev_call"],
        labels={"pot_odds": "Pot Odds Required", "equity": "Equity vs Range",
                "decision_correct": "Correct"},
        size_max=12,
    )
    line_x = [0, 1]
    fig_sc.add_trace(go.Scatter(x=line_x, y=line_x, mode="lines",
                                line=dict(color="#6366f1", dash="dash"), name="Breakeven"))
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), xaxis_tickformat=".0%", yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption("Points above the line = profitable calls. Points below = should fold. Color = your actual decision quality.")


# ===========================================================================
# TAB 4: AI Coaching Chatbot
# ===========================================================================
with tab4:
    st.subheader("🤖 AI Coaching Chat")
    st.caption("Ask me anything about your session — I will look up the real data to answer.")

    if not api_key:
        st.warning("⚠️ Enter your Anthropic API key in the sidebar to use the coaching chatbot.")
        st.stop()

    if st.session_state["chatbot"] is None:
        st.session_state["chatbot"] = ChatbotCoach(api_key)

    st.markdown("**Quick Questions:**")
    qcols = st.columns(3)
    suggestions = [
        "What's my biggest leak this session?",
        "Which hand was my worst mistake?",
        "How is my accuracy on the river vs flop?",
        "Am I over-folding or over-calling?",
        "Walk me through hand #1",
        "How much EV did I lose from bad calls?",
    ]
    for i, q in enumerate(suggestions):
        with qcols[i % 3]:
            if st.button(q, key=f"sugg_{i}"):
                st.session_state["pending_question"] = q

    st.markdown("---")

    for msg in st.session_state["chat_history_display"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["text"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask your coach...")

    question = None
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
    elif user_input:
        question = user_input

    if question:
        st.session_state["chat_history_display"].append({"role": "user", "text": question})
        with st.spinner("Coach is thinking..."):
            response = st.session_state["chatbot"].chat(question, df)
        st.session_state["chat_history_display"].append({"role": "bot", "text": response})
        st.rerun()

    col_reset = st.columns([1, 4])[0]
    with col_reset:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_history_display"] = []
            if st.session_state["chatbot"]:
                st.session_state["chatbot"].reset()
            st.rerun()
