# 🃏 Poker Decision Coach v2

An AI-powered poker coaching tool built with Streamlit. Upload your hand history, get range-aware equity calculations, LLM hand analysis, leak detection, and an interactive AI coaching chatbot — all in one app.

---

## Features

### 📊 Session Overview
- Key metrics: decision quality %, net session EV, negative EV calls, and missed +EV folds
- EV per hand bar chart (green = positive EV, red = negative EV)
- Equity vs pot odds comparison chart across all hands
- Decision accuracy breakdown by street (Flop / Turn / River)
- Full session table with all calculated metrics

### 🔎 Spot Review + LLM Analysis
- Navigate hand-by-hand using a slider
- Equity gauge showing your equity vs the breakeven threshold
- Model recommendation (call/fold) vs your actual action
- One-click LLM hand analysis powered by Anthropic Claude, returning:
  - Situation summary and hand strength assessment
  - Opponent range analysis
  - Equity assessment (overestimated / underestimated)
  - Recommended action with reasoning
  - Mistake classification and severity (none / minor / moderate / major)
  - Key concept and drill suggestion

### 📈 Leak Detection
- Identifies all negative EV calls and positive EV folds
- Quantifies total EV lost from bad calls and missed value from bad folds
- Bar chart of leak magnitude per hand
- Scatter plot: equity vs pot odds with breakeven line — color coded by decision correctness

### 🤖 AI Coaching Chatbot
- Conversational AI coach with full access to your session data
- Suggested quick questions (biggest leak, worst mistake, street accuracy, etc.)
- Uses tool calling to look up real hand data before answering
- Persistent chat history within the session

### ⚙️ Settings & Data Input
- Upload your own CSV hand history
- Add hands manually from the sidebar
- Choose opponent range profile: Nit, Tight, Balanced, Loose, or Maniac
- Select Monte Carlo precision: Fast (1,200), Balanced (3,000), or High (8,000) iterations

---

## How to Run the App

1. Clone the repository:
```bash
git clone https://github.com/petterprydz/poker_decision_coach2.git
cd poker_decision_coach2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app_v3.py
```

The app will open in your browser at `http://localhost:8501`.

The Anthropic API key is pre-configured. LLM features (hand analysis and chatbot) will work automatically.

---

## Data Format

The app loads `demo_spots.csv` by default. You can upload your own CSV with the following columns:

| Column | Description | Example |
|---|---|---|
| `hero_hand` | Your two hole cards | `QhJh` |
| `board` | Community cards (3–5) | `Ts9d2c7h` |
| `pot` | Pot size in $ | `100` |
| `bet` | Bet you are facing in $ | `30` |
| `action` | Your decision | `call` or `fold` |
