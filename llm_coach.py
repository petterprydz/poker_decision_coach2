"""
llm_coach.py
------------
Two non-straightforward LLM features using Anthropic Claude:

1. structured_hand_analysis(hand_context) -> dict
   Carefully engineered system prompt that returns structured JSON
   with range analysis, recommended line, mistake classification, and
   street-by-street reasoning. Python post-processes the JSON to render
   rich coaching cards.

2. ChatbotCoach class
   Multi-turn chatbot with tool use. LLM decides which tools to call
   (get_session_summary, get_hand_details, get_worst_mistakes, etc.),
   we execute them against live session data, feed results back, and
   Claude synthesises a coaching answer.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional
import anthropic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def _safe_json(text: str) -> Optional[dict]:
    """Extract first JSON object from LLM text, even if wrapped in markdown."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _infer_street(board: str) -> str:
    n = len(board.strip()) // 2
    if n == 0: return "preflop"
    if n == 3: return "flop"
    if n == 4: return "turn"
    return "river"


# ---------------------------------------------------------------------------
# Feature 1: Structured Hand Analysis
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """You are an expert No-Limit Texas Hold'em cash game coach with deep knowledge of GTO strategy, range analysis, and exploitative play.

You will receive a poker hand situation and must respond ONLY with a valid JSON object — no preamble, no markdown, no explanation outside the JSON.

The JSON must follow this exact schema:
{
  "street": "flop|turn|river|preflop",
  "situation_summary": "1-2 sentence description of the spot",
  "hero_hand_strength": "nuts|strong|medium|marginal|bluff_catcher|air",
  "opponent_range_analysis": "2-3 sentences on what villain's range likely contains given the profile and board",
  "equity_assessment": "realistic|overestimated|underestimated",
  "equity_comment": "1 sentence on why the raw equity vs random was misleading (if applicable)",
  "recommended_action": "fold|call|raise|check|bet",
  "recommended_action_reasoning": "2-3 sentences explaining the GTO/exploitative reasoning",
  "mistake_type": "none|loose_call|tight_fold|sizing_error|range_imbalance",
  "mistake_severity": "none|minor|moderate|major",
  "mistake_explanation": "1-2 sentences if a mistake was made, else empty string",
  "key_concept": "The single most important poker concept illustrated by this hand",
  "drill_suggestion": "One specific drill or study task to improve in this spot type"
}"""


def _build_analysis_prompt(ctx: dict) -> str:
    board_cards = ctx.get("board", "")
    street = _infer_street(board_cards)
    return f"""Analyse this NLHE cash game hand:

HERO HAND: {ctx.get('hero_hand_pretty', ctx.get('hero_hand', ''))}
BOARD: {ctx.get('board_pretty', ctx.get('board', ''))}
STREET: {street}
MADE HAND: {ctx.get('made_hand', 'unknown')}
POT SIZE: ${ctx.get('pot', 0)}
BET FACED: ${ctx.get('bet', 0)}
POT ODDS REQUIRED: {ctx.get('pot_odds', 0):.1%}
HERO EQUITY (vs {ctx.get('opp_range', 'Balanced')} range): {ctx.get('equity', 0):.1%}
HERO ACTION: {ctx.get('action', 'unknown').upper()}
EV OF CALLING: ${ctx.get('ev_call', 0):.2f}
OPPONENT RANGE PROFILE: {ctx.get('opp_range', 'Balanced')} — {ctx.get('opp_range_desc', '')}
MODEL RECOMMENDATION: {ctx.get('model_reco', 'unknown').upper()}
DECISION WAS CORRECT: {ctx.get('decision_correct', False)}

Provide your full coaching analysis as a JSON object following the schema exactly."""


def structured_hand_analysis(ctx: dict, api_key: str) -> dict:
    """
    Non-straightforward LLM use:
    - Carefully engineered system prompt to enforce JSON schema
    - Python post-processes the structured output into coaching cards
    - Falls back gracefully if JSON parse fails
    """
    client = _get_client(api_key)
    prompt = _build_analysis_prompt(ctx)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        temperature=0.3,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    result = _safe_json(raw)

    # Retry if JSON parse failed
    if result is None:
        response2 = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            temperature=0.1,
            system=ANALYSIS_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your response was not valid JSON. Reply ONLY with the JSON object, starting with { and ending with }. No other text."},
            ],
        )
        result = _safe_json(response2.content[0].text.strip())

    if result is None:
        result = {
            "situation_summary": "Could not parse LLM response.",
            "opponent_range_analysis": raw[:300],
            "recommended_action": ctx.get("model_reco", "n/a"),
            "recommended_action_reasoning": "See raw output above.",
            "mistake_type": "none",
            "mistake_severity": "none",
            "mistake_explanation": "",
            "key_concept": "n/a",
            "drill_suggestion": "n/a",
            "equity_assessment": "realistic",
            "equity_comment": "",
            "hero_hand_strength": "unknown",
            "street": _infer_street(ctx.get("board", "")),
        }

    return result


# ---------------------------------------------------------------------------
# Feature 2: Coaching Chatbot with Tool Use
# ---------------------------------------------------------------------------

CHATBOT_SYSTEM = """You are an expert No-Limit Texas Hold'em cash game coach.
You have access to tools that let you query the player's current session data.
Always use tools to fetch real data before answering data-specific questions.
Be concise, direct, and tactical. Use poker terminology appropriately.
When you identify leaks, be specific about hands and numbers."""

TOOLS_SCHEMA = [
    {
        "name": "get_session_summary",
        "description": "Returns overall session statistics: total hands, decision quality %, net EV, number of mistakes.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_hand_details",
        "description": "Returns full details for a specific hand by spot_id including hero hand, board, equity, EV, action taken, and whether it was correct.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spot_id": {
                    "type": "integer",
                    "description": "The hand/spot ID number to retrieve",
                }
            },
            "required": ["spot_id"],
        },
    },
    {
        "name": "get_worst_mistakes",
        "description": "Returns the top N biggest EV mistakes in the session (both bad calls and missed value folds).",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of mistakes to return (default 3)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_equity_leaks",
        "description": "Returns all hands where the player called with negative EV, sorted by worst first.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_missed_value",
        "description": "Returns all hands where the player folded a +EV call, sorted by most missed value first.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_hands_by_street",
        "description": "Returns hands grouped by street (flop/turn/river) with accuracy per street.",
        "input_schema": {
            "type": "object",
            "properties": {
                "street": {
                    "type": "string",
                    "description": "Filter to a specific street: flop, turn, river, or all",
                }
            },
            "required": [],
        },
    },
]


def _execute_tool(tool_name: str, params: dict, session_df) -> str:
    """Execute a tool call against the live session DataFrame."""
    import pandas as pd
    import numpy as np

    df = session_df.copy()
    valid = df[df["error"] == ""].copy() if "error" in df.columns else df.copy()

    if tool_name == "get_session_summary":
        total    = len(df)
        valid_n  = len(valid)
        correct  = int(valid["decision_correct"].sum()) if "decision_correct" in valid.columns else 0
        quality  = correct / valid_n * 100 if valid_n else 0
        net_ev   = float(valid["ev_realized"].sum()) if "ev_realized" in valid.columns else 0
        neg_ev_calls = int(((valid["action"] == "call") & (valid["ev_call"] < 0)).sum())
        pos_ev_folds = int(((valid["action"] == "fold") & (valid["ev_call"] > 0)).sum())
        return json.dumps({
            "total_hands": total,
            "valid_hands": valid_n,
            "decision_quality_pct": round(quality, 1),
            "net_ev": round(net_ev, 2),
            "negative_ev_calls": neg_ev_calls,
            "missed_value_folds": pos_ev_folds,
            "total_mistakes": neg_ev_calls + pos_ev_folds,
        })

    elif tool_name == "get_hand_details":
        sid = params.get("spot_id")
        row = df[df["spot_id"] == sid]
        if row.empty:
            return json.dumps({"error": f"Hand {sid} not found"})
        r = row.iloc[0].to_dict()
        return json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in r.items() if k != "error"}, default=str)

    elif tool_name == "get_worst_mistakes":
        n = params.get("n", 3)
        bad_calls = valid[(valid["action"] == "call") & (valid["ev_call"] < 0)].copy()
        bad_calls["mistake_ev"] = bad_calls["ev_call"]
        missed = valid[(valid["action"] == "fold") & (valid["ev_call"] > 0)].copy()
        missed["mistake_ev"] = -missed["ev_call"]
        combined = pd.concat([bad_calls, missed]).sort_values("mistake_ev").head(n)
        cols = ["spot_id", "hero_hand", "board", "action", "ev_call", "equity", "pot_odds"]
        cols = [c for c in cols if c in combined.columns]
        result = combined[cols].to_dict(orient="records")
        return json.dumps([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                            for k, v in row.items()} for row in result], default=str)

    elif tool_name == "get_equity_leaks":
        leaks = valid[(valid["action"] == "call") & (valid["ev_call"] < 0)].sort_values("ev_call")
        cols = ["spot_id", "hero_hand", "board", "equity", "pot_odds", "ev_call"]
        cols = [c for c in cols if c in leaks.columns]
        result = leaks[cols].to_dict(orient="records")
        return json.dumps([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                            for k, v in row.items()} for row in result], default=str)

    elif tool_name == "get_missed_value":
        missed = valid[(valid["action"] == "fold") & (valid["ev_call"] > 0)].sort_values("ev_call", ascending=False)
        cols = ["spot_id", "hero_hand", "board", "equity", "pot_odds", "ev_call"]
        cols = [c for c in cols if c in missed.columns]
        result = missed[cols].to_dict(orient="records")
        return json.dumps([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                            for k, v in row.items()} for row in result], default=str)

    elif tool_name == "get_hands_by_street":
        street_filter = params.get("street", "all")

        def _street(board):
            n = len(str(board).strip()) // 2
            if n <= 3: return "flop"
            if n == 4: return "turn"
            return "river"

        valid["street"] = valid["board"].apply(_street)
        if street_filter != "all" and street_filter in ["flop", "turn", "river"]:
            subset = valid[valid["street"] == street_filter]
        else:
            subset = valid

        summary = subset.groupby("street").apply(
            lambda g: {
                "count": len(g),
                "accuracy_pct": round(g["decision_correct"].mean() * 100, 1),
                "avg_ev_call": round(float(g["ev_call"].mean()), 2),
            }
        ).to_dict()
        return json.dumps(summary, default=str)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


class ChatbotCoach:
    """
    Multi-turn coaching chatbot with tool use.
    Maintains conversation history across turns.
    """

    def __init__(self, api_key: str):
        self.client  = _get_client(api_key)
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        self.history = []

    def chat(self, user_message: str, session_df) -> str:
        """
        Send a message, handle tool calls, return the final response.
        Uses Anthropic's tool_use content blocks for reliable tool calling.
        """
        self.history.append({"role": "user", "content": user_message})

        max_rounds = 5
        for _ in range(max_rounds):

            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                system=CHATBOT_SYSTEM,
                tools=TOOLS_SCHEMA,
                messages=self.history,
            )

            # No tool use — final answer
            if response.stop_reason == "end_turn":
                final_text = response.content[0].text
                self.history.append({"role": "assistant", "content": response.content})
                return final_text

            # Tool use — execute and feed results back
            if response.stop_reason == "tool_use":
                self.history.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        output = _execute_tool(block.name, block.input or {}, session_df)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        })

                self.history.append({"role": "user", "content": tool_results})
                continue

            # Fallback
            break

        return "I wasn't able to complete that analysis. Please try rephrasing your question."

