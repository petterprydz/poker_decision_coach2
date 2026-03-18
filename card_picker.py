"""
card_picker.py
--------------
Pure Streamlit card picker — no HTML/JS bridge needed.
Fully reliable communication via session state.

Usage:
    from card_picker import card_picker_ui
    hero, board = card_picker_ui(key="hand_builder")
    # hero  -> "AhKd"
    # board -> "Ts9d2c"
"""

import streamlit as st

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']
SUIT_SYMBOLS = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
SUIT_COLORS  = {'s': '#6366f1', 'h': '#ef4444', 'd': '#f97316', 'c': '#22c55e'}


def format_card(card: str) -> str:
    """e.g. 'Ah' -> 'A♥'"""
    if len(card) < 2:
        return card
    rank = card[0].upper()
    suit = card[1].lower()
    return rank + SUIT_SYMBOLS.get(suit, suit)


def card_picker_ui(key: str = "card_picker") -> tuple[str, str]:
    """
    Renders an interactive card picker using pure Streamlit widgets.
    Returns (hero_hand, board) as compact strings e.g. ("AhKd", "Ts9d2c").
    """
    hero_key  = f"_cp_hero_{key}"
    board_key = f"_cp_board_{key}"
    suit_key  = f"_cp_suit_{key}"
    mode_key  = f"_cp_mode_{key}"

    # Initialise state
    if hero_key  not in st.session_state: st.session_state[hero_key]  = []
    if board_key not in st.session_state: st.session_state[board_key] = []
    if suit_key  not in st.session_state: st.session_state[suit_key]  = 's'
    if mode_key  not in st.session_state: st.session_state[mode_key]  = 'hero'

    hero_cards  = st.session_state[hero_key]
    board_cards = st.session_state[board_key]
    used_cards  = set(c.lower() for c in hero_cards + board_cards)
    current_suit = st.session_state[suit_key]
    current_mode = st.session_state[mode_key]

    # ── Mode toggle ──────────────────────────────────────────────────────────
    col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
    with col_m1:
        if st.button(
            f"🂠 Hero ({len(hero_cards)}/2)",
            key=f"{key}_mode_hero",
            type="primary" if current_mode == 'hero' else "secondary",
            use_container_width=True,
        ):
            st.session_state[mode_key] = 'hero'
            st.rerun()
    with col_m2:
        if st.button(
            f"⬜ Board ({len(board_cards)}/5)",
            key=f"{key}_mode_board",
            type="primary" if current_mode == 'board' else "secondary",
            use_container_width=True,
        ):
            st.session_state[mode_key] = 'board'
            st.rerun()

    # ── Selected cards display ───────────────────────────────────────────────
    disp_col1, disp_col2 = st.columns(2)
    with disp_col1:
        st.markdown(
            f"<div style='background:#161b27;border:1.5px solid #334155;"
            f"border-radius:8px;padding:8px 12px;min-height:52px'>"
            f"<div style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;"
            f"color:#64748b;margin-bottom:4px'>Hero Hand</div>"
            f"<div style='font-size:1.1rem;font-weight:700;letter-spacing:0.05em'>"
            + ("".join(
                f"<span style='color:{SUIT_COLORS[c[1].lower()]};margin-right:6px'>{format_card(c)}</span>"
                for c in hero_cards
            ) if hero_cards else "<span style='color:#334155;font-style:italic;font-size:0.8rem'>empty</span>")
            + "</div></div>",
            unsafe_allow_html=True,
        )
    with disp_col2:
        st.markdown(
            f"<div style='background:#161b27;border:1.5px solid #334155;"
            f"border-radius:8px;padding:8px 12px;min-height:52px'>"
            f"<div style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;"
            f"color:#64748b;margin-bottom:4px'>Board</div>"
            f"<div style='font-size:1.1rem;font-weight:700;letter-spacing:0.05em'>"
            + ("".join(
                f"<span style='color:{SUIT_COLORS[c[1].lower()]};margin-right:6px'>{format_card(c)}</span>"
                for c in board_cards
            ) if board_cards else "<span style='color:#334155;font-style:italic;font-size:0.8rem'>empty</span>")
            + "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

    # ── Suit selector ────────────────────────────────────────────────────────
    suit_labels = {'s': '♠ Spades', 'h': '♥ Hearts', 'd': '♦ Diamonds', 'c': '♣ Clubs'}
    suit_cols = st.columns(4)
    for i, suit in enumerate(SUITS):
        with suit_cols[i]:
            color = SUIT_COLORS[suit]
            is_active = current_suit == suit
            if st.button(
                suit_labels[suit],
                key=f"{key}_suit_{suit}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state[suit_key] = suit
                st.rerun()

    # ── Rank grid ────────────────────────────────────────────────────────────
    rank_cols = st.columns(13)
    for i, rank in enumerate(RANKS):
        card = rank + current_suit
        card_lower = card.lower()
        is_hero  = card_lower in [c.lower() for c in hero_cards]
        is_board = card_lower in [c.lower() for c in board_cards]
        is_used  = card_lower in used_cards

        with rank_cols[i]:
            # Style the label to indicate state
            if is_hero:
                label = f"**{rank}**"
                btn_type = "primary"
            elif is_board:
                label = f"*{rank}*"
                btn_type = "primary"
            else:
                label = rank
                btn_type = "secondary"

            pressed = st.button(
                label,
                key=f"{key}_rank_{rank}_{current_suit}",
                type=btn_type,
                disabled=is_used and not (is_hero or is_board),
                use_container_width=True,
            )

            if pressed:
                if is_hero:
                    st.session_state[hero_key] = [c for c in hero_cards if c.lower() != card_lower]
                elif is_board:
                    st.session_state[board_key] = [c for c in board_cards if c.lower() != card_lower]
                elif current_mode == 'hero':
                    if len(hero_cards) < 2:
                        st.session_state[hero_key].append(card)
                        if len(st.session_state[hero_key]) == 2:
                            st.session_state[mode_key] = 'board'
                    else:
                        st.warning("Hero hand full. Switch to Board or remove a card.")
                else:
                    if len(board_cards) < 5:
                        st.session_state[board_key].append(card)
                    else:
                        st.warning("Board full. Remove a card first.")
                st.rerun()

    # ── Action row ───────────────────────────────────────────────────────────
    act_col1, act_col2, act_col3, _ = st.columns([1, 1, 1, 2])
    with act_col1:
        if st.button("↺ Clear All", key=f"{key}_clear_all", use_container_width=True):
            st.session_state[hero_key]  = []
            st.session_state[board_key] = []
            st.session_state[mode_key]  = 'hero'
            st.rerun()
    with act_col2:
        if st.button("Clear Hero", key=f"{key}_clear_hero", use_container_width=True):
            st.session_state[hero_key] = []
            st.session_state[mode_key] = 'hero'
            st.rerun()
    with act_col3:
        if st.button("Clear Board", key=f"{key}_clear_board", use_container_width=True):
            st.session_state[board_key] = []
            st.rerun()

    # Return values
    hero  = "".join(st.session_state[hero_key])
    board = "".join(st.session_state[board_key])
    return hero, board


def card_picker_sidebar(key: str = "sidebar_picker") -> tuple[str, str]:
    """Compact version for sidebar — same API."""
    return card_picker_ui(key=key)