from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_TO_VAL = {r: i for i, r in enumerate(RANKS, start=2)}
SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}

HAND_CATEGORY = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush",
]

# ---------------------------------------------------------------------------
# Opponent range profiles  (list of 2-char combos or wildcards)
# Each profile is a set of canonical starting hand descriptors.
# We resolve them to actual Card pairs at runtime against the remaining deck.
# ---------------------------------------------------------------------------

# Canonical hand notation: "AKs", "AKo", "AA", etc.
# We store them as (rank1, rank2, suited) tuples for fast matching.

def _hand_key(c1: "Card", c2: "Card") -> Tuple[str, str, bool]:
    """Return canonical (hi_rank, lo_rank, suited) for two cards."""
    r1, r2 = c1.r, c2.r
    v1, v2 = RANK_TO_VAL[r1], RANK_TO_VAL[r2]
    if v1 < v2:
        r1, r2 = r2, r1
    return (r1, r2, c1.s == c2.s)


# Range profiles expressed as sets of canonical hand descriptors
# Format: "XX" = pocket pair, "XYs" = suited, "XYo" = offsuit, "XY" = both
RANGE_PROFILES: Dict[str, List[str]] = {
    "Nit": [
        "AA", "KK", "QQ", "JJ", "TT",
        "AKs", "AQs", "AJs", "AKo",
    ],
    "Tight": [
        "AA", "KK", "QQ", "JJ", "TT", "99", "88",
        "AKs", "AQs", "AJs", "ATs", "KQs",
        "AKo", "AQo", "AJo",
    ],
    "Balanced": [
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s",
        "KQs", "KJs", "KTs", "QJs", "QTs", "JTs",
        "AKo", "AQo", "AJo", "ATo", "KQo", "KJo",
        "T9s", "98s", "87s", "76s", "65s",
    ],
    "Loose": [
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "QJs", "QTs", "Q9s", "JTs", "J9s", "T9s", "98s", "87s", "76s", "65s", "54s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o",
        "KQo", "KJo", "KTo", "QJo", "QTo", "JTo",
    ],
    "Maniac": [],  # empty = truly random (fallback)
}


def _parse_range_descriptor(desc: str) -> Tuple[str, str, Optional[bool]]:
    """
    Parse "AKs" -> ("A","K", True)
         "AKo" -> ("A","K", False)
         "AK"  -> ("A","K", None)   # both
         "AA"  -> ("A","A", None)
    """
    suited: Optional[bool] = None
    if desc.endswith("s"):
        suited = True
        desc = desc[:-1]
    elif desc.endswith("o"):
        suited = False
        desc = desc[:-1]
    r1, r2 = desc[0].upper(), desc[1].upper()
    return r1, r2, suited


def _hand_matches_range(c1: "Card", c2: "Card", range_descs: List[str]) -> bool:
    if not range_descs:
        return True  # Maniac = any hand
    hi, lo, is_suited = _hand_key(c1, c2)
    for desc in range_descs:
        r1, r2, suited_req = _parse_range_descriptor(desc)
        # Normalize order
        if RANK_TO_VAL.get(r1, 0) < RANK_TO_VAL.get(r2, 0):
            r1, r2 = r2, r1
        if r1 == hi and r2 == lo:
            if suited_req is None:
                return True
            if suited_req == is_suited:
                return True
    return False


@dataclass(frozen=True)
class Card:
    r: str
    s: str

    @property
    def val(self) -> int:
        return RANK_TO_VAL[self.r]

    def pretty(self) -> str:
        return f"{self.r}{SUIT_SYMBOL.get(self.s, self.s)}"


def parse_cards(compact: str) -> List[Card]:
    compact = str(compact).strip()
    if len(compact) == 0:
        return []
    if len(compact) % 2 != 0:
        raise ValueError("Card string must have even length (e.g. 'AhKd').")
    cards: List[Card] = []
    for i in range(0, len(compact), 2):
        r = compact[i].upper()
        s = compact[i + 1].lower()
        if r not in RANKS or s not in SUITS:
            raise ValueError(f"Invalid card token: '{compact[i:i+2]}'")
        cards.append(Card(r, s))
    return cards


def format_cards(compact: str) -> str:
    cards = parse_cards(compact)
    return " ".join(c.pretty() for c in cards)


def validate_no_duplicates(hero_hand: str, board: str) -> None:
    hero = parse_cards(hero_hand)
    brd = parse_cards(board)
    if len(hero) != 2:
        raise ValueError("Hero hand must contain exactly 2 cards.")
    if len(brd) > 5:
        raise ValueError("Board can't have more than 5 cards.")
    seen = set()
    for c in hero + brd:
        key = (c.r, c.s)
        if key in seen:
            raise ValueError(f"Duplicate card: {c.pretty()}")
        seen.add(key)


def deck_excluding(excluded: List[Card]) -> List[Card]:
    ex = {(c.r, c.s) for c in excluded}
    return [Card(r, s) for r in RANKS for s in SUITS if (r, s) not in ex]


def _is_straight(vals: List[int]) -> Tuple[bool, int]:
    v = sorted(set(vals), reverse=True)
    if len(v) < 5:
        return False, 0
    for i in range(len(v) - 4):
        window = v[i:i + 5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return True, window[0]
    if {14, 5, 4, 3, 2}.issubset(set(v)):
        return True, 5
    return False, 0


def evaluate_5(cards: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    if len(cards) != 5:
        raise ValueError("evaluate_5 needs exactly 5 cards")
    vals = sorted([c.val for c in cards], reverse=True)
    suits = [c.s for c in cards]
    counts: Dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    pattern = sorted(counts.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    is_str, str_high = _is_straight(vals)
    if is_flush and is_str:
        return 8, (str_high,)
    if pattern == [4, 1]:
        return 7, (groups[0][0], groups[1][0])
    if pattern == [3, 2]:
        return 6, (groups[0][0], groups[1][0])
    if is_flush:
        return 5, tuple(vals)
    if is_str:
        return 4, (str_high,)
    if pattern == [3, 1, 1]:
        kickers = sorted([g[0] for g in groups[1:]], reverse=True)
        return 3, (groups[0][0], *kickers)
    if pattern == [2, 2, 1]:
        return 2, (groups[0][0], groups[1][0], groups[2][0])
    if pattern == [2, 1, 1, 1]:
        kickers = sorted([g[0] for g in groups[1:]], reverse=True)
        return 1, (groups[0][0], *kickers)
    return 0, tuple(vals)


def best_hand_rank(cards7: List[Card]) -> Tuple[int, Tuple[int, ...], str]:
    best = (-1, ())
    for combo in combinations(cards7, 5):
        rank = evaluate_5(list(combo))
        if rank > best:
            best = rank
    cat, tb = best
    return cat, tb, HAND_CATEGORY[cat]


def compare(hero7: List[Card], opp7: List[Card]) -> int:
    hr = best_hand_rank(hero7)[:2]
    orr = best_hand_rank(opp7)[:2]
    if hr > orr:
        return 1
    if hr < orr:
        return -1
    return 0


def monte_carlo_equity(
    hero_hand: str,
    board: str,
    iters: int = 3000,
    seed: int = 42,
    opp_range_profile: str = "Balanced",
) -> float:
    """
    Range-aware heads-up equity estimate.
    Opponent hands are sampled only from the specified range profile.
    Falls back to random if no valid range hands are found after many attempts.
    """
    validate_no_duplicates(hero_hand, board)
    random.seed(seed)

    hero = parse_cards(hero_hand)
    brd = parse_cards(board)
    known = hero + brd
    d = deck_excluding(known)

    range_descs = RANGE_PROFILES.get(opp_range_profile, [])
    need_board = 5 - len(brd)
    wins = ties = 0
    valid_iters = 0
    attempts = 0
    max_attempts = iters * 20

    while valid_iters < iters and attempts < max_attempts:
        attempts += 1
        if len(d) < 2 + need_board:
            break

        # Sample opponent hand
        opp_candidates = random.sample(d, min(20, len(d)))
        opp = None
        for i in range(0, len(opp_candidates) - 1, 2):
            c1, c2 = opp_candidates[i], opp_candidates[i + 1]
            if _hand_matches_range(c1, c2, range_descs):
                opp = [c1, c2]
                break

        if opp is None:
            # Fallback: just pick 2 random cards (Maniac / no match)
            opp = random.sample(d, 2)

        remaining = [c for c in d if c not in opp]
        if len(remaining) < need_board:
            continue

        runout = random.sample(remaining, need_board)
        full_board = brd + runout

        res = compare(hero + full_board, opp + full_board)
        if res == 1:
            wins += 1
        elif res == 0:
            ties += 1
        valid_iters += 1

    if valid_iters == 0:
        return 0.5
    return (wins + 0.5 * ties) / valid_iters


def get_range_description(profile: str) -> str:
    descriptions = {
        "Nit": "Top 3% of hands — premium pairs and AK only",
        "Tight": "Top 8% — strong pairs and big broadway hands",
        "Balanced": "Top 20% — solid TAG range with suited connectors",
        "Loose": "Top 40% — wide range including speculative hands",
        "Maniac": "Any two cards — truly random opponent",
    }
    return descriptions.get(profile, "Unknown range")
