"""
NLP Parser (Module 2) — extracts temporal features and intent from natural language.

Supports English and Spanish (Barcelona context).
No external dependencies — uses only re and datetime (stdlib).

Usage:
    from src.nlp_parser import parse_query

    result = parse_query("What's traffic like at 8am on Monday?")
    result = parse_query("Como es el trafico el lunes a las 8?")
    # ParsedQuery(hour=8, day_of_week=0, intent='forecast', is_ambiguous=False, ...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    hour: Optional[int]         # 0–23, None if not found
    day_of_week: Optional[int]  # 0=Monday … 6=Sunday, None if not found
    intent: str                 # see _INTENTS below
    raw_text: str               # original input unchanged
    is_ambiguous: bool          # True when hour or day_of_week is None

# Valid intent values:
# "forecast"  — user wants a prediction for a specific time
# "current"   — user wants right-now traffic
# "advice"    — user wants a recommendation (best/worst time to travel)
# "unknown"   — no temporal or advisory cue found


# ---------------------------------------------------------------------------
# Day name lookup  (English + Spanish, full + abbreviated)
# ---------------------------------------------------------------------------

_DAY_MAP: dict[str, int] = {
    # English
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
    # Spanish (accented and unaccented variants)
    "lunes": 0, "lun": 0,
    "martes": 1, "mar": 1,
    "miercoles": 2, "miércoles": 2, "mie": 2, "mié": 2,
    "jueves": 3, "jue": 3,
    "viernes": 4, "vie": 4,
    "sabado": 5, "sábado": 5, "sab": 5, "sáb": 5,
    "domingo": 6, "dom": 6,
}

# ---------------------------------------------------------------------------
# Named time-period lookup  (English + Spanish, longest match first)
# ---------------------------------------------------------------------------

_PERIOD_MAP: list[tuple[str, int]] = [
    # ── English multi-word (checked before single words) ──────────────────
    (r"early\s+morning",            7),
    (r"late\s+night",              23),
    (r"afternoon\s+rush",          17),
    (r"morning\s+rush",             8),
    (r"rush\s+hour",                8),
    (r"lunch\s+(?:time|hour)?",    12),
    # ── Spanish multi-word ────────────────────────────────────────────────
    (r"hora\s+punta",               8),   # rush hour (morning default)
    (r"primera\s+hora",             7),   # first thing in the morning
    (r"por\s+la\s+ma[nñ]ana",       9),   # in the morning
    (r"por\s+la\s+tarde",          14),   # in the afternoon
    (r"por\s+la\s+noche",          21),   # in the evening/night
    (r"esta\s+ma[nñ]ana",           9),   # this morning
    (r"la\s+ma[nñ]ana",             9),   # the morning
    # ── English single-word ───────────────────────────────────────────────
    (r"midnight",                   0),
    (r"dawn",                       6),
    (r"morning",                    9),
    (r"\bnoon\b|\bmidday\b",       12),
    (r"afternoon",                 14),
    (r"evening",                   18),
    (r"night|tonight",             21),
    (r"\brush\b",                   8),
    # ── Spanish single-word ───────────────────────────────────────────────
    (r"medianoche",                 0),   # midnight
    (r"madrugada",                  3),   # early hours / wee hours
    (r"amanecer|alba",              6),   # dawn
    (r"mediod[ií]a",               12),  # noon/midday
    (r"\btarde\b",                 14),  # afternoon
    (r"\bnoche\b",                 21),  # night
]

# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------

_CURRENT_RE = re.compile(
    r"\b(now|currently|right\s+now|at\s+the\s+moment|at\s+this\s+moment"
    r"|ahora|ahora\s+mismo|en\s+este\s+momento)\b",
    re.IGNORECASE,
)

_ADVICE_RE = re.compile(
    r"\b(best\s+time|worst\s+time|when\s+(?:is|to|should)|avoid|recommend"
    r"|lighter|less\s+busy|least\s+busy|most\s+busy|heaviest"
    r"|mejor\s+(?:hora|momento)|cuando\s+(?:es|hay|debo)|evitar"
    r"|recomend(?:ar|acion)|menos\s+tr[aá]fico|m[aá]s\s+tr[aá]fico)\b",
    re.IGNORECASE,
)

# "a las 8", "a las 14", "a las 8:30" — Spanish "at N o'clock"
_SPANISH_AT_RE = re.compile(
    r"\ba\s+las?\s+(\d{1,2})(?::(\d{2}))?\b",
    re.IGNORECASE,
)

# "at 8", "at 14" — English bare number after "at"
_AT_NUM_RE = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\b",
    re.IGNORECASE,
)

# "around 8pm", "around 14:00"
_AROUND_RE = re.compile(
    r"\baround\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.IGNORECASE,
)

# Full explicit clock: "8am", "8:30", "08:00", "8 o'clock", "14h"
_CLOCK_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|h\b|o[\'\s]?clock)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(text: str) -> ParsedQuery:
    """
    Extract hour, day_of_week, and intent from a natural language string.
    Supports English and Spanish.

    Args:
        text: Any user query string.

    Returns:
        ParsedQuery. hour / day_of_week are None when not found (is_ambiguous=True).
    """
    norm = text.lower().strip()

    hour = _extract_hour(norm)
    day_of_week = _extract_day(norm)
    intent = _classify_intent(norm, hour, day_of_week)
    is_ambiguous = hour is None or day_of_week is None

    return ParsedQuery(
        hour=hour,
        day_of_week=day_of_week,
        intent=intent,
        raw_text=text,
        is_ambiguous=is_ambiguous,
    )


# ---------------------------------------------------------------------------
# Hour extraction
# ---------------------------------------------------------------------------

def _extract_hour(text: str) -> Optional[int]:
    """
    Return 0–23 or None.
    Priority: explicit clock with suffix → Spanish "a las N" →
              English "at N" / "around N" → named period → "now".
    """

    # 1. Explicit clock with am/pm/h/o'clock qualifier  e.g. "8am", "14h", "8 o'clock"
    m = _CLOCK_RE.search(text)
    if m:
        h = _apply_ampm(int(m.group(1)), (m.group(3) or "").lower().strip())
        if h is not None:
            return h

    # 2. Spanish "a las N" / "a las N:MM"  e.g. "a las 8", "a las 14:30"
    m = _SPANISH_AT_RE.search(text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h

    # 3. English "at N" or "at N:MM"  e.g. "at 8", "at 17:00"
    m = _AT_NUM_RE.search(text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h

    # 4. "around N[am/pm]"  e.g. "around 8pm", "around 14"
    m = _AROUND_RE.search(text)
    if m:
        h = _apply_ampm(int(m.group(1)), (m.group(3) or "").lower().strip())
        if h is not None:
            return h

    # 5. Named time periods (longest-match order guaranteed by list order)
    for pattern, hour in _PERIOD_MAP:
        if re.search(pattern, text, re.IGNORECASE):
            return hour

    # 6. "now" / "ahora" → current local hour
    if _CURRENT_RE.search(text):
        return datetime.now().hour

    return None


def _apply_ampm(raw: int, suffix: str) -> Optional[int]:
    """Convert raw hour + am/pm suffix to 0–23. Returns None if out of range."""
    if raw > 23:
        return None
    if suffix == "pm" and raw != 12:
        raw += 12
    elif suffix == "am" and raw == 12:
        raw = 0
    return min(raw, 23)


# ---------------------------------------------------------------------------
# Day extraction
# ---------------------------------------------------------------------------

def _extract_day(text: str) -> Optional[int]:
    """Return 0–6 or None. Priority: explicit name → relative word."""
    today = datetime.now().weekday()

    # 1. Explicit day names (English + Spanish)
    for token, idx in _DAY_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", text, re.IGNORECASE):
            return idx

    # 2. Relative English keywords
    if re.search(r"\btoday\b", text, re.IGNORECASE):
        return today
    if re.search(r"\btomorrow\b", text, re.IGNORECASE):
        return (today + 1) % 7
    if re.search(r"\byesterday\b", text, re.IGNORECASE):
        return (today - 1) % 7
    if re.search(r"\bweekend\b", text, re.IGNORECASE):
        return 5  # Saturday as representative
    if re.search(r"\bweekday\b", text, re.IGNORECASE):
        return 0  # Monday as representative

    # 3. Relative Spanish keywords
    # "mañana" = tomorrow when it appears standalone (not as part of "por la mañana" / "esta mañana")
    # Check each occurrence individually so "manana por la manana" → tomorrow (first) + morning (second)
    for m in re.finditer(r"\bma[nñ]ana\b", text, re.IGNORECASE):
        prefix = text[: m.start()]
        if not re.search(r"\b(por\s+la|esta)\s*$", prefix.rstrip(), re.IGNORECASE):
            return (today + 1) % 7
    if re.search(r"\bhoy\b", text, re.IGNORECASE):
        return today
    if re.search(r"\bayer\b", text, re.IGNORECASE):
        return (today - 1) % 7
    if re.search(r"\bfin\s+de\s+semana\b", text, re.IGNORECASE):
        return 5  # Saturday
    if re.search(r"\bd[ií]a\s+laborable\b", text, re.IGNORECASE):
        return 0  # Monday

    # 4. "now" / "ahora" → today
    if _CURRENT_RE.search(text):
        return today

    return None


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _classify_intent(
    text: str,
    hour: Optional[int],
    day_of_week: Optional[int],
) -> str:
    if _ADVICE_RE.search(text):
        return "advice"
    if _CURRENT_RE.search(text):
        return "current"
    if hour is not None or day_of_week is not None:
        return "forecast"
    return "unknown"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _today = datetime.now().weekday()
    _tomorrow = (_today + 1) % 7
    _yesterday = (_today - 1) % 7

    test_cases: list[tuple[str, Optional[int], Optional[int], str]] = [
        # ── English: explicit clock ────────────────────────────────────────
        ("What's traffic like at 8am on Monday?",         8,        0,         "forecast"),
        ("How busy is it at 17:00 on Friday?",            17,       4,         "forecast"),
        ("Traffic at 8 on Wednesday",                     8,        2,         "forecast"),
        ("Around 3pm on Saturday",                        15,       5,         "forecast"),
        ("At 12:30 on Tuesday",                           12,       1,         "forecast"),
        # ── English: named periods ─────────────────────────────────────────
        ("Is it busy during rush hour on Wednesday?",     8,        2,         "forecast"),
        ("Check traffic this afternoon on Saturday",      14,       5,         "forecast"),
        ("Traffic on Sunday night",                       21,       6,         "forecast"),
        ("Morning commute tomorrow",                      9,        _tomorrow, "forecast"),
        ("What about noon today?",                        12,       _today,    "forecast"),
        ("Traffic at midnight",                           0,        None,      "forecast"),
        ("Afternoon rush on Thursday",                    17,       3,         "forecast"),
        ("Show me the weekend traffic",                   None,     5,         "forecast"),
        # ── English: current ──────────────────────────────────────────────
        ("How is traffic right now?",                     None,     None,      "current"),
        ("Is it congested now?",                          None,     None,      "current"),
        # ── English: advice ───────────────────────────────────────────────
        ("What is the best time to drive?",               None,     None,      "advice"),
        ("When should I avoid the road?",                 None,     None,      "advice"),
        ("When is traffic lightest on Monday?",           None,     0,         "advice"),
        # ── Spanish: days ─────────────────────────────────────────────────
        ("Como es el trafico el lunes a las 8?",          8,        0,         "forecast"),
        ("Hay mucho trafico el viernes por la tarde?",    14,       4,         "forecast"),
        ("Trafico el sabado por la noche",                21,       5,         "forecast"),
        ("Como esta el trafico ahora mismo?",             None,     None,      "current"),
        ("El trafico manana por la manana",               9,        _tomorrow, "forecast"),
        ("Mejor hora para conducir el miercoles",         None,     2,         "advice"),
        # ── Edge cases ────────────────────────────────────────────────────
        ("hello",                                         None,     None,      "unknown"),
        ("Tell me about traffic",                         None,     None,      "unknown"),
    ]

    pad = max(len(q) for q, *_ in test_cases)
    pass_count = 0

    sections = [
        ("English: explicit clock", 0, 5),
        ("English: named periods",  5, 13),
        ("English: current",       13, 15),
        ("English: advice",        15, 18),
        ("Spanish",                18, 24),
        ("Edge cases",             24, 26),
    ]

    idx = 0
    for section_name, start, end in sections:
        print(f"\n--- {section_name} ---")
        for query, exp_hour, exp_dow, exp_intent in test_cases[start:end]:
            result = parse_query(query)
            is_current = exp_intent == "current"
            hour_ok   = True if is_current else (result.hour == exp_hour)
            dow_ok    = True if is_current else (result.day_of_week == exp_dow)
            intent_ok = result.intent == exp_intent
            ok = hour_ok and dow_ok and intent_ok
            status = "PASS" if ok else "FAIL"
            if ok:
                pass_count += 1
            print(
                f"  [{status}] {query:<{pad}} "
                f"hour={str(result.hour):<5} dow={str(result.day_of_week):<5} "
                f"intent={result.intent}"
            )
            if not ok:
                print(f"           expected: hour={exp_hour} dow={exp_dow} intent={exp_intent}")

    total = len(test_cases)
    print(f"\n{pass_count}/{total} passed")
