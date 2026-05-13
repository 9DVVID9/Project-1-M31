"""
LLM Enhancement (Module 4) — uses Gemini Flash Lite to turn raw ML predictions
into natural language responses.

Requires: GEMINI_API_KEY environment variable (or pass api_key to constructor).

Usage:
    from src.llm_enhancer import GeminiEnhancer
    from src.connector import ModelConnector

    mc  = ModelConnector()
    llm = GeminiEnhancer()          # reads GEMINI_API_KEY from env
    response = llm.respond("Is it busy on Monday at 8am?", mc)
    print(response["text"])
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Optional

from google import genai
from google.genai import types

from src.connector import ModelConnector
from src.nlp_parser import ParsedQuery, parse_query

try:
    from zoneinfo import ZoneInfo
    _LOCAL_TZ = ZoneInfo("Europe/Madrid")
except Exception:
    _LOCAL_TZ = None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a concise, helpful traffic assistant for Arago Street in Barcelona, Spain.

Rules:
- Reply in the SAME language the user wrote in (English or Spanish).
- Keep answers to 2–3 short sentences.
- Be direct: state the traffic level, then give one practical tip.
- Never mention model names, confidence percentages, or technical details.
- If traffic is High, suggest an alternative time or route.
- If traffic is Low, confirm it is a good time to travel.
- If asked for advice (best/worst time), give a concrete time range.
- TENSE RULE: Only use "currently" or present tense when the context explicitly says "RIGHT NOW".
  For all other queries, use future/conditional tense: "is expected to be", "will likely be", "you can expect".
- SCOPE RULE: You only know about TRAFFIC conditions, not weather forecasts. If asked about weather,
  clarify you can only provide traffic information, and mention the weather context only as it affects driving.
"""

# ---------------------------------------------------------------------------
# Scope filter (out-of-topic detection)
# ---------------------------------------------------------------------------

# Traffic + weather vocabulary. Lowercased; matched with word boundaries
# against the raw user text. Weather is in scope because the model uses
# weather features and users naturally ask about it.
_ON_TOPIC_KEYWORDS = {
    # English — traffic
    "traffic", "congestion", "congested", "jam", "jammed", "busy",
    "rush", "road", "roads", "street", "drive", "driving",
    "commute", "commuting", "car", "cars", "vehicle", "vehicles",
    # English — weather
    "weather", "rain", "raining", "rainy", "snow", "snowing", "wind",
    "windy", "temperature", "hot", "cold", "sunny", "cloudy", "forecast",
    # Spanish — traffic
    "tráfico", "trafico", "atasco", "atascos", "congestión", "congestion",
    "carretera", "calle", "coche", "coches", "vehículo", "vehiculo",
    "conducir", "circulación", "circulacion",
    # Spanish — weather
    "tiempo", "clima", "lluvia", "lloviendo", "llover", "nieve", "viento",
    "temperatura", "calor", "frío", "frio", "soleado", "nublado",
    "pronóstico", "pronostico",
}

# Cheap Spanish-vs-English signal used only for the rejection message.
_SPANISH_HINTS_CHARS = {"á", "é", "í", "ó", "ú", "ñ", "¿", "¡"}
_SPANISH_HINTS_WORDS = {
    "qué", "que", "cómo", "como", "cuándo", "cuando", "cuál", "cual",
    "dónde", "donde", "quién", "quien", "está", "esta", "hoy", "ayer",
    "mañana", "manana", "lunes", "martes", "miércoles", "miercoles",
    "jueves", "viernes", "sábado", "sabado", "domingo", "hola",
    "porque", "gracias",
}

_OUT_OF_SCOPE_EN = (
    "I can only answer questions about traffic on Aragó Street, Barcelona. "
    "My purpose is to forecast traffic conditions, not general questions. "
    "Try something like: \"How's traffic Monday at 12 pm?\""
)
_OUT_OF_SCOPE_ES = (
    "Solo puedo responder preguntas sobre el tráfico en la Avenida Aragó, Barcelona. "
    "Mi propósito es predecir condiciones de tráfico, no responder preguntas generales. "
    "Prueba con algo como: \"¿Cómo está el tráfico el lunes a las 12?\""
)


def _detect_language(text: str) -> str:
    """Return 'es' if the text shows Spanish hints, else 'en'."""
    lowered = text.lower()
    for ch in _SPANISH_HINTS_CHARS:
        if ch in lowered:
            return "es"
    for word in _SPANISH_HINTS_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return "es"
    return "en"


def _is_on_topic(parsed: ParsedQuery, raw_text: str) -> bool:
    """Off-topic when the parser found no time/day AND no traffic/weather word appears."""
    if parsed.hour is not None or parsed.day_of_week is not None:
        return True
    lowered = raw_text.lower()
    for kw in _ON_TOPIC_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            return True
    return False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GeminiEnhancer:
    """Wraps Gemini Flash Lite and converts ML predictions to natural language."""

    MODEL = "gemini-flash-lite-latest"

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            self._client = None
            self._config = None
            return
        self._client = genai.Client(api_key=key)
        self._config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.4,
        )

    @property
    def is_fallback_mode(self) -> bool:
        """True when no API key was provided — responses use template fallback."""
        return self._client is None

    # ------------------------------------------------------------------
    # High-level pipeline entry point
    # ------------------------------------------------------------------

    def respond(self, user_text: str, mc: ModelConnector) -> dict:
        """
        Full pipeline: parse → predict → enhance.

        Args:
            user_text: Raw user query string.
            mc:        Initialised ModelConnector instance.

        Returns dict with keys:
            text        – natural language response (str)
            traffic_level – "Low" | "Medium" | "High"
            confidence  – float 0–1
            hour        – resolved hour used for prediction
            day_name    – resolved day name
            weather     – weather dict used
            intent      – parsed intent
            is_ambiguous – whether hour or day was defaulted
        """
        parsed = parse_query(user_text)

        # Reject off-topic queries before touching the model or the API.
        if not _is_on_topic(parsed, user_text):
            lang = _detect_language(user_text)
            return {
                "text": _OUT_OF_SCOPE_ES if lang == "es" else _OUT_OF_SCOPE_EN,
                "traffic_level": None,
                "confidence": None,
                "hour": None,
                "day_name": None,
                "weather": {},
                "intent": "out_of_scope",
                "is_ambiguous": False,
            }

        now  = datetime.now(_LOCAL_TZ)
        hour = parsed.hour        if parsed.hour        is not None else now.hour
        dow  = parsed.day_of_week if parsed.day_of_week is not None else now.weekday()

        prediction = mc.predict(hour=hour, day_of_week=dow)
        text = self.enhance(prediction, parsed, user_text, hour)

        return {
            "text": text,
            "traffic_level": prediction["traffic_level"],
            "confidence": prediction["confidence"],
            "hour": hour,
            "day_name": prediction["day_name"],
            "weather": prediction["weather"],
            "intent": parsed.intent,
            "is_ambiguous": parsed.is_ambiguous,
        }

    # ------------------------------------------------------------------
    # LLM enhancement
    # ------------------------------------------------------------------

    def enhance(
        self,
        prediction: dict,
        parsed: ParsedQuery,
        original_query: str = "",
        hour: Optional[int] = None,
    ) -> str:
        """
        Build a Gemini prompt from prediction data and return the response.
        Falls back to a template-based response if the API call fails.
        """
        resolved_hour = hour if hour is not None else (
            parsed.hour if parsed.hour is not None else datetime.now(_LOCAL_TZ).hour
        )
        if self._client is None:
            return _fallback_response(prediction, resolved_hour)
        prompt = _build_prompt(prediction, parsed, original_query, resolved_hour)
        try:
            response = self._client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
                config=self._config,
            )
            text = (response.text or "").strip()
            if not text:
                return _fallback_response(prediction, resolved_hour)
            return text
        except Exception as e:
            logging.warning("Gemini API call failed: %s", e)
            return _fallback_response(prediction, resolved_hour)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(
    prediction: dict,
    parsed: ParsedQuery,
    original_query: str,
    hour: int,
) -> str:
    level   = prediction["traffic_level"]
    day     = prediction["day_name"]
    weather = prediction["weather"]
    intent  = parsed.intent

    hour_str = f"{hour:02d}:00"
    temp     = weather.get("temperature", "?")
    precip   = weather.get("precipitation", 0.0)
    wind     = weather.get("windspeed", "?")
    rain_note = "with some rain" if float(precip) > 0.5 else "dry conditions"

    if intent == "current":
        context = (
            f"The user asked about traffic RIGHT NOW on Arago Street, Barcelona.\n"
            f"Current conditions: traffic is {level}, {temp}°C, {rain_note}, wind {wind} km/h."
        )
    elif intent == "advice":
        context = (
            f"The user is asking for ADVICE on when to travel on {day}.\n"
            f"Based on historical patterns, traffic at {hour_str} on {day} "
            f"is typically {level}.\n"
            f"Weather context: {temp}°C, {rain_note}."
        )
    else:  # forecast or unknown
        context = (
            f"FORECAST (not current) — predicted traffic for {day} at {hour_str} "
            f"on Arago Street, Barcelona: {level}.\n"
            f"Expected weather: {temp}°C, {rain_note}, wind {wind} km/h."
        )

    return (
        f"User query: \"{original_query or parsed.raw_text}\"\n\n"
        f"{context}\n\n"
        f"Write a helpful 2–3 sentence response to the user's query."
    )


# ---------------------------------------------------------------------------
# Fallback (no API / quota exceeded)
# ---------------------------------------------------------------------------

def _fallback_response(prediction: dict, hour: int) -> str:
    """Template-based response used when Gemini API is unavailable."""
    level = prediction["traffic_level"]
    day   = prediction["day_name"]

    messages: dict[str, str] = {
        "Low": (
            f"Traffic on {day} at {hour:02d}:00 is expected to be light. "
            "It's a good time to travel on Arago Street."
        ),
        "Medium": (
            f"Traffic on {day} at {hour:02d}:00 is expected to be moderate. "
            "Allow a few extra minutes for your journey."
        ),
        "High": (
            f"Traffic on {day} at {hour:02d}:00 is expected to be heavy. "
            "Consider travelling earlier or later to avoid congestion."
        ),
    }
    return messages.get(level, f"Traffic on {day} at {hour:02d}:00 is predicted to be {level}.")
