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
from datetime import datetime
from typing import Optional

from google import genai
from google.genai import types

from src.connector import ModelConnector
from src.nlp_parser import ParsedQuery, parse_query

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
# Main class
# ---------------------------------------------------------------------------

class GeminiEnhancer:
    """Wraps Gemini Flash Lite and converts ML predictions to natural language."""

    MODEL = "gemini-flash-lite-latest"

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. "
                "Set the GEMINI_API_KEY environment variable or pass api_key=."
            )
        self._client = genai.Client(api_key=key)
        self._config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.4,
        )

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

        now  = datetime.now()
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
            parsed.hour if parsed.hour is not None else datetime.now().hour
        )
        prompt = _build_prompt(prediction, parsed, original_query, resolved_hour)
        try:
            response = self._client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
                config=self._config,
            )
            return response.text.strip()
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
