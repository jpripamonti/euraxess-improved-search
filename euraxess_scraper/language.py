from __future__ import annotations

LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "pl": "Polish",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "sk": "Slovak",
    "hr": "Croatian",
    "sl": "Slovenian",
    "el": "Greek",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "uk": "Ukrainian",
    "ru": "Russian",
    "tr": "Turkish",
    "ar": "Arabic",
}

LANGUAGE_UNKNOWN = "unknown"


def detect_language(text: str) -> str:
    """Return ISO 639-1 code for the dominant language of text, or 'unknown'."""
    if not text or len(text.strip()) < 30:
        return LANGUAGE_UNKNOWN
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 42
        code = detect(text[:4000])
        return code if code else LANGUAGE_UNKNOWN
    except Exception:
        return LANGUAGE_UNKNOWN


def language_label(code: str) -> str:
    """Return human-readable label for an ISO 639-1 code."""
    if code == LANGUAGE_UNKNOWN:
        return "Unknown"
    return LANGUAGE_LABELS.get(code, code.upper())


def default_language_labels() -> dict[str, str]:
    """Return ordered dict mapping language codes to display labels for UI dropdowns."""
    labels: dict[str, str] = {"all": "All languages"}
    labels.update(LANGUAGE_LABELS)
    return labels
