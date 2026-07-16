import re


_REPLACEMENTS = [
    (r"\baccident(s)?\b", "road"),
    (r"\bpothole(s)?\b", "road"),
    (r"\bcrater(s)?\b", "road"),
    (r"\bkhadda(s)?\b", "road"),
    (r"\broad(s)?\b", "road"),
    (r"\bstreet(s)?\b", "road"),
    (r"\btraffic\b", "road"),
    (r"\bwater\b", "paani"),
    (r"\btap\b", "paani"),
    (r"\bpipeline(s)?\b", "paani"),
    (r"\bsupply\b", "paani"),
    (r"\bleak(s|age)?\b", "paani"),
    (r"\belectricity\b", "bijli"),
    (r"\bpower\b", "bijli"),
    (r"\blight(s)?\b", "bijli"),
    (r"\bcurrent\b", "bijli"),
    (r"\bvoltage\b", "bijli"),
    (r"\btransformer(s)?\b", "bijli"),
    (r"\bgarbage\b", "kachra"),
    (r"\btrash\b", "kachra"),
    (r"\bwaste\b", "kachra"),
    (r"\bdustbin(s)?\b", "kachra"),
    (r"\bsanitation\b", "kachra"),
    (r"\bdrain(s|age)?\b", "nali"),
    (r"\bsewer(s|age)?\b", "nali"),
]


def normalize_complaint_text(text):
    value = str(text).lower()
    value = re.sub(r"[^a-z0-9\u0900-\u097f\s]+", " ", value)
    for pattern, replacement in _REPLACEMENTS:
        value = re.sub(pattern, f" {replacement} ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def augment_complaint_text(text):
    base = normalize_complaint_text(text)
    variants = {base}

    synonyms = {
        "road": ["road", "sadak", "street"],
        "paani": ["paani", "water", "tap"],
        "bijli": ["bijli", "power", "light"],
        "kachra": ["kachra", "garbage", "trash"],
    }

    for token, choices in synonyms.items():
        if token in base:
            for replacement in choices:
                variants.add(re.sub(rf"\b{token}\b", replacement, base))

    return list(variants)


def infer_priority_override(text):
    raw = str(text).lower()
    normalized = normalize_complaint_text(text)
    high_priority_markers = [
        "accident",
        "injury",
        "danger",
        "urgent",
        "fire",
        "collapse",
        "collapsed",
        "pothole",
        "leak",
        "no water",
        "no bijli",
        "power cut",
        "electrocution",
        "burst",
        "road closed",
        "broken road",
    ]

    for marker in high_priority_markers:
        if marker in raw or marker in normalized:
            return "High"

    return None