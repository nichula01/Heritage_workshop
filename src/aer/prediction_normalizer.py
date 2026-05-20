#!/usr/bin/env python3
import re

def normalize_prediction(pred: str) -> str:

    s = str(pred).strip()
    s = s.strip('"').strip("'").strip()

    prefixes = [
        r"^the author of the (?:element|cultural asset|inscription attached to the cultural asset) is\s+",
        r"^the language of the (?:writing|inscription(?: of the element attached to the cultural asset)?) is\s+",
        r"^the caption says\s+",
        r"^the caption states\s+",
        r"^the inscription on the cultural asset says\s+",
        r"^it says\s+",
        r"^the drawing is in the manuscript\s+",
        r"^the drawing is in\s+",
        r"^it was extracted from(?: the work)?[:\s]+",
        r"^the element on the cultural asset is\s+",
        r"^the technical characteristics of the element attached to the cultural asset are\s+",
    ]

    for p in prefixes:
        s2 = re.sub(p, "", s, flags=re.IGNORECASE)
        if s2 != s:
            s = s2

    s = s.strip().strip(".").strip()

    replacements = {
        "on the back": "Back",
        "back": "Back",
        "print": "Printing",
        "printing": "Printing",
        "in pencil": "Pencil",
    }

    low = s.lower()
    if low in replacements:
        return replacements[low]

    return s
