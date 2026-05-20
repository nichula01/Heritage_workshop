#!/usr/bin/env python3
import re
from typing import Optional

def clean_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def strip_trailing_punct(s: str) -> str:
    return re.sub(r'[\s\.\,\;\:\!\?"]+$', "", s).strip()

def truncate_structured_field(s: str) -> str:
    s = clean_text(s)

    stop_markers = [
        ". The ",
        ". It ",
        ". It's ",
        ". These ",
        ". This ",
        ". On the ",
        ". A ",
        ". An ",
    ]

    cut_positions = []

    for marker in stop_markers:
        idx = s.find(marker)
        if idx != -1:
            cut_positions.append(idx)

    if cut_positions:
        s = s[:min(cut_positions)]

    return strip_trailing_punct(s)

def quoted_span(text: str) -> Optional[str]:
    m = re.search(r'["“](.+?)["”]', text)
    if m:
        return strip_trailing_punct(m.group(1))
    return None

def find_after_prefix(text: str, prefixes):
    text = clean_text(text)
    lower = text.lower()

    for p in prefixes:
        idx = lower.find(p.lower())
        if idx != -1:
            out = text[idx + len(p):].strip()
            out = truncate_structured_field(out)
            return out

    return None

def extract_affixed_language(e: str) -> Optional[str]:
    prefixes = [
        "the language of the inscription of the element attached to the cultural asset is ",
        "the language of the writing on the element attached to the cultural asset is ",
        "the language of the inscription of the element attached to cultural asset is ",
    ]
    return find_after_prefix(e, prefixes)

def extract_author(e: str) -> Optional[str]:
    prefixes = [
        "the author of the inscription attached to the cultural asset is ",
        "the author of the dedication to the cultural asset is ",
        "the author of the inscription affixed to the cultural asset is ",
        "the author of the cultural asset is ",
    ]
    return find_after_prefix(e, prefixes)

def extract_authorcriterion(e: str) -> Optional[str]:
    low = clean_text(e).lower()

    patterns = [
        ("based on stylistic analysis", "Stylistic analysis"),
        ("based on documentation", "Documentation"),
        ("based on context", "Background"),
        ("based on origin", "Origin"),
        ("based on the signature", "Signature"),
        ("based on signature", "Signature"),
        ("based on inscription", "Inscription"),
    ]

    for pat, out in patterns:
        if pat in low:
            return out

    return None

def extract_affixedposition(e: str) -> Optional[str]:

    e = clean_text(e)
    low = e.lower()

    direct = [
        ("on the back", "On the back"),
        ("back", "Back"),
        ("in the right mirror", "In the right mirror"),
        ("on the horizontal band", "On the horizontal band"),
        ("on the pergamino: recto", "On the pergamino: recto"),
        ("on the recto", "On the recto"),
        ("on the verso", "On the verso"),
    ]

    for pat, out in direct:
        if pat in low:
            return out

    patterns = [
        r"found in the ([^\.]+)",
        r"is found in the ([^\.]+)",
        r"is on the ([^\.]+)",
        r"is located on the ([^\.]+)",
        r"is located in the ([^\.]+)",
    ]

    for pat in patterns:
        m = re.search(pat, low)
        if m:
            return truncate_structured_field(m.group(1).strip().title())

    return None

def extract_affixedtranscript(e: str) -> Optional[str]:

    e = clean_text(e)

    q = quoted_span(e)
    if q:
        return q

    m = re.search(r"on the cultural asset it says:\s*(.+)$", e, re.IGNORECASE)

    if m:

        out = m.group(1).strip()

        stop_markers = [
            ". The ",
            ". It ",
            ". It's ",
            ". These ",
            ". This ",
            ". On the ",
        ]

        cut_positions = []

        for marker in stop_markers:
            idx = out.find(marker)
            if idx != -1:
                cut_positions.append(idx)

        if cut_positions:
            out = out[:min(cut_positions)]

        return strip_trailing_punct(out)

    return None

def extract_affixedtechnique(e: str) -> Optional[str]:

    low = clean_text(e).lower()

    if "made in print" in low or "was made in print" in low:
        return "Printing"

    if "was penned" in low or "made in pencil" in low:
        return "Pencil"

    if "made by engraving" in low:
        return "Engraving"

    if "made to perfection" in low:
        return "Fitting"

    m = re.search(r"was made in ([a-z ]+)", low)

    if m:
        out = m.group(1).strip()

        mapping = {
            "print": "Printing",
            "printing": "Printing",
            "pencil": "Pencil",
        }

        return mapping.get(out, out.title())

    return None

def extract_affixedelement(e: str) -> Optional[str]:

    low = clean_text(e).lower()

    if "inscription" in low:
        return "Inscription"

    if "coat of arms" in low:
        return "Coat of arms"

    return None

def extract_book(e: str) -> Optional[str]:

    prefixes = [
        "the drawing is in the manuscript ",
        "the drawing is in the cultural asset ",
        "it was extracted from the work: ",
        "it was extracted from ",
        "from the work ",
        "the title of the cultural asset is ",
    ]

    return find_after_prefix(e, prefixes)

def extract_answer(question: str, evidence_text: str, route: str = "", template_id: str = "") -> Optional[str]:

    q = clean_text(question).lower()
    e = clean_text(evidence_text)
    tpl = str(template_id or "").upper()

    if tpl == "AFFIXEDLANGUAGE" or "what is the language" in q:
        ans = extract_affixed_language(e)
        if ans:
            return strip_trailing_punct(ans)

    if tpl in {"AFFIXEDAUTHOR", "AUTHOR"} or "who is the author" in q or "who's the author" in q:
        ans = extract_author(e)
        if ans:
            return strip_trailing_punct(ans)

    if tpl == "AUTHORCRITERION" or "criterion is the cultural asset attributed" in q:
        ans = extract_authorcriterion(e)
        if ans:
            return ans

    if tpl == "AFFIXEDPOSITION" or "where in the cultural asset" in q:
        ans = extract_affixedposition(e)
        if ans:
            return ans

    if tpl == "AFFIXEDTRANSCRIPT" or "what does it say" in q or "written sentences say" in q:
        ans = extract_affixedtranscript(e)
        if ans:
            return ans

    if tpl == "AFFIXEDTECHNIQUE" or "technical characteristics of the element" in q:
        ans = extract_affixedtechnique(e)
        if ans:
            return ans

    if tpl == "AFFIXEDELEMENT" or "what is the element on the cultural asset" in q:
        ans = extract_affixedelement(e)
        if ans:
            return ans

    if tpl == "BOOK" or "what manuscript" in q or "where was it extracted from" in q:
        ans = extract_book(e)
        if ans:
            return strip_trailing_punct(ans)

    return None
