"""Three preprocessing modes for Khmer short answers.

raw      → strip invisibles + strip whitespace
clean    → strip invisibles + KCC normalize + strip punctuation
segment  → clean + khmernltk.word_tokenize

Order applied in preprocess(): NFC → strip invisibles → KCC reorder → strip
punctuation → (segment). "Invisibles" = Unicode format/control characters
(zero-width space U+200B, ZWNJ/ZWJ, BOM, soft hyphen, controls) plus bullet
markers (• ◦ ‣ ▪ …) — genuine noise that survives punctuation stripping. Digits
(Arabic and Khmer) and letters are *kept* as answer content.

Note: the released headline results were produced with the pre-refinement
cleaning (which did not strip invisibles); experiments/exp11_cleaning_ablation.py
quantifies the (negligible) effect of this refinement on the classical champion.
"""

import re
import unicodedata


# Bullet / list-marker symbols treated as formatting noise (not punctuation,
# so not caught by strip_punctuation, and not Khmer content).
_NOISE_SYMBOLS = set("•◦‣▪…∙·")


def strip_invisibles(text: str) -> str:
    """Remove zero-width/format/control characters and bullet markers.

    Drops Unicode categories Cf (format: U+200B ZWSP, U+200C/200D, U+FEFF, U+00AD)
    and Cc (control), plus a small set of bullet/ellipsis symbols. Keeps digits,
    letters, Khmer text, and ordinary whitespace (collapsed later).
    """
    out = []
    for ch in text:
        if ch in _NOISE_SYMBOLS:
            continue
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Cc") and ch not in ("\n", "\t"):
            continue
        out.append(ch)
    return "".join(out)


KHMER_CONSONANTS = set(range(0x1780, 0x17A3))
KHMER_DEPENDENT_VOWELS = set(range(0x17B6, 0x17C6))
KHMER_SIGNS = set(range(0x17C6, 0x17D4))
KHMER_COENG = 0x17D2
KHMER_PUNCT = set(range(0x17D4, 0x17DB))
ASCII_PUNCT = set(ord(c) for c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

_khmernltk_warned = False


def kcc_normalize(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    clusters, current, prev_coeng = [], [], False
    for ch in text:
        cp = ord(ch)
        if cp in KHMER_CONSONANTS and not prev_coeng and current:
            if any(0x1780 <= ord(c) <= 0x17FF for c in current):
                clusters.append(current)
                current = []
        current.append(ch)
        prev_coeng = (cp == KHMER_COENG)
    if current:
        clusters.append(current)
    normalized = []
    for cluster in clusters:
        bases, coengs, vowels, signs, others = [], [], [], [], []
        i = 0
        while i < len(cluster):
            cp = ord(cluster[i])
            if cp in KHMER_CONSONANTS and not coengs and not vowels and not signs:
                bases.append(cluster[i])
            elif cp == KHMER_COENG and i + 1 < len(cluster):
                coengs += [cluster[i], cluster[i + 1]]
                i += 1
            elif cp in KHMER_DEPENDENT_VOWELS:
                vowels.append(cluster[i])
            elif cp in KHMER_SIGNS:
                signs.append(cluster[i])
            else:
                others.append(cluster[i])
            i += 1
        normalized.extend(bases + coengs + vowels + signs + others)
    return "".join(normalized)


def strip_punctuation(text: str) -> str:
    result = []
    for ch in text:
        cp = ord(ch)
        result.append(" " if cp in ASCII_PUNCT or cp in KHMER_PUNCT else ch)
    return re.sub(r"\s+", " ", "".join(result)).strip()


def segment_khmer(text: str) -> str:
    global _khmernltk_warned
    try:
        import khmernltk
        return " ".join(khmernltk.word_tokenize(text))
    except ImportError:
        if not _khmernltk_warned:
            print("NOTE: khmernltk not installed — segmentation skipped.")
            _khmernltk_warned = True
        return text
    except Exception:
        return text


def preprocess(text: str, mode: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Remove zero-width/format/control + bullet noise in every mode, then trim.
    text = re.sub(r"\s+", " ", strip_invisibles(text)).strip()
    if mode == "raw":
        return text
    text = strip_punctuation(kcc_normalize(text))
    if mode == "clean":
        return text
    if mode == "segment":
        return segment_khmer(text)
    raise ValueError(f"Unknown preprocess mode: {mode}")
