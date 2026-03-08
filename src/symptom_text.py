from __future__ import annotations

import re
from difflib import get_close_matches


# Colloquial phrases -> likely clinical symptom keys.
PHRASE_TO_SYMPTOMS: dict[str, list[str]] = {
    "cold": ["continuous_sneezing", "cough", "chills", "runny_nose"],
    "common cold": ["continuous_sneezing", "cough", "chills", "runny_nose"],
    "feeling sick": ["fatigue", "weakness", "nausea", "vomiting"],
    "feel sick": ["fatigue", "weakness", "nausea", "vomiting"],
    "not feeling well": ["fatigue", "weakness", "malaise"],
    "under the weather": ["fatigue", "weakness"],
    "body pain": ["muscle_pain", "joint_pain", "back_pain"],
    "throat pain": ["sore_throat"],
    "chest pain": ["chest_pain"],
    "breathing difficulty": ["breathlessness", "difficulty_breathing"],
    "shortness of breath": ["breathlessness", "difficulty_breathing"],
    "high temperature": ["fever"],
    "tired": ["fatigue", "weakness"],
}


def normalize_symptom_token(text: str) -> str:
    token = text.strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def extract_symptoms_from_text(text: str, symptom_vocab: list[str]) -> tuple[list[str], list[str]]:
    vocab = set(symptom_vocab)
    matched: set[str] = set()
    unknown: list[str] = []

    normalized_text = normalize_symptom_token(text).replace("_", " ")
    padded_text = f" {normalized_text} "

    # 1) Colloquial phrase expansion.
    for phrase, mapped_symptoms in PHRASE_TO_SYMPTOMS.items():
        if f" {phrase} " in padded_text:
            for symptom in mapped_symptoms:
                if symptom in vocab:
                    matched.add(symptom)

    # 2) Chunk-based direct/fuzzy mapping.
    raw_chunks = re.split(r"[,;\n]|\\band\\b|\\bwith\\b|\\balso\\b", text.lower())
    for chunk in raw_chunks:
        normalized = normalize_symptom_token(chunk)
        if not normalized:
            continue
        if normalized in vocab:
            matched.add(normalized)
            continue
        close = get_close_matches(normalized, symptom_vocab, n=1, cutoff=0.82)
        if close:
            matched.add(close[0])
        else:
            unknown.append(normalized)

    # 3) N-gram phrase scan against known symptom vocabulary.
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w]
    for n in range(1, 5):
        for i in range(0, max(0, len(words) - n + 1)):
            candidate = "_".join(words[i : i + n])
            if candidate in vocab:
                matched.add(candidate)

    # 4) Explicit symptom phrase lookup in full text.
    for symptom in symptom_vocab:
        phrase = symptom.replace("_", " ")
        if f" {phrase} " in padded_text:
            matched.add(symptom)

    return sorted(matched), sorted(set(unknown))

