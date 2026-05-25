"""Lightweight runtime helpers for deployed inference."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

try:
    from preprocessing import build_ohe_vectorizer, clean_text
except ImportError:  # pragma: no cover
    from .preprocessing import build_ohe_vectorizer, clean_text


def _sentence_split(article: str) -> list[str]:
    """Split article into rough sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(article)) if s.strip()]


def _extract_subject_predicate(sentence: str) -> tuple[str, str]:
    """Heuristically split a sentence into subject and predicate."""
    words = sentence.strip().split()
    if not words:
        return ("", sentence)
    subject = words[0]
    predicate = " ".join(words[1:]) if len(words) > 1 else ""
    return subject, predicate


def _to_base_form(verb: str) -> str:
    """Approximate a base verb form without external NLP dependencies."""
    irregulars = {
        "visited": "visit", "borrowed": "borrow", "gave": "give",
        "went": "go", "came": "come", "saw": "see", "read": "read",
        "was": "be", "were": "be", "had": "have", "did": "do",
        "got": "get", "took": "take", "made": "make", "said": "say",
        "told": "tell", "found": "find", "left": "leave", "kept": "keep",
    }
    v = verb.lower()
    if v in irregulars:
        return irregulars[v]
    if v.endswith("ed") and len(v) > 4:
        if len(v) > 5 and v[-3] == v[-4]:
            return v[:-3]
        return v[:-2]
    if v.endswith("ing") and len(v) > 5:
        return v[:-3]
    if v.endswith("s") and len(v) > 3 and not v.endswith("ss"):
        return v[:-1]
    return v


def _make_template(sentence: str) -> list[str]:
    """Create simple template questions from a declarative sentence."""
    cleaned = sentence.strip().rstrip(".!?")
    words = cleaned.split()
    if not words:
        return []

    subject, predicate = _extract_subject_predicate(cleaned)
    pred_words = predicate.split()
    lowered = cleaned.lower()
    tokens = set(clean_text(lowered).split())

    questions: list[str] = []
    if pred_words:
        base_verb = _to_base_form(pred_words[0])
        rest = " ".join(pred_words[1:]) if len(pred_words) > 1 else ""
        if rest:
            questions.append(f"What did {subject} {base_verb} {rest}?")
        else:
            questions.append(f"What did {subject} {base_verb}?")
    else:
        questions.append(f"What is {subject}?")

    person_refs = {
        "he", "she", "they", "mr", "mrs", "ms", "teacher", "student",
        "boy", "girl", "man", "woman", "her", "his", "their",
    }
    location_refs = {
        "school", "city", "room", "house", "park", "country", "village",
        "street", "library", "museum", "hospital", "store", "market",
    }
    time_refs = {
        "today", "yesterday", "tomorrow", "morning", "evening", "night",
        "year", "month", "day", "week", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday",
    }

    if tokens.intersection(person_refs) and pred_words:
        questions.append(f"Who {pred_words[0]} {' '.join(pred_words[1:])}?".strip())
    if tokens.intersection(location_refs):
        questions.append(f"Where did {subject} go?")
    if tokens.intersection(time_refs):
        questions.append(f"When did {subject} do this?")

    seen: set[str] = set()
    unique: list[str] = []
    for question in questions:
        question = question.strip()
        if question and question not in seen:
            seen.add(question)
            unique.append(question)
    return unique


def generate_questions(article: str, correct_answer: str) -> list[tuple[str, float]]:
    """Generate template-based questions ranked by text similarity."""
    sentences = _sentence_split(article)
    if not sentences:
        return [("What is the main idea of the passage?", 0.0)]
    vectorizer = build_ohe_vectorizer(max_features=5000)
    try:
        matrix = vectorizer.fit_transform(sentences + [correct_answer])
        scores = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    except Exception:
        scores = np.zeros(len(sentences), dtype=float)
    ranked_sentence_idx = np.argsort(scores)[::-1][:3]
    candidates: list[tuple[str, float]] = []
    for idx in ranked_sentence_idx:
        for question in _make_template(sentences[int(idx)]):
            candidates.append((question, float(scores[int(idx)])))
    if not candidates:
        candidates.append(("What is supported by the passage?", 0.0))
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def extract_candidate_phrases(article: str, correct_answer: str) -> list[str]:
    """Extract plausible distractor phrases from the passage."""
    correct_cleaned = clean_text(correct_answer)
    correct_words = set(correct_cleaned.split())
    raw_words = str(article).split()
    multi_word: list[str] = []
    for n in (3, 2):
        for i in range(len(raw_words) - n + 1):
            chunk = raw_words[i : i + n]
            chunk_clean = [clean_text(w) for w in chunk]
            if any(w in ENGLISH_STOP_WORDS or len(w) < 2 for w in chunk_clean):
                continue
            phrase = " ".join(chunk_clean)
            phrase_words = set(phrase.split())
            if phrase_words.intersection(correct_words):
                continue
            if phrase and phrase != correct_cleaned:
                multi_word.append(phrase)
    seen: set[str] = set()
    unique_multi: list[str] = []
    for phrase in multi_word:
        if phrase not in seen:
            seen.add(phrase)
            unique_multi.append(phrase)
    tokens = [
        clean_text(token)
        for token in str(article).split()
        if clean_text(token) not in ENGLISH_STOP_WORDS
        and len(clean_text(token)) > 2
        and clean_text(token) not in correct_words
    ]
    unique_tokens = list(dict.fromkeys(tokens))
    return unique_multi + [token for token in unique_tokens if token not in seen]


def frequency_based_substitution(article: str, correct_answer: str) -> list[str]:
    """Generate candidates from words with similar article frequency."""
    tokens = [t for t in clean_text(article).split() if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    counts = Counter(tokens)
    answer_words = [w for w in clean_text(correct_answer).split() if w]
    if not answer_words or not counts:
        return []
    answer_freq = np.mean([counts.get(w, 1) for w in answer_words])
    return [
        word for word, freq in counts.items()
        if word not in answer_words and abs(freq - answer_freq) <= max(1.0, answer_freq)
    ][:20]


def _candidate_similarity(
    candidates: list[str],
    correct_answer: str,
    vectorizer: Optional[CountVectorizer],
) -> tuple[np.ndarray, object]:
    """Vectorize candidates and compute cosine similarity to the answer."""
    texts = candidates + [correct_answer]
    vec = vectorizer or build_ohe_vectorizer(max_features=5000)
    try:
        matrix = vec.transform(texts) if hasattr(vec, "vocabulary_") else vec.fit_transform(texts)
    except ValueError:
        vec = CountVectorizer(binary=True)
        matrix = vec.fit_transform(texts)
    sims = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    return sims, matrix[:-1]


def rank_distractors(
    candidates: list[str],
    correct_answer: str,
    vectorizer: Optional[CountVectorizer],
    top_k: int = 3,
) -> list[str]:
    """Rank distractors by moderate similarity plus diversity."""
    cleaned_candidates = [
        candidate for candidate in dict.fromkeys(candidates)
        if clean_text(candidate) != clean_text(correct_answer) and candidate.strip()
    ]
    if not cleaned_candidates:
        return ["Another option", "A different answer", "None of the above"][:top_k]

    sims, matrix = _candidate_similarity(cleaned_candidates, correct_answer, vectorizer)
    medium_bonus = np.where((sims >= 0.05) & (sims <= 0.65), 1.0, 0.2)
    base_scores = medium_bonus - np.abs(sims - 0.35)

    selected: list[int] = []
    for _ in range(min(top_k, len(cleaned_candidates))):
        best_idx, best_score = -1, -np.inf
        for idx, score in enumerate(base_scores):
            if idx in selected:
                continue
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = float(np.max(cosine_similarity(matrix[idx], matrix[selected]).ravel()))
            final_score = float(score) - 0.35 * diversity_penalty
            if final_score > best_score:
                best_idx, best_score = idx, final_score
        if best_idx >= 0:
            selected.append(best_idx)

    results = [cleaned_candidates[idx].title() for idx in selected]
    fallbacks = ["Another option", "A different answer", "None of the above"]
    while len(results) < top_k:
        results.append(fallbacks[len(results) % len(fallbacks)])
    return results[:top_k]


def _overlap(left: str, right: str) -> float:
    """Compute token overlap ratio."""
    left_set = set(clean_text(left).split())
    right_set = set(clean_text(right).split())
    return len(left_set.intersection(right_set)) / max(1, min(len(left_set), len(right_set)))


def generate_hints(article: str, question: str, correct_answer: str) -> list[str]:
    """Generate three graduated hints ordered from vague to explicit."""
    sentences = _sentence_split(article)
    if not sentences:
        return [
            "Review the passage topic.",
            "Look for details related to the question.",
            f"Consider: {correct_answer}",
        ]

    vectorizer = build_ohe_vectorizer(max_features=5000)
    try:
        matrix = vectorizer.fit_transform(sentences + [question])
        sims = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    except Exception:
        sims = np.asarray([_overlap(sentence, question) for sentence in sentences], dtype=float)

    ranked_idx = np.argsort(sims)
    low_idx = int(ranked_idx[0])
    mid_idx = int(ranked_idx[len(ranked_idx) // 2])
    high_idx = int(ranked_idx[-1])

    used: set[int] = set()
    hint_indices: list[int] = []
    for idx in [low_idx, mid_idx, high_idx]:
        if idx not in used:
            hint_indices.append(idx)
            used.add(idx)
        else:
            for fallback in ranked_idx:
                if int(fallback) not in used:
                    hint_indices.append(int(fallback))
                    used.add(int(fallback))
                    break

    hints = [sentences[i] for i in hint_indices[:3]]
    answer_words = set(clean_text(correct_answer).split())
    if answer_words and not answer_words.intersection(clean_text(hints[-1]).split()):
        hints[-1] = f"{hints[-1]} Focus on the part of the passage mentioning '{correct_answer}'."

    fallback_hints = [
        "Review the main topic of the passage.",
        "Look for the sentence most closely related to the question.",
        f"Focus on the part of the passage mentioning '{correct_answer}'.",
    ]
    while len(hints) < 3:
        hints.append(fallback_hints[len(hints)])
    return hints[:3]
