"""Unified inference API for the RACE Quiz AI system."""

from __future__ import annotations

import os
import re
import random
import time
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from preprocessing import extract_lexical_features
    from runtime_helpers import (
        extract_candidate_phrases,
        frequency_based_substitution,
        generate_hints,
        generate_questions,
        rank_distractors,
    )
except ImportError:  # pragma: no cover
    from .preprocessing import extract_lexical_features
    from .runtime_helpers import (
        extract_candidate_phrases,
        frequency_based_substitution,
        generate_hints,
        generate_questions,
        rank_distractors,
    )

RANDOM_STATE = 42


class RaceInferenceEngine:
    """Load trained artifacts and expose answer verification plus generation APIs."""

    def __init__(self, model_dir: str = "models") -> None:
        """Initialize the engine and fall back to demo mode if artifacts are missing.

        Args:
            model_dir: Root directory containing model_a and model_b folders.
        """
        self.model_dir = model_dir
        self.use_demo_mode = False
        self.lr_model: Optional[Any] = None
        self.svm_model: Optional[Any] = None
        self.ohe_vectorizer: Optional[Any] = None
        self.model_b_artifacts: dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load saved models and vectorizers with graceful demo fallback."""
        model_a_dir = os.path.join(self.model_dir, "model_a", "traditional")
        model_b_dir = os.path.join(self.model_dir, "model_b", "traditional")
        project_root = os.path.abspath(os.path.join(self.model_dir, os.pardir))
        processed_dir = os.path.join(project_root, "data", "processed")
        model_a_ready = False
        model_b_ready = False
        try:
            self.lr_model = joblib.load(os.path.join(model_a_dir, "logistic_regression.pkl"))
            self.svm_model = joblib.load(os.path.join(model_a_dir, "linear_svc_calibrated.pkl"))
        except Exception as exc:
            print(f"Model A classifiers unavailable; demo mode enabled: {exc}")
        try:
            self.ohe_vectorizer = joblib.load(os.path.join(processed_dir, "ohe_vectorizer.pkl"))
        except Exception as exc:
            print(f"Processed vectorizer unavailable; trying Model B fallback: {exc}")
        try:
            self.model_b_artifacts = joblib.load(os.path.join(model_b_dir, "model_b_artifacts.pkl"))
            if self.ohe_vectorizer is None and "vectorizer" in self.model_b_artifacts:
                self.ohe_vectorizer = self.model_b_artifacts["vectorizer"]
            model_b_ready = True
        except Exception as exc:
            print(f"Model B artifacts unavailable; rule-based generation enabled: {exc}")
        model_a_ready = (
            self.lr_model is not None
            and self.svm_model is not None
            and self.ohe_vectorizer is not None
        )
        self.use_demo_mode = not (model_a_ready and model_b_ready and self.ohe_vectorizer is not None)

    @staticmethod
    def _combined_text(article: str, question: str, option: str) -> str:
        """Build the exact combined text format used during preprocessing."""
        return f"article {article} question {question} option {option}"

    def _vectorize(self, article: str, question: str, option: str) -> sparse.csr_matrix:
        """Vectorize one article-question-option sample.

        Args:
            article: Passage text.
            question: Question text.
            option: Candidate answer option.

        Returns:
            Sparse combined feature row.
        """
        if self.ohe_vectorizer is None:
            raise RuntimeError("OHE vectorizer is not loaded.")
        text = self._combined_text(article, question, option)
        X_ohe = self.ohe_vectorizer.transform([text])
        meta = pd.DataFrame([{"article": article, "question": question, "option": option}])
        lexical = sparse.csr_matrix(extract_lexical_features(meta))
        return sparse.hstack([X_ohe, lexical], format="csr")

    def _demo_score(self, article: str, question: str, option: str) -> float:
        """Compute a deterministic rule-based fallback score."""
        article_words = set(str(article).lower().split())
        question_words = set(str(question).lower().split())
        option_words = set(str(option).lower().split())
        overlap = len(article_words.intersection(option_words)) + 0.5 * len(question_words.intersection(option_words))
        denom = max(1, len(option_words))
        return float(max(0.01, min(0.99, overlap / denom / 2.0)))

    def verify_answer(self, article: str, question: str, option: str) -> float:
        """Estimate the probability that an option correctly answers the question.

        Args:
            article: Passage text.
            question: Question text.
            option: Candidate answer option.

        Returns:
            Probability score from 0.0 to 1.0.
        """
        if self.use_demo_mode or self.lr_model is None or self.svm_model is None:
            return self._demo_score(article, question, option)
        try:
            X = self._vectorize(article, question, option)
            probs = [self.lr_model.predict_proba(X)[:, 1][0], self.svm_model.predict_proba(X)[:, 1][0]]
            return float(np.mean(probs))
        except Exception as exc:
            print(f"Inference failed; using demo score: {exc}")
            return self._demo_score(article, question, option)

    def predict_correct_option(self, article: str, question: str, options_dict: dict[str, str]) -> tuple[str, float]:
        """Predict the best option label from A-D.

        Args:
            article: Passage text.
            question: Question text.
            options_dict: Mapping from option label to option text.

        Returns:
            Tuple of predicted label and confidence.
        """
        scores = {label: self.verify_answer(article, question, option) for label, option in options_dict.items()}
        best = max(scores, key=scores.get)
        return best, float(scores[best])

    def generate_question(self, article: str) -> str:
        """Generate the best reading-comprehension question for a passage.

        FIX: The seed answer used to be the first 8 raw words of the article,
        which included stop words and made the cosine ranking meaningless.  We
        now use the first complete sentence stripped of stop words so the
        ranking picks the most informative sentence in the passage.

        Args:
            article: Passage text.

        Returns:
            Generated question string.
        """
        # Split into sentences and use the first sentence as seed
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(article)) if s.strip()]
        if sentences:
            seed_answer = sentences[0]
        else:
            seed_answer = " ".join(str(article).split()[:10]) or "the passage"
        return generate_questions(article, seed_answer)[0][0]

    def generate_distractors(self, article: str, question: str, correct_answer: str) -> list[str]:
        """Generate three plausible distractors.

        Args:
            article: Passage text.
            question: Question text.
            correct_answer: Correct answer phrase.

        Returns:
            List of three distractor strings.
        """
        candidates = extract_candidate_phrases(article, correct_answer)
        candidates.extend(frequency_based_substitution(article, correct_answer))
        vectorizer = self.ohe_vectorizer or self.model_b_artifacts.get("vectorizer")
        distractors = rank_distractors(candidates, correct_answer, vectorizer, top_k=3)
        while len(distractors) < 3:
            distractors.append(f"alternative {len(distractors) + 1}")
        return distractors[:3]

    def generate_hints(self, article: str, question: str, correct_answer: str) -> list[str]:
        """Generate three graduated hints.

        Args:
            article: Passage text.
            question: Question text.
            correct_answer: Correct answer phrase.

        Returns:
            Three hint strings.
        """
        return generate_hints(article, question, correct_answer)

    def _choose_demo_answer(self, article: str) -> str:
        """Choose a meaningful answer phrase when no answer is supplied.

        FIX: The original version returned the first few raw words of the
        article (e.g. "Sara visited the library after") which are not useful
        answer phrases.  We now pick the most frequent meaningful content word
        from the first sentence, which is much more likely to be a plausible
        answer target (proper noun, place, or key noun).

        Args:
            article: Passage text.

        Returns:
            A short, meaningful answer phrase.
        """
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        sentences = [s.strip() for s in str(article).replace("?", ".").split(".") if s.strip()]
        if not sentences:
            return "the passage"

        # Use the first sentence as the answer source
        first = sentences[0]
        # Pick content words (non-stop, len > 2)
        words = [
            w.strip(".,!?\"'")
            for w in first.split()
            if w.lower().strip(".,!?\"'") not in ENGLISH_STOP_WORDS
            and len(w.strip(".,!?\"'")) > 2
        ]
        if words:
            # Return the last meaningful word (often an object/location)
            return words[-1]
        return first.split()[0] if first.split() else "the passage"

    def run_full_pipeline(
        self,
        article: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a full quiz package for the UI.

        FIX: The correct_answer is now title-cased to be visually consistent
        with the distractor options (which are also title-cased by rank_distractors).
        The shuffle seed is randomised per call so the correct answer does not
        always land in the same slot.

        Args:
            article: Passage text.
            question: Optional existing question (e.g. from RACE dataset).
            answer: Optional correct answer (e.g. from RACE dataset).

        Returns:
            Dictionary with question, options, correct label, hints,
            confidence, inference_time_ms, and demo_mode flag.
        """
        start = time.perf_counter()
        if not article or not article.strip():
            raise ValueError("Article text is required.")

        generated_question = question or self.generate_question(article)
        correct_answer = answer or self._choose_demo_answer(article)

        # Normalise capitalisation so correct answer looks like an option
        correct_answer_display = correct_answer.strip().title()

        distractors = self.generate_distractors(article, generated_question, correct_answer)

        option_values = [correct_answer_display] + distractors
        # Use a time-based seed so the correct option is not always in slot A
        rng = random.Random(int(time.perf_counter() * 1e6) % (2**31))
        rng.shuffle(option_values)
        labels = ["A", "B", "C", "D"]
        options = dict(zip(labels, option_values))
        correct_label = next(
            label for label, value in options.items()
            if value == correct_answer_display
        )

        _, confidence = self.predict_correct_option(article, generated_question, options)

        return {
            "question": generated_question,
            "options": options,
            "correct": correct_label,
            "correct_answer": correct_answer_display,
            "hints": self.generate_hints(article, generated_question, correct_answer),
            "confidence": confidence,
            "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "demo_mode": self.use_demo_mode,
        }


def main() -> None:
    """Run a small smoke test from the command line."""
    engine = RaceInferenceEngine()
    article = "Maria studied the passage carefully at school. She answered the question after reading every sentence."
    result = engine.run_full_pipeline(article)
    print(result)


if __name__ == "__main__":
    main()
