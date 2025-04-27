# agents/evaluator_agent.py
"""
evaluator_agent.py – Evidently-powered ‘LLM-as-Judge’ helper

Given a time-tagged transcript it:
1.  Parses utterances + roles.
2.  Creates simple heuristic “ground-truth” labels
    (sentiment, toxicity, supportiveness).
3.  Runs the three HF models already used in your project:
       • cardiffnlp/twitter-roberta-base-sentiment
       • unitary/toxic-bert
       • facebook/bart-large-mnli
4.  Uses a simple scoring function to compute a 0–10 score,
    and generates a brief human-readable explanation.
5.  Returns a dict per model:
      { model_name: {
          score: int,
          correct: [examples…],
          incorrect: [examples…],
          explanation: str
        }
      }
"""

import re
import json
import logging
from typing import List, Tuple, Dict, Any

import pandas as pd

try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # allow imports even if transformers missing

LOG = logging.getLogger("care_monitor")


def _parse_transcript(txt: str) -> Tuple[List[str], List[str]]:
    """
    Split the transcript into parallel lists of roles and utterances.
    Strips leading time tags like “[13:31]” or “(13:31)”.
    """
    roles, utts = [], []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        # remove timestamp
        line = re.sub(r'^[\[(]\d{1,2}:\d{2}(?:\s?[AP]M)?[\])]\s*', '', line)
        if line.lower().startswith("caregiver:"):
            roles.append("Caregiver")
            utts.append(line[len("Caregiver:"):].strip())
        elif line.lower().startswith("child:"):
            roles.append("Child")
            utts.append(line[len("Child:"):].strip())
        else:
            # continuation of previous speaker
            if utts:
                utts[-1] += " " + line
    return roles, utts


def _heuristic_labels(roles: List[str], utts: List[str]) -> Dict[str, List[str]]:
    """
    Create simple ground-truth labels:
      - sent: "POSITIVE" if a positive keyword appears, else "NEGATIVE"
      - tox:   "TOXIC" if caregiver uses insult keywords, else "NON_TOXIC"
      - sup:   "supportive" if caregiver uses caring keywords, else "unsupportive"
    """
    pos_words = {"love", "great", "happy", "thanks", "good job", "proud", "best"}
    tox_words = {"stupid", "shut up", "get over", "worthless", "hate", "idiot", "demanding", "whining", "complaints", "stop crying"}
    sup_words = {"love", "care", "proud", "well done", "good job", "hug", "support"}

    sent, tox, sup = [], [], []
    for role, u in zip(roles, utts):
        low = u.lower()
        # sentiment
        sent.append("POSITIVE" if any(w in low for w in pos_words) else "NEGATIVE")
        # toxicity only for caregiver
        if role == "Caregiver" and any(w in low for w in tox_words):
            tox.append("TOXIC")
        else:
            tox.append("NON_TOXIC")
        # supportiveness only for caregiver
        if role == "Caregiver" and any(w in low for w in sup_words):
            sup.append("supportive")
        else:
            sup.append("unsupportive")
    return {"sent": sent, "tox": tox, "sup": sup}


def evaluate_models(transcript: str) -> Dict[str, Dict[str, Any]]:
    """
    Main entrypoint – returns evaluation dict per model including:
      - score (0–10)
      - correct list
      - incorrect list
      - explanation string
    """
    roles, utts = _parse_transcript(transcript)
    if not utts:
        return {}

    truth = _heuristic_labels(roles, utts)

    # default predictions if transformers missing
    preds_sent = ["NEGATIVE"] * len(utts)
    preds_tox = ["NON_TOXIC"] * len(utts)
    preds_sup = ["unsupportive"] * len(utts)

    if pipeline:
        try:
            sent_pipe = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment"
            )
            out = sent_pipe(utts, truncation=True)
            preds_sent = [
                "POSITIVE" if ("POSITIVE" in r["label"].upper() or r["label"].endswith("2"))
                else "NEGATIVE"
                for r in out
            ]
        except Exception:
            LOG.exception("[Eval] sentiment pipeline failed")

        try:
            tox_pipe = pipeline(
                "text-classification",
                model="unitary/toxic-bert"
            )
            out = tox_pipe(utts, truncation=True)
            preds_tox = [
                "TOXIC" if "TOXIC" in r["label"].upper()
                else "NON_TOXIC"
                for r in out
            ]
        except Exception:
            LOG.exception("[Eval] toxicity pipeline failed")

        try:
            sup_pipe = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            temp = []
            for u in utts:
                res = sup_pipe(u, candidate_labels=["supportive", "unsupportive"], truncation=True)
                temp.append(res["labels"][0])
            preds_sup = temp
        except Exception:
            LOG.exception("[Eval] supportiveness pipeline failed")

    def _score(actual: List[str], pred: List[str], label_name: str) -> Tuple[int, List[str], List[str], str]:
        """
        Compute:
          - score = round(accuracy * 10)
          - lists of correct / incorrect utterances
          - a one-sentence explanation
        """
        correct, incorrect = [], []
        for u, a, p in zip(utts, actual, pred):
            if a == p:
                correct.append(f'"{u}"')
            else:
                incorrect.append(f'"{u}" (predicted: {p}, expected: {a})')

        total = len(actual)
        acc = len(correct) / total
        score = round(acc * 10)

        # Build a simple explanation
        if len(incorrect) == 0:
            explanation = f"Perfect! All {total}/{total} utterances correctly labeled for {label_name}."
        else:
            if label_name.lower().startswith("toxic"):
                # Over- vs. under- flag child lines
                over = sum(1 for ex in incorrect if "predicted: TOXIC" in ex and roles[utts.index(ex.split('"')[1])] == "Child")
                under = sum(1 for ex in incorrect if "predicted: NON_TOXIC" in ex and roles[utts.index(ex.split('"')[1])] == "Caregiver")
                if over > under:
                    explanation = f"Labeled {len(correct)}/{total} correctly; tended to over-flag child utterances as toxic."
                elif under > over:
                    explanation = f"Labeled {len(correct)}/{total} correctly; tended to under-flag caregiver toxicity."
                else:
                    explanation = f"Labeled {len(correct)}/{total} correctly; made mixed errors on both roles."
            else:
                explanation = f"Labeled {len(correct)}/{total} correctly for {label_name}; misclassified {len(incorrect)} utterances."

        return score, correct, incorrect, explanation

    # Score each model
    s_sent, corr_s, inc_s, expl_s = _score(truth["sent"], preds_sent, "Sentiment")
    s_tox, corr_t, inc_t, expl_t = _score(truth["tox"], preds_tox, "Toxicity")
    s_sup, corr_p, inc_p, expl_p = _score(truth["sup"], preds_sup, "Supportiveness")

    return {
        "cardiffnlp/twitter-roberta-base-sentiment (Sentiment)": {
            "score": s_sent,
            "correct": corr_s,
            "incorrect": inc_s,
            "explanation": expl_s
        },
        "unitary/toxic-bert (Toxicity)": {
            "score": s_tox,
            "correct": corr_t,
            "incorrect": inc_t,
            "explanation": expl_t
        },
        "facebook/bart-large-mnli (Supportiveness)": {
            "score": s_sup,
            "correct": corr_p,
            "incorrect": inc_p,
            "explanation": expl_p
        },
    }
