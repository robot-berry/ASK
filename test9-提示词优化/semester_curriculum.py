"""Validated per-request semester selection and immutable prompt snapshots."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def catalog():
    return json.loads((ROOT / "curriculum_terms.json").read_text(encoding="utf-8"))

def validate_term(term):
    if term and term not in {item["id"] for item in catalog()}:
        raise ValueError("请选择有效的年级和学期")
    return term

def snapshot(subject, term):
    validate_term(term)
    if not term:
        return None
    subject = "physics" if subject == "physics" else "math"
    item = next(item for item in catalog() if item["id"] == term)
    text = (ROOT / "curriculum_prompts" / subject / (term + ".md")).read_text(encoding="utf-8")
    # Physics explanations must also respect same-semester mathematics.
    if subject == "physics":
        text += "\n本学期允许的数学工具上限：" + "；".join(item["subjects"]["math"]["allowed"]) + "。\n"
    return {"id":term, "label":item["label"], "stage":item["stage"],
            "subject":subject, "prompt":text,
            "revision":hashlib.sha256(text.encode("utf-8")).hexdigest()}

def apply_scope(inferred, selected):
    if not selected:
        return inferred
    return {**inferred, "stage":selected["stage"], "reason":"explicit_semester_selection",
            "semester": selected["id"], "semester_label":selected["label"]}
