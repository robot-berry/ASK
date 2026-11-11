"""Pure helpers for recovering one complete question from full-page local OCR."""

from __future__ import annotations

import re


_QUESTION_START = re.compile(
    r"(?m)^\s*([1-9]\d{0,2})"
    r"(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*"
    # Composite exam items are sometimes printed as ``14-I（1）（2分）`` or
    # with the subquestion on the next line.  The former gate only accepted a
    # score marker immediately after the number, so the complete structured
    # OCR result could not be matched back to the truncated primary text.
    r"(?:[.．。、:：](?!\d)|(?=[（(]\s*\d+\s*(?:分|[）)])))"
)
_OPTION_MARKER = re.compile(r"(?m)(?:^|\s)([A-D])[.．、:：]\s*")


def first_question_number(text: str) -> int | None:
    match = _QUESTION_START.search(text or "")
    return int(match.group(1)) if match else None


def extract_numbered_question(text: str, target_number: int | None) -> str:
    """Extract the target top-level question and stop before the next one."""

    value = str(text or "").strip()
    if not value or target_number is None:
        return ""
    matches = list(_QUESTION_START.finditer(value))
    candidates: list[str] = []
    for index, match in enumerate(matches):
        if int(match.group(1)) != target_number:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        candidates.append(value[match.start():end].strip())
    if not candidates:
        return ""
    return max(
        candidates,
        key=lambda candidate: (
            len(choice_labels(candidate)),
            len(re.sub(r"\s+", "", candidate)),
        ),
    )


def choice_labels(text: str) -> list[str]:
    return sorted({match.group(1) for match in _OPTION_MARKER.finditer(text or "")})


def repair_missing_choice_labels(text: str) -> str:
    """Restore labels only when four option positions are already observable.

    Formula OCR often recognizes the formula to the left of B/D but misses the
    small A/C glyph.  The two-column row itself provides deterministic reading
    order, so no answer or subject knowledge is used here.
    """

    value = str(text or "").strip()
    # Formula OCR can wrap the printed option label into the same math span as
    # an isotope/vector expression, for example ``$A _ {2}^{4}He$``.  The
    # leading A-D is still an observable line label; move only that glyph out
    # of the math span and keep the formula payload unchanged.
    value = "\n".join(
        re.sub(
            r"^(\s*)\$([A-D])(?=\s*[_^])\s*",
            r"\1\2. $",
            line,
        )
        for line in value.splitlines()
    )
    labels = set(choice_labels(value))
    lines = value.splitlines()
    choice_cue_present = bool(re.search(
        r"(?:下列|以上|其中).{0,24}(?:正确|错误|符合|不正确|选项)",
        value,
    ))
    if len(labels) >= 2:
        normalized_lines: list[str] = []
        for line in lines:
            missing_punctuation = re.match(
                r"^(\s*)([A-D])(?=\s*(?:\$|[\u4e00-\u9fff]|[（(\-+0-9]))",
                line,
            )
            if missing_punctuation and missing_punctuation.group(2) not in labels:
                line = (
                    missing_punctuation.group(1)
                    + missing_punctuation.group(2)
                    + ". "
                    + line[missing_punctuation.end():].lstrip()
                )
                labels.add(missing_punctuation.group(2))
            normalized_lines.append(line)
        lines = normalized_lines
    repaired: list[str] = []
    for line in lines:
        current = line
        if "A" not in labels and "B" in labels:
            match = re.match(
                r"^\s*((?:\$[^$\n]+\$\s*)+)B[.．、:：]\s*(.*)$",
                current,
            )
            if match:
                current = f"A. {match.group(1).strip()} B. {match.group(2).strip()}"
                labels.add("A")
        if "C" not in labels and "D" in labels:
            match = re.match(
                r"^\s*((?:\$[^$\n]+\$\s*)+)D[.．、:：]\s*(.*)$",
                current,
            )
            if match:
                current = f"C. {match.group(1).strip()} D. {match.group(2).strip()}"
                labels.add("C")
        repaired.append(current)
    value = "\n".join(repaired)
    labels = set(choice_labels(value))

    if {"A", "C", "D"}.issubset(labels) and "B" not in labels:
        lines = value.splitlines()
        a_index = next(
            (index for index, line in enumerate(lines) if re.match(r"^\s*A[.．、:：]", line)),
            -1,
        )
        c_index = next(
            (index for index, line in enumerate(lines) if re.match(r"^\s*C[.．、:：]", line)),
            -1,
        )
        between = [
            index
            for index in range(a_index + 1, c_index)
            if lines[index].strip()
        ] if 0 <= a_index < c_index else []
        if len(between) == 1:
            lines[between[0]] = "B. " + lines[between[0]].strip()
            value = "\n".join(lines)
            labels = set(choice_labels(value))

    if {"A", "B", "D"}.issubset(labels) and "C" not in labels:
        lines = value.splitlines()
        b_index = next(
            (index for index, line in enumerate(lines) if re.match(r"^\s*B[.．、:：]", line)),
            -1,
        )
        d_index = next(
            (index for index, line in enumerate(lines) if re.match(r"^\s*D[.．、:：]", line)),
            -1,
        )
        between = [
            index
            for index in range(b_index + 1, d_index)
            if lines[index].strip()
        ] if 0 <= b_index < d_index else []
        if 1 <= len(between) <= 3:
            lines[between[0]] = "C. " + lines[between[0]].strip()
            value = "\n".join(lines)
            labels = set(choice_labels(value))

    # When OCR loses the printed labels beside handwritten ticks, use the
    # observable choice instruction and the intact C/D tail as boundaries.
    # This only restores labels; it never supplies or changes option content.
    if {"C", "D"}.issubset(labels) and not ({"A", "B"} & labels):
        cue_line = next(
            (
                index
                for index, line in enumerate(value.splitlines())
                if re.search(
                    r"(?:下列|以上|其中).{0,24}(?:正确|错误|符合|不正确|选项)",
                    line,
                )
            ),
            -1,
        )
        lines = value.splitlines()
        c_index = next(
            (
                index
                for index, line in enumerate(lines)
                if re.match(r"^\s*C[.．、:：]", line)
            ),
            -1,
        )
        between = [
            index
            for index in range(cue_line + 1, c_index)
            if lines[index].strip()
        ] if 0 <= cue_line < c_index else []
        if len(between) == 2:
            lines[between[0]] = "A. " + lines[between[0]].strip()
            lines[between[1]] = "B. " + lines[between[1]].strip()
            value = "\n".join(lines)
            labels = set(choice_labels(value))

    if {"B", "C", "D"}.issubset(labels) and "A" not in labels:
        lines = value.splitlines()
        cue_line = next(
            (
                index
                for index, line in enumerate(lines)
                if re.search(
                    r"(?:下列|以上|其中).{0,24}(?:正确|错误|符合|不正确|选项)",
                    line,
                )
            ),
            -1,
        )
        b_index = next(
            (
                index
                for index, line in enumerate(lines)
                if re.match(r"^\s*B[.．、:：]", line)
            ),
            -1,
        )
        between = [
            index
            for index in range(cue_line + 1, b_index)
            if lines[index].strip()
        ] if 0 <= cue_line < b_index else []
        if len(between) == 1:
            lines[between[0]] = "A. " + lines[between[0]].strip()
            value = "\n".join(lines)
            labels = set(choice_labels(value))
        elif b_index > 0:
            previous = lines[b_index - 1].strip()
            previous_cjk = sum("\u4e00" <= char <= "\u9fff" for char in previous)
            if (
                4 <= previous_cjk <= 48
                and not re.match(r"^\s*(?:\d+[.．、]|[（(]\d+[）)]|第.+题图)", previous)
                and not previous.endswith(("，", ",", "：", ":", "；", ";"))
            ):
                lines[b_index - 1] = "A. " + previous
                value = "\n".join(lines)
                labels = set(choice_labels(value))

    # A complete B/C/D tail with one short formula-only line immediately above
    # has an observable missing A label.  This covers formula-only choices
    # without interpreting the formula or guessing its value.
    if labels == {"B", "C", "D"}:
        lines = value.splitlines()
        b_index = next(
            (
                index
                for index, line in enumerate(lines)
                if re.match(r"^\s*B[.．、:：]", line)
            ),
            -1,
        )
        if b_index > 0:
            previous = lines[b_index - 1].strip()
            if (
                "$" in previous
                and not any("\u4e00" <= char <= "\u9fff" for char in previous)
                and len(re.sub(r"\s+", "", previous)) <= 160
            ):
                lines[b_index - 1] = "A. " + previous
                value = "\n".join(lines)

    if "已知集合" in value and "全集" in value:
        value = re.sub(r"(?m)(^|\s)A[.．、:：]\s*0(?=\s+B[.．、:：])", r"\1A. ∅", value)
    return value.strip()


def extract_complete_choice_tail(text: str) -> str:
    """Return an A-D option tail after observable label repair."""

    value = repair_missing_choice_labels(text)
    lines = value.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if re.match(r"^\s*A[.．、:：]", line)),
        -1,
    )
    if start < 0:
        return ""
    end = next(
        (
            index
            for index in range(start, len(lines))
            if re.match(r"^\s*D[.．、:：]", lines[index])
            or re.search(r"\sD[.．、:：]", lines[index])
        ),
        -1,
    )
    if end < start:
        return ""
    tail = "\n".join(lines[start:end + 1]).strip()
    return tail if choice_labels(tail) == ["A", "B", "C", "D"] else ""


def trim_after_complete_choice_question(text: str) -> str:
    """Drop a neighbouring question fragment after a completed A-D tail.

    Dataset page crops sometimes include the first one or two OCR lines of the
    next question while its number is outside the image. Once option D reaches
    sentence-ending punctuation, later free prose cannot belong to the
    completed choice question.
    """

    value = str(text or "").strip()
    if set(choice_labels(value)) != set("ABCD"):
        return value
    lines = [line.rstrip() for line in value.splitlines() if line.strip()]
    seen_labels: set[str] = set()
    d_position = -1
    for marker in _OPTION_MARKER.finditer(value):
        label = marker.group(1)
        if label == "D" and set("ABC").issubset(seen_labels):
            d_position = marker.start()
            break
        seen_labels.add(label)
    d_index = value.count("\n", 0, d_position) if d_position >= 0 else -1
    if d_index < 0:
        return value
    # A-D can also be a complete choice block inside part (1) of a compound
    # experiment question.  If an observable numbered/circled subquestion
    # follows option D, later prose belongs to the same top-level question and
    # must never be trimmed as neighbouring-page noise.
    after_first_option_tail = "\n".join(lines[d_index + 1:])
    if re.search(
        r"(?m)^\s*(?:[（(]\s*[1-9]\s*[）)]|[①②③④⑤⑥⑦⑧⑨⑩])",
        after_first_option_tail,
    ):
        return value
    # A visible next top-level question marker is decisive even when option D
    # lost its final Chinese period during OCR.
    for marker in _QUESTION_START.finditer(value):
        marker_index = value.count("\n", 0, marker.start())
        if marker.start() > d_position and marker_index > d_index:
            return value[: marker.start()].rstrip()
    for end_index in range(d_index, len(lines)):
        if re.search(r"[。！？!?]\s*$", lines[end_index]):
            if end_index + 1 < len(lines):
                return "\n".join(lines[: end_index + 1]).strip()
            break
    return value


def full_page_recovery_candidate(
    primary_text: str,
    full_page_text: str,
) -> tuple[str, str]:
    """Build a target-only structured candidate from a second local OCR pass."""

    target = first_question_number(primary_text)
    block = extract_numbered_question(full_page_text, target)
    if block:
        return repair_missing_choice_labels(block), "target_question_block"
    if target is None:
        markers = list(_QUESTION_START.finditer(str(full_page_text or "")))
        primary_labels = set(choice_labels(primary_text))
        # Formula-heavy single-row questions can leave RapidOCR with only the
        # A-D option tail.  When the full-page pass observes exactly one
        # numbered question and retains all already-seen options, use that
        # complete block instead of appending a duplicate option tail.
        if len(markers) == 1:
            candidate = str(full_page_text[markers[0].start():]).strip()
            candidate_labels = set(choice_labels(candidate))
            primary_cjk = sum("\u4e00" <= char <= "\u9fff" for char in primary_text)
            candidate_cjk = sum("\u4e00" <= char <= "\u9fff" for char in candidate)
            if (
                candidate
                and (not primary_labels or candidate_labels.issuperset(primary_labels))
                and candidate_cjk >= primary_cjk + 2
                and len(re.sub(r"\s+", "", candidate))
                >= max(18, len(re.sub(r"\s+", "", primary_text)))
            ):
                return repair_missing_choice_labels(candidate), "single_question_block"
        primary_compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", primary_text)
        primary_chars = {
            primary_compact[index:index + 2]
            for index in range(max(0, len(primary_compact) - 1))
        }
        ranked: list[tuple[float, int, str]] = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(full_page_text)
            candidate = str(full_page_text[marker.start():end]).strip()
            candidate_compact = re.sub(
                r"[^0-9A-Za-z\u4e00-\u9fff]", "", candidate
            )
            candidate_chars = {
                candidate_compact[position:position + 2]
                for position in range(max(0, len(candidate_compact) - 1))
            }
            shared = primary_chars & candidate_chars
            score = len(shared) / max(1, min(len(primary_chars), len(candidate_chars)))
            ranked.append((score, len(shared), candidate))
        if ranked:
            score, shared_count, candidate = max(ranked, key=lambda item: (item[0], item[1]))
            if score >= 0.45 and shared_count >= 10:
                return repair_missing_choice_labels(candidate), "matched_question_block"
    # Never append A-D a second time.  A complete primary option tail needs a
    # stem replacement (handled above), not another copy of the same choices.
    if set(choice_labels(primary_text)) == set("ABCD"):
        return "", ""
    tail = extract_complete_choice_tail(full_page_text)
    if tail:
        return primary_text.rstrip() + "\n" + tail, "complete_choice_tail"
    return "", ""
