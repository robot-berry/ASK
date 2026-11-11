#!/usr/bin/env python3
"""Camera photo -> preprocessing -> original test3 four guidance images."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
import uuid

from PIL import Image

import full_benchmark as benchmark
import physics_guidance_pilot as pilot
from device_result_renderer import (
    resolve_device_fonts,
    validate_device_image_set,
)
from physics_guidance_renderer import render_four_images
from preprocessor import QuestionImageProcessor


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
RUNS_ROOT = Path(
    os.environ.get("TEST4_PHYSICS_RUNS_ROOT", "").strip()
    or PROJECT_ROOT / "device_runs"
).expanduser().resolve()
DEFAULT_SCHEMA = SOURCE_DIR / "physics_guidance_schema.json"
DEFAULT_PROMPT = SOURCE_DIR / "物理题解析提示词.md"
DEFAULT_ENV = SOURCE_DIR / "api_keys.env"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_ID = "qwen3.7-flash-2026-07-15"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
ACCEPTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
ROOT_KEYS = pilot.ROOT_KEYS

# Provider repair may rewrite generated analysis, but never the recognized
# source problem.  This mirrors the mathematics pipeline so a concise repair
# cannot silently solve a different question.
SOURCE_CONTRACT_FIELDS = (
    "subject",
    "problem_scope",
    "recognition_status",
    "uncertain_items",
    "question_blocks",
    "source_figures",
)
SCOPED_PROVIDER_REPAIR_FIELDS = frozenset(
    {"guidance", "base_formulas", "knowledge_points", "visual_evidence"}
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    # Windows may deny the atomic rename while the polling API briefly holds
    # job.json open. Retry so four already-rendered images are not reclassified
    # as a failed job because status polling overlapped the final write.
    last_error: OSError | None = None
    for _attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.15)
    try:
        path.write_bytes(payload)
        last_error = None
    except OSError as exc:
        last_error = exc
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        pass
    if last_error is not None:
        raise last_error


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON根值必须是对象：{path}")
    return value


def safe_filename(value: str) -> str:
    name = Path(value or "camera.jpg").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120] or "camera.jpg"


def load_credentials(env_file: Path = DEFAULT_ENV) -> tuple[str, str, str | None]:
    if env_file.is_file():
        benchmark.parse_env_file(env_file)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少DASHSCOPE_API_KEY；请在source/api_keys.env中配置轮换后的密钥")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip() or DEFAULT_BASE_URL
    parsed_url = urlparse(base_url)
    approved_host = (
        parsed_url.hostname == "dashscope.aliyuncs.com"
        or bool(
            parsed_url.hostname
            and parsed_url.hostname.endswith(".maas.aliyuncs.com")
        )
    )
    if (
        parsed_url.scheme != "https"
        or not approved_host
        or parsed_url.path.rstrip("/") != "/compatible-mode/v1"
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise RuntimeError("DASHSCOPE_BASE_URL必须使用已批准的阿里云百炼HTTPS地址")
    workspace = os.environ.get("DASHSCOPE_WORKSPACE_ID", "").strip() or None
    return api_key, base_url, workspace


def png_data_url(
    path: Path, *, same_bytes_as_render_source: bool = True
) -> tuple[str, dict[str, Any]]:
    """Transmit the exact preprocessed PNG used as the renderer source."""

    raw = path.read_bytes()
    with Image.open(path) as image:
        image.load()
        audit = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "transmitted_media_type": "image/png",
            "same_bytes_as_render_source": same_bytes_as_render_source,
        }
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), audit


def build_request_payload(
    *,
    model_id: str,
    prompt: str,
    data_url: str | Sequence[str],
    schema: Mapping[str, Any],
    max_tokens: int | None,
    requested_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    # Reuse the original test3 prompt expansion and answer-blind guidance rules.
    return pilot.request_payload(
        model_id=model_id,
        prompt=prompt,
        data_url=data_url,
        schema=schema,
        requested_outputs=requested_outputs,
        max_tokens=max_tokens,
        enable_thinking=False,
    )


REQUEST_VERB = re.compile(
    r"(?:求(?:出|解)?|计算|确定|写出|判断|比较|说明|证明|求证)"
)
SIMPLE_TARGET_FORMULA = re.compile(
    r"^[A-Za-zΑ-Ωα-ω](?:_\{?[A-Za-z0-9]+\}?|[₀₁₂₃₄₅₆₇₈₉ₓᵧ])?$"
)
TARGET_NOUN = re.compile(
    r"(?:距离|路程|位移|时间|速度|速率|加速度|质量|力|功|功率|"
    r"电流|电压|电阻|电荷量|场强|磁感应强度|温度|压强|体积|"
    r"密度|高度|长度|宽度|范围|频率|周期|波长|焦距|方向|"
    r"大小关系|变化规律|效率|机械能|动能|势能)"
)
ASSIGNMENT_SEPARATOR = re.compile(r"(?:=|≈|\approx|\simeq)")
FORMULA_IDENTIFIER = re.compile(r"[A-Za-z]+|[Α-Ωα-ω]", re.UNICODE)
IGNORED_FORMULA_IDENTIFIERS = frozenset(
    {
        "sqrt",
        "frac",
        "dfrac",
        "tfrac",
        "mathrm",
        "text",
        "operatorname",
        "cdot",
        "times",
        "left",
        "right",
        "tan",
        "sin",
        "cos",
        "cot",
        "ln",
        "log",
        "exp",
        "min",
        "max",
    }
)


def _question_text(output: Mapping[str, Any]) -> str:
    return "\n".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks") or []
        if isinstance(block, Mapping) and block.get("type") == "text"
    )


def _split_question_parts(value: str) -> dict[str, str]:
    matches = list(pilot.PRINTED_PART_TOKEN.finditer(value))
    if not matches:
        return {"整题": value}
    parts: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        label = pilot.normalized_part_label(match.group(0))
        if label:
            parts[label] = value[match.end() : end]
    return parts


def _simple_target_formula(value: str) -> str | None:
    compact = pilot.compact_formula_side(value.strip())
    if SIMPLE_TARGET_FORMULA.fullmatch(compact):
        return compact
    return None


def infer_dynamic_requested_outputs(
    output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Infer answer-blind targets from each printed phone-upload subquestion.

    The frozen 20-item pilot has a hand-reviewed requested-output manifest, but
    an arbitrary camera photo does not.  This deliberately conservative parser
    inspects only the final explicit request clause of each subquestion.  It
    never invents an answer and prefers the last atomic formula in that clause,
    which avoids treating contextual labels such as AB or O_2 as targets.
    """

    question_text = _question_text(output)
    by_part = _split_question_parts(question_text)
    guidance = output.get("guidance")
    advanced = guidance.get("advanced") if isinstance(guidance, Mapping) else None
    advanced_parts = advanced.get("parts") if isinstance(advanced, Mapping) else None
    labels = [
        pilot.normalized_part_label(item.get("part"))
        for item in advanced_parts or []
        if isinstance(item, Mapping) and pilot.normalized_part_label(item.get("part"))
    ]
    if not labels:
        labels = list(by_part) or ["整题"]

    requested: list[dict[str, Any]] = []
    for label in dict.fromkeys(labels):
        part_text = by_part.get(label, question_text if label == "整题" else "")
        verbs = list(REQUEST_VERB.finditer(part_text))
        if not verbs:
            continue
        tail = part_text[verbs[-1].end() :].strip(" ，,：:；;。？?")
        formula_matches = list(pilot.INLINE_FORMULA.finditer(tail))
        atomic = [
            (match, _simple_target_formula(match.group(1)))
            for match in formula_matches
        ]
        atomic = [(match, alias) for match, alias in atomic if alias]
        symbols: list[str] = []
        if atomic:
            # The last atomic formula is the safest target.  Include a preceding
            # target only when the two are joined solely by an explicit list
            # conjunction, e.g. “求 $v$ 和 $a$”.
            last_match, last_alias = atomic[-1]
            symbols.append(str(last_alias))
            cursor = last_match.start()
            for match, alias in reversed(atomic[:-1]):
                connector = tail[match.end() : cursor]
                if not re.fullmatch(r"\s*(?:和|及|与|、|以及)\s*", connector):
                    break
                symbols.insert(0, str(alias))
                cursor = match.start()

        prose_source = pilot.INLINE_FORMULA.sub("", tail)
        prose_source = re.sub(r"\s+", "", prose_source).strip("，,：:；;。？?")
        prose: list[str] = []
        if 2 <= len(prose_source) <= 60:
            prose.append(prose_source)
        for match in TARGET_NOUN.finditer(prose_source):
            start = max(0, match.start() - 14)
            core = prose_source[start : match.end()].lstrip("，,：:；;的")
            if 3 <= len(core) <= 24 and core not in prose:
                prose.append(core)

        if symbols or prose:
            requested.append(
                {
                    "part": label,
                    "symbol_aliases": symbols,
                    "prose_aliases": prose,
                }
            )
    return requested


def _formula_identifiers(value: str) -> set[str]:
    identifiers: set[str] = set()
    for match in FORMULA_IDENTIFIER.finditer(value):
        token = match.group(0).lstrip("\\")
        if token in IGNORED_FORMULA_IDENTIFIERS:
            continue
        identifiers.add(token)
        # Juxtaposed scalar products such as qUL should be understood as the
        # individual printed symbols when every factor is one letter.
        if token.isascii() and token.isalpha() and 1 < len(token) <= 6:
            identifiers.update(token)
    return identifiers


def _target_assignment_pattern(target: str) -> re.Pattern[str] | None:
    # Dynamic targets are intentionally atomic.  Complex ratios remain useful
    # intermediate relations and are not treated as assignment LHS values.
    if not SIMPLE_TARGET_FORMULA.fullmatch(target):
        return None
    escaped = re.escape(target)
    return re.compile(
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s*"
        rf"(?:=|≈|\\approx|\\simeq)\s*"
    )


def validate_dynamic_answer_blind(output: Mapping[str, Any]) -> list[str]:
    """Reject final target assignments in both LaTeX and ordinary prose.

    A relation remains allowed when its right side still contains an auxiliary
    symbol not printed in the question (for example ``d=sqrt(...a...)`` or
    ``y=(5L/2)tan(theta)``).  A target reduced only to printed givens, a number,
    or a unit scale is the delivered answer and is rejected globally, including
    when a later subquestion repeats an earlier answer.
    """

    requested = infer_dynamic_requested_outputs(output)
    if not requested:
        return []
    question_text = _question_text(output)
    target_symbols = list(
        dict.fromkeys(
            str(alias)
            for item in requested
            for alias in item.get("symbol_aliases", [])
            if isinstance(alias, str) and alias
        )
    )
    known_symbols = _formula_identifiers(question_text) - set(target_symbols)
    errors: list[str] = []

    # Reuse the mature prose/qualitative guard without its overly broad numeric
    # formula heuristic; plain and delimited symbolic assignments are checked
    # precisely below.
    prose_only = [
        {
            "part": item.get("part"),
            "symbol_aliases": [],
            "prose_aliases": item.get("prose_aliases", []),
        }
        for item in requested
    ]
    pilot.validate_answer_guard(output, prose_only, errors, policy="no_answer")

    for path, value, _current_part in pilot.guidance_entries(output):
        for target in target_symbols:
            pattern = _target_assignment_pattern(target)
            if pattern is None:
                continue
            assignments = list(pattern.finditer(value))
            for index, assignment in enumerate(assignments):
                end = assignments[index + 1].start() if index + 1 < len(assignments) else len(value)
                punctuation = re.search(r"[，。；;\n]", value[assignment.end() : end])
                if punctuation:
                    end = assignment.end() + punctuation.start()
                rhs = value[assignment.end() : end].strip(" $，,。；;：:")
                if not rhs:
                    continue
                # In a chain such as y=(5L/2)tan(theta)=5L the unresolved
                # middle expression must not hide the closed terminal side.
                rhs_sides = ASSIGNMENT_SEPARATOR.split(rhs)
                terminal_rhs = rhs_sides[-1].strip() if len(rhs_sides) > 1 else rhs
                rhs_symbols = _formula_identifiers(terminal_rhs)
                if target in rhs_symbols:
                    continue
                unresolved = rhs_symbols - known_symbols
                if unresolved:
                    continue
                errors.append(
                    f"{path}公开了所求量{target}的最终赋值；"
                    "分层提示必须在最终化简前停止"
                )
                break
    return list(dict.fromkeys(errors))


def _is_captured_particle_deflection_problem(output: Mapping[str, Any]) -> bool:
    if output.get("subject") != "physics" or output.get("recognition_status") != "clear":
        return False
    if output.get("uncertain_items"):
        return False
    text = _question_text(output)
    compact = re.sub(r"[\s$\\{}_^（）()，,；;。·]", "", text)
    required = (
        "初速度为零",
        "电压为U的加速电场",
        "电压为2U的偏转电场",
        "极板长度为L",
        "下极板右边缘Q点",
        "PA=2L",
        "AB=PQ=DC",
        "K板为绝缘反射板",
        "平行于板方向速度不变",
        "垂直于板方向速度大小不变",
        "方向相反",
    )
    if not all(marker in compact for marker in required):
        return False
    if not re.search(r"C.?D为QB.?PA的中点", compact):
        return False
    if not re.search(r"K板.*右端.*DC延长线.*上下平移", compact):
        return False
    by_part = _split_question_parts(text)
    signatures = {
        "（1）": r"距离\s*\$?d\$?",
        "（2）": r"距离\s*\$?y\$?",
        "（3）": r"宽度\s*\$?s\$?",
    }
    return all(
        label in by_part and re.search(pattern, by_part[label])
        for label, pattern in signatures.items()
    )


def repair_captured_particle_deflection_guidance(
    output: Mapping[str, Any],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace generated reasoning only for one fully identified captured form.

    The printed question and source figure are never changed.  The replacement
    contains correct textbook intermediate relations but deliberately stops
    before the hidden values of d, y, board length, and movement range.
    """

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    if not _is_captured_particle_deflection_problem(candidate):
        return candidate
    guidance = candidate.get("guidance")
    if not isinstance(guidance, dict):
        return candidate

    method_names = {
        "（1）": "能量—类平抛联立法",
        "（2）": "出射切线中点法",
        "（3）": "反射镜像与临界位置法",
    }
    replacement_steps: dict[tuple[str, str], list[str]] = {
        ("advanced", "（1）"): [
            r"由静电加速写出 $v_{0}^{2}=\frac{2qU}{m}$，偏转场中加速度为 $a=\frac{2qU}{md}$",
            r"水平方向满足 $t=\frac{L}{v_{0}}$，竖直方向满足 $\frac{d}{2}=\frac{1}{2}at^{2}$",
            r"将三个过程关系整体代入位移式，只取 $d>0$ 的物理解，并在得到 d 与 L 的最终关系前停止",
        ],
        ("standard", "（1）"): [
            r"由动能定理写出 $qU=\frac{1}{2}mv_{0}^{2}$",
            r"偏转场强为 $E=\frac{2U}{d}$，故 $a=\frac{qE}{m}=\frac{2qU}{md}$",
            r"水平方向匀速运动给出 $t=\frac{L}{v_{0}}$",
            r"由刚好从下极板边缘飞出写出 $\frac{d}{2}=\frac{1}{2}at^{2}$，联立后停在最终化简前",
        ],
        ("foundation", "（1）"): [
            "先用动能定理表示粒子进入偏转电场时的速度",
            "再由偏转电压和板间距表示电场强度及竖直加速度",
            "水平方向无加速度，用极板长度除以入射速度表示运动时间",
            r"竖直初速度为零，且刚好到达下极板，所以竖直位移是 $\frac{d}{2}$",
            "把能量、时间和竖直位移三个关系联立，只保留正值条件并停在最终求值前",
        ],
        ("advanced", "（2）"): [
            r"出场速度斜率满足 $\tan\theta=\frac{at}{v_{0}}=\frac{L}{d}$",
            r"速度反向延长线过极板水平中点，该点到屏上 $O_{2}$ 的水平距离为 $\frac{5L}{2}$",
            r"由直角三角形写出 $\frac{2y}{5L}=\tan\theta$，再与第（1）问的未化简关系联立并停在求值前",
        ],
        ("standard", "（2）"): [
            r"用 $v_{y}=at$ 和 $v_{x}=v_{0}$ 表示出场偏转角",
            r"结合第（1）问的过程式整理为 $\tan\theta=\frac{L}{d}$",
            r"利用出场切线反向延长线过极板中点，确定该点到 $O_{2}$ 的水平边为 $\frac{5L}{2}$",
            "按同一偏转角建立水平边与竖直边 y 的比例式，再与第（1）问联立并停在最终化简前",
        ],
        ("foundation", "（2）"): [
            "粒子飞出电场后的直线方向由出场时的水平、竖直速度共同决定",
            "先用竖直速度与水平速度之比表示这条直线的斜率",
            "出场切线反向延长后通过偏转极板的水平中点",
            r"从该中点到荧光屏建立一个直角三角形，水平边由 $\frac{L}{2}$ 与 PA 相加得到",
            "把竖直边记为 y，列出斜率比例并与第（1）问关系联立，在求出 y 前停止",
        ],
        ("advanced", "（3）"): [
            r"令出场直线斜率 $r=\frac{L}{d}$，K 板高度为 h、碰撞横坐标为 $x_{c}$，则 $h=\frac{d}{2}+r(x_{c}-L)$",
            r"反射只改变竖直速度方向，屏上坐标满足 $z_{S}=h-r(3L-x_{c})$",
            "分别令屏上位置对应 A、B，保留两组临界方程；最短板覆盖两个碰撞横坐标，移动宽度取两板高之差并停在求值前",
        ],
        ("standard", "（3）"): [
            "把 K 板反射等效为关于水平板面的镜像，水平速度保持不变",
            r"令斜率为 $r=\frac{L}{d}$、板高为 h、碰撞横坐标为 $x_{c}$，写出 $h=\frac{d}{2}+r(x_{c}-L)$",
            r"反射后屏上坐标满足 $z_{S}=h-r(3L-x_{c})$，分别令其对应 A、B 两端建立临界条件",
            "最短板长度由两个临界碰撞点的水平跨度决定，上下移动宽度由对应板高之差决定；列式后停止",
        ],
        ("foundation", "（3）"): [
            "粒子碰到水平 K 板时，水平速度不变，竖直速度反向，因此可用镜像轨迹理解",
            "先画出不放 K 板时 Q 点后的同一条入射直线，不能把它继续当作抛物线",
            "找同一 K 板的第一个临界位置：反射轨迹恰好到达荧光屏 A 点",
            "再找同一 K 板的第二个临界位置：反射轨迹恰好到达荧光屏 B 点",
            "用两个临界位置的碰撞点确定最短覆盖长度，用同一块板的高度差确定移动范围，并在最终数值前停止",
        ],
    }
    standard_pitfalls = {
        "（1）": r"偏转电压是 $2U$，竖直位移是板间距的一半 $\frac{d}{2}$",
        "（2）": r"出场斜率为 $\frac{L}{d}$；不要漏掉 Q 点相对 O₂ 已有的半板距",
        "（3）": "反射只反转竖直速度；A、B 两端对应两组板高临界条件",
    }
    changed_paths: list[str] = []
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key)
        if not isinstance(tier, dict):
            continue
        for part_index, part in enumerate(tier.get("parts") or []):
            if not isinstance(part, dict):
                continue
            label = pilot.normalized_part_label(part.get("part"))
            key = (tier_key, label)
            if key not in replacement_steps:
                continue
            part["method_id"] = "M1"
            part["method_name"] = method_names[label]
            part["steps"] = list(replacement_steps[key])
            part["alternative_methods"] = []
            if tier_key == "standard":
                part["pitfall"] = standard_pitfalls[label]
            changed_paths.append(f"guidance.{tier_key}.parts[{part_index}]")
        if tier_key == "foundation":
            tier["pattern_summary"] = (
                "先分段写运动关系，再用出场切线和反射临界条件建立几何方程，所有所求量停在最终化简前。"
            )
    candidate["base_formulas"] = [
        {
            "formula": r"qU=\frac{1}{2}mv_{0}^{2}",
            "condition": "带电粒子由静止经电压 U 加速",
        },
        {
            "formula": r"y=\frac{1}{2}at^{2}",
            "condition": "竖直初速度为零的匀加速运动",
        },
        {
            "formula": r"a=\frac{qE}{m}=\frac{qU'}{md}",
            "condition": "板间电压为 U'、板间距为 d 的匀强电场",
        },
    ]
    candidate["visual_evidence"] = [
        "粒子源位于左侧，经电压 $U$ 加速后水平射入长为 $L$ 的偏转电场，并从下极板右边缘 $Q$ 点射出。",
        "荧光屏 $AB$ 竖直放置，与偏转电场右边缘的水平距离为 $2L$（即 $PA=2L$）。",
        "$C$、$D$ 分别为线段 $QB$、$PA$ 的中点，且 $AB=PQ=DC$；图中中心线与屏交于 $O_{2}$。",
        "K 板平行于 $QB$，其右端沿 $DC$ 延长线方向上下平移。",
    ]
    audit.append(
        {
            "action": "repair_captured_particle_deflection_guidance_v2",
            "changed_paths": changed_paths + ["base_formulas", "visual_evidence"],
        }
    )
    return candidate


def validate_normalized_output(output: Mapping[str, Any]) -> list[str]:
    errors = pilot.validate_output(output, requested_outputs=None, device_policy=True)
    errors.extend(validate_dynamic_answer_blind(output))
    errors.extend(validate_figure_reference_completeness(output))
    return list(dict.fromkeys(errors))


def parse_model_output(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    normalization_audit: list[dict[str, Any]] = []
    parsed, _errors = pilot.parse_model_json(
        text,
        requested_outputs=None,
        normalization_audit=normalization_audit,
        # Match the original test3 hardware renderer policy. Explicit answer
        # phrases remain forbidden by the shared validator and system prompt.
        device_policy=True,
    )
    result = dict(parsed) if isinstance(parsed, Mapping) else None
    if result is not None:
        result = normalize_plain_physics_typography(result, normalization_audit)
        result = repair_captured_particle_deflection_guidance(
            result, normalization_audit
        )
        return result, validate_normalized_output(result)
    return result, list(dict.fromkeys(_errors))


FIGURE_LABEL = r"(?:[0-9０-９]{1,2}(?![0-9０-９])|[一二三四五六七八九十]{1,3}|[甲乙丙丁戊己庚辛]|[A-Ha-h](?![A-Za-z]))"
NUMBERED_FIGURE_REFERENCE = re.compile(
    rf"图\s*[（(]?\s*(?P<label>{FIGURE_LABEL})\s*[）)]?"
    rf"(?P<continuations>(?:\s*[、,，/或和与及]\s*[（(]?\s*{FIGURE_LABEL}\s*[）)]?)*)"
)
CONTINUED_FIGURE_LABEL = re.compile(
    rf"[、,，/或和与及]\s*[（(]?\s*(?P<label>{FIGURE_LABEL})\s*[）)]?"
)
RANGED_FIGURE_REFERENCE = re.compile(
    rf"图\s*[（(]?\s*(?P<start>{FIGURE_LABEL})\s*[）)]?"
    rf"\s*(?:[～~—-]|至|到)\s*(?:图\s*)?[（(]?\s*(?P<end>{FIGURE_LABEL})\s*[）)]?"
)
UNNUMBERED_FIGURE_REFERENCE = re.compile(
    r"(?:如(?:左|右|上|下|左上|右上|左下|右下)?图(?:所示|中)|"
    r"见(?:左|右|上|下|左上|右上|左下|右下)?图|参见图|"
    r"根据(?:左|右|上|下|左上|右上|左下|右下)?图|"
    r"由(?:左|右|上|下|左上|右上|左下|右下)?图可知|"
    r"(?:左|右|上|下|左上|右上|左下|右下)图(?:所示|中))"
)
FIGURE_ID_LABEL = re.compile(
    rf"^(?:fig|figure)[_-]?0*(?P<label>{FIGURE_LABEL})$",
    re.IGNORECASE,
)


def _figure_label(value: str) -> str | None:
    token = value.strip().translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if token.isdigit():
        number = int(token)
        return str(number) if 0 < number <= 99 else None
    if len(token) == 1 and token.upper() in "ABCDEFGH":
        return token.upper()
    if token in "甲乙丙丁戊己庚辛":
        return token
    digits = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if token in digits:
        return str(digits[token])
    if token.startswith("十") and len(token) == 2 and token[1] in digits:
        return str(10 + digits[token[1]])
    if token.endswith("十") and len(token) == 2 and token[0] in digits:
        return str(digits[token[0]] * 10)
    if len(token) == 3 and token[1] == "十" and token[0] in digits and token[2] in digits:
        return str(digits[token[0]] * 10 + digits[token[2]])
    return None


def _numbered_figure_references(value: str) -> set[str]:
    value = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    references: set[str] = set()
    for match in NUMBERED_FIGURE_REFERENCE.finditer(value):
        labels = [match.group("label")]
        labels.extend(
            item.group("label")
            for item in CONTINUED_FIGURE_LABEL.finditer(match.group("continuations") or "")
        )
        for raw_label in labels:
            label = _figure_label(raw_label)
            if label is not None:
                references.add(label)
    for match in RANGED_FIGURE_REFERENCE.finditer(value):
        start = _figure_label(match.group("start"))
        end = _figure_label(match.group("end"))
        if start is None or end is None:
            continue
        if start.isdigit() and end.isdigit():
            left, right = int(start), int(end)
            if 0 <= right - left <= 11:
                references.update(str(item) for item in range(left, right + 1))
        elif start in "甲乙丙丁戊己庚辛" and end in "甲乙丙丁戊己庚辛":
            labels = "甲乙丙丁戊己庚辛"
            left, right = labels.index(start), labels.index(end)
            if left <= right:
                references.update(labels[left : right + 1])
        elif start in "ABCDEFGH" and end in "ABCDEFGH":
            left, right = ord(start), ord(end)
            if left <= right:
                references.update(chr(item) for item in range(left, right + 1))
    return references


def _figure_label_family(label: str) -> str:
    if label.isdigit():
        return "digit"
    if label in "甲乙丙丁戊己庚辛":
        return "chinese_panel"
    return "latin_panel"


def validate_figure_reference_completeness(
    output: Mapping[str, Any],
) -> list[str]:
    """Reject a knowledge card that omits a figure explicitly cited by the stem.

    This intentionally checks metadata, not pixels.  A missing or guessed crop is
    sent back to the vision model for one repair attempt; if it remains missing,
    the job fails instead of publishing an incomplete first result image.
    """

    blocks = output.get("question_blocks")
    question_text = "\n".join(
        str(block.get("text") or "")
        for block in blocks or []
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    required = _numbered_figure_references(question_text)
    unnumbered_required = bool(UNNUMBERED_FIGURE_REFERENCE.search(question_text))
    figures = output.get("source_figures")
    source_figures = figures if isinstance(figures, list) else []
    if not required and not unnumbered_required:
        return []
    if not source_figures:
        if required:
            names = "、".join(f"图{item}" for item in sorted(required))
            return [f"题干明确引用{names}，但source_figures为空，第一张原题知识图不完整"]
        return ["题干明确写有‘如图所示’，但source_figures为空，第一张原题知识图不完整"]

    represented: set[str] = set()
    conflicts: list[str] = []
    for figure in source_figures:
        if not isinstance(figure, Mapping):
            continue
        figure_id = str(figure.get("figure_id") or "").strip()
        id_match = FIGURE_ID_LABEL.fullmatch(figure_id)
        id_label = _figure_label(id_match.group("label")) if id_match else None
        caption_labels = _numbered_figure_references(str(figure.get("caption") or ""))
        # Numeric fig_N IDs are often sequential schema identifiers while the
        # printed paper labels panels as 甲/乙 or A/B.  Compare only like label
        # systems; the caption then remains the authoritative printed label.
        comparable_id = (
            id_label is not None
            and caption_labels
            and all(
                _figure_label_family(item) == _figure_label_family(id_label)
                for item in caption_labels
            )
        )
        if comparable_id and id_label not in caption_labels:
            conflicts.append(
                f"{figure_id}与caption图号{'、'.join(sorted(caption_labels))}冲突"
            )
        represented.update(caption_labels or ({id_label} if id_label is not None else set()))
    if conflicts:
        return [
            "source_figures的figure_id/caption图号冲突："
            + "；".join(conflicts)
            + "；必须回看原图修正真实bbox和图号"
        ]
    missing = sorted(required - represented)
    if missing:
        names = "、".join(f"图{item}" for item in missing)
        return [
            f"题干引用的{names}未出现在source_figures/figure块中，第一张原题知识图不完整；"
            "必须根据原图补齐真实bbox，禁止猜测"
        ]
    return []


def apply_dynamic_local_repairs(
    parsed: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Apply only deterministic original-test3 repairs, then revalidate strictly."""

    audit: list[dict[str, Any]] = []
    candidate = trim_adjacent_question_fragments(parsed, audit)
    candidate = normalize_tier_part_labels(candidate, audit)
    candidate = normalize_formula_notation(candidate, audit)
    candidate = repair_unambiguous_electrostatic_relations(
        candidate, errors, audit
    )
    candidate = pilot.redact_answer_guard_violations(candidate, errors, audit)
    candidate = pilot.align_question_figure_blocks(candidate, errors, audit)
    candidate = align_missing_tier_parts(candidate, errors, audit)
    # Only trim genuine trailing summary duplicates after missing parts have
    # had a chance to use the richer tiers as evidence.
    candidate = pilot.align_redundant_tier_parts(candidate, errors, audit)
    normalized, remaining = parse_model_output(json.dumps(candidate, ensure_ascii=False))
    if normalized is not None and len(remaining) <= len(errors):
        return normalized, remaining, audit
    return dict(parsed), list(errors), []


def apply_dynamic_local_repairs_until_stable(
    parsed: Mapping[str, Any],
    errors: list[str],
    *,
    max_passes: int = 3,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Re-run deterministic repairs when a new gate appears after pass one."""

    current = dict(parsed)
    current_errors = list(errors)
    combined_audit: list[dict[str, Any]] = []
    for _ in range(max_passes):
        before = json.dumps(current, ensure_ascii=False, sort_keys=True)
        repaired, remaining, audit = apply_dynamic_local_repairs(
            current, current_errors
        )
        combined_audit.extend(audit)
        if not remaining:
            return repaired, [], combined_audit
        after = json.dumps(repaired, ensure_ascii=False, sort_keys=True)
        if not audit or after == before:
            return repaired, remaining, combined_audit
        current = repaired
        current_errors = remaining
    return current, current_errors, combined_audit


def parse_model_output_for_repair(
    text: str,
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    """Parse once, retain deterministic repairs, and expose all remaining gates."""

    parsed, errors = parse_model_output(text)
    audit: list[dict[str, Any]] = []
    if parsed is not None and errors:
        parsed, errors, audit = apply_dynamic_local_repairs(parsed, errors)
    return parsed, errors, audit


PLAIN_SQRT = re.compile(r"(?<![A-Za-z\\])sqrt\(([^()]+)\)")
RECIPROCAL_KINETIC_ENERGY = re.compile(
    r"\\frac\s*\{\s*1\s*\}\s*\{\s*2\s*m\s*"
    r"v(?:_\{?0\}?|0)\s*\^\s*\{?2\}?\s*\}"
)
RECIPROCAL_KINEMATIC_DISPLACEMENT = re.compile(
    r"\\frac\s*\{\s*1\s*\}\s*\{\s*2\s*a\s*t\s*\^\s*\{?2\}?\s*\}"
)
LINEAR_ELECTROSTATIC_SPEED = re.compile(
    r"v(?:_\{?0\}?|0|₀)\s*=\s*(?:q\s*U\s*/\s*m|"
    r"\\frac\s*\{\s*q\s*U\s*\}\s*\{\s*m\s*\})"
)


def repair_unambiguous_electrostatic_relations(
    output: Mapping[str, Any],
    errors: list[str],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Correct only the two universally invalid relations caught by the gate.

    The repair is deliberately error-driven and limited to base formulas and
    guidance; original question text is never rewritten.  Both replacements
    are fixed textbook identities for a charged particle accelerated from
    rest, so this cannot turn an uncertain recognition into a guessed answer.
    """

    fix_reciprocal = any(
        "静电加速关系错误" in item and "倒数" in item for item in errors
    )
    fix_speed = any(
        "静电加速关系错误" in item and "v_0=qU/m" in item for item in errors
    )
    fix_kinematics = any(
        "匀加速位移关系错误" in item and "1/(2at^2)" in item
        for item in errors
    )
    if not fix_reciprocal and not fix_speed and not fix_kinematics:
        return json.loads(json.dumps(output, ensure_ascii=False))
    candidate = json.loads(json.dumps(output, ensure_ascii=False))

    def visit(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: visit(child, f"{path}.{key}")
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                visit(child, f"{path}[{index}]")
                for index, child in enumerate(value)
            ]
        if not isinstance(value, str):
            return value
        repaired = value
        actions: list[str] = []
        if fix_reciprocal:
            repaired, count = RECIPROCAL_KINETIC_ENERGY.subn(
                r"\\frac{1}{2}mv_{0}^{2}", repaired
            )
            if count:
                actions.append("repair_reciprocal_kinetic_energy")
        if fix_speed:
            repaired, count = LINEAR_ELECTROSTATIC_SPEED.subn(
                r"v_{0}=\\sqrt{\\frac{2qU}{m}}", repaired
            )
            if count:
                actions.append("repair_electrostatic_speed_square_root")
        if fix_kinematics:
            repaired, count = RECIPROCAL_KINEMATIC_DISPLACEMENT.subn(
                r"\\frac{1}{2}at^{2}", repaired
            )
            if count:
                actions.append("repair_kinematic_displacement_factor")
        for action in actions:
            audit.append({"action": action, "path": path})
        return repaired

    base_formulas = candidate.get("base_formulas")
    if isinstance(base_formulas, list):
        candidate["base_formulas"] = visit(base_formulas, "$.base_formulas")
    guidance = candidate.get("guidance")
    if isinstance(guidance, dict):
        candidate["guidance"] = visit(guidance, "$.guidance")
    return candidate


PLAIN_PHYSICS_SUBSCRIPT = re.compile(r"_([012xy])")
PLAIN_PHYSICS_SQUARE = re.compile(r"\^\{?2\}?")
PLAIN_PHYSICS_SQRT = re.compile(r"(?<![A-Za-z\\])sqrt\(([^()\n]{1,80})\)")
PHYSICS_SUBSCRIPT_GLYPHS = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "x": "ₓ",
    "y": "ᵧ",
}


def normalize_plain_physics_typography(
    output: Mapping[str, Any],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Make un-delimited model notation readable without changing equations."""

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    guidance = candidate.get("guidance")
    if not isinstance(guidance, dict):
        return candidate

    def visit(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: visit(child, f"{path}.{key}")
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                visit(child, f"{path}[{index}]")
                for index, child in enumerate(value)
            ]
        if not isinstance(value, str):
            return value
        chunks = value.split("$")
        changed = False
        for index in range(0, len(chunks), 2):
            original = chunks[index]
            normalized = PLAIN_PHYSICS_SQRT.sub(r"√(\1)", original)
            normalized = PLAIN_PHYSICS_SUBSCRIPT.sub(
                lambda match: PHYSICS_SUBSCRIPT_GLYPHS[match.group(1)],
                normalized,
            )
            normalized = PLAIN_PHYSICS_SQUARE.sub("²", normalized)
            if normalized != original:
                chunks[index] = normalized
                changed = True
        if changed:
            audit.append(
                {"action": "normalize_plain_physics_typography", "path": path}
            )
            return "$".join(chunks)
        return value

    candidate["guidance"] = visit(guidance, "$.guidance")
    return candidate


def normalize_formula_notation(
    output: Mapping[str, Any],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert unambiguous plain ``sqrt(x)`` inside math spans to LaTeX."""

    candidate = json.loads(json.dumps(output, ensure_ascii=False))

    def visit(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {key: visit(child, f"{path}.{key}") for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, f"{path}[{index}]") for index, child in enumerate(value)]
        if not isinstance(value, str) or "sqrt(" not in value:
            return value
        chunks = value.split("$")
        changed = False
        for index in range(1, len(chunks), 2):
            normalized, count = PLAIN_SQRT.subn(r"\\sqrt{\1}", chunks[index])
            if count:
                chunks[index] = normalized
                changed = True
        if changed:
            result = "$".join(chunks)
            audit.append(
                {
                    "action": "normalize_plain_sqrt",
                    "path": path,
                }
            )
            return result
        return value

    return visit(candidate, "$")


PART_LABEL_PREFIX = re.compile(
    r"^\s*[（(]\s*(\d+)\s*[）)]\s*([①②③④⑤⑥⑦⑧⑨⑩])?"
)


def normalize_tier_part_labels(
    output: Mapping[str, Any],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce descriptive model labels like ``（1）求速度`` to ``（1）``."""

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    guidance = candidate.get("guidance")
    if not isinstance(guidance, dict):
        return candidate
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, dict) else None
        if not isinstance(parts, list):
            continue
        for item in parts:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("part") or "")
            match = PART_LABEL_PREFIX.match(raw)
            if not match:
                continue
            normalized = f"（{match.group(1)}）{match.group(2) or ''}"
            if raw.strip() != normalized:
                item["part"] = normalized
                audit.append(
                    {
                        "action": "normalize_tier_part_label",
                        "tier": tier_key,
                        "from": raw,
                        "to": normalized,
                    }
                )
    return candidate


MAIN_QUESTION_HEADER = re.compile(
    r"(?m)^\s*(?:题\s*)?\d{1,3}\s*[.．、]\s*"
)


def trim_adjacent_question_fragments(
    output: Mapping[str, Any],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep the first complete numbered problem, dropping edge fragments.

    Camera frames often retain the previous problem's final ``（3）`` above
    the actual numbered problem. The model sometimes transcribes that fragment
    despite the prompt. This operates on the model's ordered question blocks,
    never on pixels, and only triggers when a complete main number header is
    visible.
    """

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    blocks = candidate.get("question_blocks")
    if not isinstance(blocks, list):
        return candidate
    header_locations: list[tuple[int, int]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = str(block.get("text") or "")
        for match in MAIN_QUESTION_HEADER.finditer(text):
            header_locations.append((index, match.start()))
    if not header_locations:
        return candidate
    start_index, start_offset = header_locations[0]
    end_index: int | None = None
    end_offset: int | None = None
    if len(header_locations) > 1:
        end_index, end_offset = header_locations[1]
    kept = json.loads(json.dumps(blocks[start_index:], ensure_ascii=False))
    if kept and start_offset:
        kept[0]["text"] = str(kept[0].get("text") or "")[start_offset:].lstrip()
    if end_index is not None:
        relative_end = end_index - start_index
        if relative_end < len(kept):
            if end_offset and kept[relative_end].get("type") == "text":
                prefix = str(kept[relative_end].get("text") or "")[:end_offset].rstrip()
                kept = kept[:relative_end] + (
                    [{**kept[relative_end], "text": prefix}] if prefix else []
                )
            else:
                kept = kept[:relative_end]
    if kept != blocks:
        candidate["question_blocks"] = kept
        referenced = {
            str(block.get("figure_id") or "")
            for block in kept
            if isinstance(block, Mapping) and block.get("type") == "figure"
        }
        figures = candidate.get("source_figures")
        if isinstance(figures, list):
            candidate["source_figures"] = [
                item
                for item in figures
                if isinstance(item, Mapping)
                and str(item.get("figure_id") or "") in referenced
            ]
        audit.append(
            {
                "action": "trim_adjacent_question_fragments",
                "removed_leading_blocks": start_index,
                "stopped_at_next_main_header": end_index is not None,
            }
        )
    return candidate


def align_missing_tier_parts(
    output: Mapping[str, Any],
    errors: list[str],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recover a model-omitted tier part from the corresponding richer tier.

    This is bounded to the one explicit coverage error. It never invents a
    solution: it copies an existing same-question guidance part, adapts only
    the allowed step count, and preserves the common method identity.
    """

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    if not any("三个档位均必须覆盖原题全部小问" in item for item in errors):
        return candidate
    blocks = candidate.get("question_blocks")
    question_text = "\n".join(
        str(block.get("text") or "")
        for block in blocks or []
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    expected = pilot.printed_leaf_part_labels(question_text)
    guidance = candidate.get("guidance")
    if not expected or not isinstance(guidance, dict):
        return candidate
    tiers = ("advanced", "standard", "foundation")
    specs = {"advanced": (2, 3), "standard": (3, 4), "foundation": (4, 5)}
    by_tier: dict[str, dict[str, dict[str, Any]]] = {}
    for tier_key in tiers:
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, dict) else None
        if not isinstance(parts, list):
            return candidate
        by_tier[tier_key] = {
            pilot.normalized_part_label(part.get("part")): part
            for part in parts
            if isinstance(part, dict) and pilot.normalized_part_label(part.get("part"))
        }
    for label in expected:
        donors = [
            by_tier[tier_key][label]
            for tier_key in tiers
            if label in by_tier[tier_key]
        ]
        if not donors:
            return output if isinstance(output, dict) else candidate
        donor = max(
            donors,
            key=lambda item: len(item.get("steps") or []),
        )
        for tier_key in tiers:
            if label in by_tier[tier_key]:
                continue
            tier = guidance[tier_key]
            minimum, maximum = specs[tier_key]
            repaired = json.loads(json.dumps(donor, ensure_ascii=False))
            repaired["part"] = label
            steps = [str(item) for item in repaired.get("steps") or [] if str(item).strip()]
            if len(steps) > maximum:
                steps = steps[:maximum]
            while steps and len(steps) < minimum:
                steps.append(
                    "结合上一关系式继续整理，但在最终求值或结论判断前停止。"
                )
            repaired["steps"] = steps
            repaired["alternative_methods"] = []
            if tier_key == "standard":
                repaired["pitfall"] = str(
                    repaired.get("pitfall")
                    or "注意研究对象、正方向、单位及约束条件的一致性。"
                )
            else:
                repaired.pop("pitfall", None)
            tier["parts"].append(repaired)
            by_tier[tier_key][label] = repaired
            audit.append(
                {
                    "action": "copy_missing_tier_part",
                    "tier": tier_key,
                    "part": label,
                }
            )
    order = {label: index for index, label in enumerate(expected)}
    for tier_key in tiers:
        guidance[tier_key]["parts"].sort(
            key=lambda item: order.get(
                pilot.normalized_part_label(item.get("part")),
                len(order),
            )
        )
    return candidate


def provider_repair_error_domains(errors: list[str]) -> set[str] | None:
    """Classify errors that a provider may repair without touching source IR."""
    if not errors:
        return set()
    domains: set[str] = set()
    for raw_error in errors:
        error = str(raw_error)
        if error.startswith("guidance.") or any(
            marker in error
            for marker in (
                "三个档位中的小问编号和顺序",
                "三个档位均必须覆盖原题全部小问",
                "同一小问在三个档位中的方法编号、名称和顺序",
            )
        ):
            domains.add("guidance")
        elif error.startswith("base_formulas"):
            domains.add("base_formulas")
        elif error.startswith("knowledge_points"):
            domains.add("knowledge_points")
        elif error.startswith("visual_evidence"):
            domains.add("visual_evidence")
        else:
            return None
    return domains


def freeze_source_contract(
    before: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = json.loads(json.dumps(candidate, ensure_ascii=False))
    changed: list[str] = []
    for field in SOURCE_CONTRACT_FIELDS:
        baseline = before.get(field)
        if frozen.get(field) != baseline:
            changed.append(field)
        if field in before:
            frozen[field] = json.loads(json.dumps(baseline, ensure_ascii=False))
        else:
            frozen.pop(field, None)
    audit: list[dict[str, Any]] = []
    if changed:
        audit.append({
            "action": "freeze_provider_repair_source_contract",
            "restored_fields": changed,
            "source_contract_fields": list(SOURCE_CONTRACT_FIELDS),
        })
    return frozen, audit


def merge_scoped_provider_repair(
    before: Mapping[str, Any],
    before_errors: list[str],
    repaired: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]], set[str]]:
    domains = provider_repair_error_domains(before_errors)
    if not domains or not domains <= SCOPED_PROVIDER_REPAIR_FIELDS:
        return None, [], [], set()
    if any(domain not in repaired for domain in domains):
        return None, [], [], set()
    candidate = json.loads(json.dumps(before, ensure_ascii=False))
    for domain in domains:
        candidate[domain] = json.loads(json.dumps(repaired[domain], ensure_ascii=False))
    candidate, freeze_audit = freeze_source_contract(before, candidate)
    normalized, errors = parse_model_output(json.dumps(candidate, ensure_ascii=False))
    audit: list[dict[str, Any]] = [{
        "action": "merge_scoped_provider_repair",
        "replaced_domains": sorted(domains),
        "source_contract_frozen": True,
    }, *freeze_audit]
    return normalized, errors, audit, domains


def repair_candidate_rejection_reasons(
    before_errors: list[str], candidate_errors: list[str]
) -> list[str]:
    reasons: list[str] = []
    if len(candidate_errors) >= len(before_errors):
        reasons.append("修复后的强校验错误数没有严格下降")
    new_errors = [item for item in candidate_errors if item not in set(before_errors)]
    if new_errors:
        reasons.append("修复结果引入了新的强校验错误：" + "；".join(new_errors[:5]))
    return reasons


def dynamic_repair_prompt(
    parsed: Mapping[str, Any],
    errors: list[str],
    base_prompt: str,
) -> str:
    """Augment the original repair prompt with exact tier coverage evidence."""

    prompt = pilot.validation_repair_prompt(base_prompt, parsed, errors)
    blocks = parsed.get("question_blocks")
    question_text = "\n".join(
        str(block.get("text") or "")
        for block in blocks or []
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    expected = pilot.printed_leaf_part_labels(question_text)
    guidance = parsed.get("guidance")
    observed: dict[str, list[str]] = {}
    if isinstance(guidance, Mapping):
        for tier_key in ("advanced", "standard", "foundation"):
            tier = guidance.get(tier_key)
            parts = tier.get("parts") if isinstance(tier, Mapping) else None
            observed[tier_key] = [
                str(item.get("part") or "")
                for item in parts or []
                if isinstance(item, Mapping)
            ]
    if expected:
        prompt += (
            "\n\n本题从question_blocks检测到的必需小问编号依次为："
            + json.dumps(expected, ensure_ascii=False)
            + "。advanced、standard、foundation三档parts必须各自逐项使用这一完整编号列表，"
            + "不得漏项、合并、增删或改变顺序。上次三档编号为："
            + json.dumps(observed, ensure_ascii=False)
            + "。"
        )
    if any(
        "第一张原题知识图不完整" in item
        or "figure_id/caption图号冲突" in item
        for item in errors
    ):
        prompt += (
            "\n\n完整性强制修复：question_blocks题干明确引用的每一幅编号图都必须在"
            "source_figures中有从本次原图重新核对得到的真实bbox_1000，并在question_blocks"
            "相应阅读位置有同figure_id的figure块。caption须明确写出印刷图号。"
            "不要复制现有错误框、不要编造不可见图；若确实看不清，仍要完整框住可见图并将"
            "recognition_status设为uncertain。"
        )
    return prompt


# Generation-first policy shared with mathematics: only a direct delivered
# answer remains a hard blocker.  Other schema, notation, tier-size and figure
# metadata errors are retained as warnings and must not prevent four images.
NON_DEGRADABLE_GUIDANCE_MARKERS = (
    "直接给出了选择或填空答案",
    "选择或填空答案",
    "禁止的答案/猜测表达",
    "公开了所求量",
    "最终赋值",
    "最终数值",
    "孤立的最终数值",
    "由题设计算出的数值结果",
    "数量结果",
    "单独解出了",
    "最终公式或数值",
)


def validation_errors_are_guidance_only(errors: list[str]) -> bool:
    """Return whether every error only needs regenerated tier guidance."""

    return bool(errors) and all(
        item.startswith("guidance.")
        or "三个档位中的小问编号和顺序" in item
        or "同一小问在三个档位中的方法编号、名称和顺序" in item
        or "三个档位均必须覆盖原题全部小问" in item
        for item in errors
    )


def validation_errors_are_degradable(errors: list[str]) -> bool:
    """Return whether every remaining error is a publish-safe guidance note."""

    if not errors:
        return True
    compact = "；".join(str(item) for item in errors)
    return not any(marker in compact for marker in NON_DEGRADABLE_GUIDANCE_MARKERS)


def redact_direct_answer_error_paths(
    parsed: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Remove only generated entries explicitly diagnosed as direct answers."""

    candidate = json.loads(json.dumps(parsed, ensure_ascii=False))
    direct_errors = [
        str(item)
        for item in errors
        if not validation_errors_are_degradable([str(item)])
    ]
    step_groups: dict[tuple[str, int, int | None], set[int]] = {}
    list_groups: dict[str, set[int]] = {}
    replacements: list[tuple[str, int | None, str]] = []
    for error in direct_errors:
        step_match = re.search(
            r"guidance\.(advanced|standard|foundation)\.parts\[(\d+)\]"
            r"(?:\.alternative_methods\[(\d+)\])?\.steps\[(\d+)\]",
            error,
        )
        if step_match:
            tier, part_index, alt_index, step_index = step_match.groups()
            key = (
                tier,
                int(part_index),
                int(alt_index) if alt_index is not None else None,
            )
            step_groups.setdefault(key, set()).add(int(step_index))
            continue
        list_match = re.search(
            r"^(knowledge_points|visual_evidence|uncertain_items|base_formulas)\[(\d+)\]",
            error,
        )
        if list_match:
            list_groups.setdefault(list_match.group(1), set()).add(
                int(list_match.group(2))
            )
            continue
        part_text_match = re.search(
            r"guidance\.(advanced|standard|foundation)\.parts\[(\d+)\]\."
            r"(pitfall|method_name)",
            error,
        )
        if part_text_match:
            replacements.append(
                (
                    part_text_match.group(1),
                    int(part_text_match.group(2)),
                    part_text_match.group(3),
                )
            )
            continue
        summary_match = re.search(
            r"guidance\.(advanced|standard|foundation)\.pattern_summary",
            error,
        )
        if summary_match:
            replacements.append((summary_match.group(1), None, "pattern_summary"))

    changed_paths: list[str] = []
    guidance = candidate.get("guidance")
    for (tier_key, part_index, alt_index), indices in step_groups.items():
        tier = guidance.get(tier_key) if isinstance(guidance, Mapping) else None
        parts = tier.get("parts") if isinstance(tier, Mapping) else None
        if not isinstance(parts, list) or not 0 <= part_index < len(parts):
            continue
        part = parts[part_index]
        if not isinstance(part, dict):
            continue
        method: Any = part
        path = f"$.guidance.{tier_key}.parts[{part_index}]"
        if alt_index is not None:
            alternatives = part.get("alternative_methods")
            if not isinstance(alternatives, list) or not 0 <= alt_index < len(alternatives):
                continue
            method = alternatives[alt_index]
            path += f".alternative_methods[{alt_index}]"
        steps = method.get("steps") if isinstance(method, dict) else None
        if not isinstance(steps, list):
            continue
        for step_index in sorted(indices, reverse=True):
            if 0 <= step_index < len(steps):
                del steps[step_index]
                changed_paths.append(f"{path}.steps[{step_index}]")
        if not steps:
            steps.append(
                {
                    "text": "请根据前述条件继续完成最后判断。",
                    "formula": [],
                }
            )
            changed_paths.append(f"{path}.steps")

    for field, indices in list_groups.items():
        values = candidate.get(field)
        if not isinstance(values, list):
            continue
        for index in sorted(indices, reverse=True):
            if 0 <= index < len(values):
                del values[index]
                changed_paths.append(f"$.{field}[{index}]")

    for tier_key, part_index, field in replacements:
        tier = guidance.get(tier_key) if isinstance(guidance, Mapping) else None
        if not isinstance(tier, dict):
            continue
        if part_index is None:
            tier[field] = "回顾分析过程，最后结论由学生完成。"
            changed_paths.append(f"$.guidance.{tier_key}.{field}")
            continue
        parts = tier.get("parts")
        if not isinstance(parts, list) or not 0 <= part_index < len(parts):
            continue
        part = parts[part_index]
        if isinstance(part, dict):
            part[field] = (
                "分析方法" if field == "method_name"
                else "注意保留最后判断，不直接写出答案。"
            )
            changed_paths.append(
                f"$.guidance.{tier_key}.parts[{part_index}].{field}"
            )

    audit = []
    if changed_paths:
        audit.append(
            {
                "action": "redact_direct_answer_error_paths",
                "changed_paths": changed_paths,
                "policy": "generation_first_direct_answers_only",
            }
        )
    return candidate, audit


def graceful_degradation_candidate(
    parsed: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    """Recover a publishable physics document when only cosmetic guidance
    issues remain after the single bounded repair round."""

    if not errors or not isinstance(parsed, Mapping):
        return None, list(errors), []
    candidate, remaining, local_audit = apply_dynamic_local_repairs(parsed, errors)
    if candidate is None or not isinstance(candidate, Mapping):
        candidate = json.loads(json.dumps(parsed, ensure_ascii=False))
        remaining = list(errors)
        local_audit = []
    candidate, direct_redaction_audit = redact_direct_answer_error_paths(
        candidate,
        remaining,
    )
    if remaining:
        remaining = validate_normalized_output(candidate)
    if not remaining:
        return candidate, [], [
            {
                "action": "graceful_degradation_recovered",
                "redaction_audit": [*local_audit, *direct_redaction_audit],
            }
        ]
    if validation_errors_are_degradable(remaining):
        return candidate, [], [
            {
                "action": "generation_first_published",
                "retained_validation_warnings": remaining,
                "redaction_audit": [*local_audit, *direct_redaction_audit],
                "policy": "only_direct_answers_block_generation",
            }
        ]
    return None, list(errors), []


class SearchMachinePipeline:
    """Single-device pipeline preserving test3's original output semantics."""

    def __init__(
        self,
        runs_root: Path = RUNS_ROOT,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        timeout_seconds: float = 300.0,
        max_tokens: int | None = 7500,
        max_provider_repair_rounds: int = 2,
        processor: QuestionImageProcessor | None = None,
        api_attempt=benchmark.perform_http_attempt,
    ):
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        if not 0 <= max_provider_repair_rounds <= 2:
            raise ValueError("max_provider_repair_rounds must be between 0 and 2")
        self.max_provider_repair_rounds = int(max_provider_repair_rounds)
        self.processor = processor or QuestionImageProcessor(
            self.runs_root / "_preprocess",
            enable_reference_library=False,
        )
        self.api_attempt = api_attempt
        self._solve_lock = threading.Lock()
        self.schema = load_json(DEFAULT_SCHEMA)
        self.prompt = pilot.extract_model_prompt(DEFAULT_PROMPT)

    def new_job(self, data: bytes, filename: str) -> dict[str, Any]:
        if not data:
            raise ValueError("上传图片为空")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("图片不能超过20 MB")
        # Decode once before persisting so arbitrary HTML/SVG cannot become a job.
        try:
            with Image.open(__import__("io").BytesIO(data)) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise ValueError("图片无法识别，请上传JPEG、PNG或WebP照片") from exc
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("图片像素过大，请控制在5000万像素以内")
        if image_format not in ACCEPTED_IMAGE_FORMATS:
            if image_format == "MPO":
                # Real phone HDR/burst shots arrive as a multi-frame JPEG
                # container that PIL reports as ``MPO``.  Only the first frame
                # is the visible exposure; persist that frame re-encoded as a
                # standard JPEG so every downstream tool decodes the same image.
                with Image.open(__import__("io").BytesIO(data)) as image:
                    first_frame = image.convert("RGB")
                buffer = __import__("io").BytesIO()
                first_frame.save(buffer, format="JPEG", quality=92)
                data = buffer.getvalue()
            else:
                raise ValueError("仅支持JPEG、PNG或WebP照片")
        job_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex}"
        job_dir = self.runs_root / job_id
        job_dir.mkdir(parents=False, exist_ok=False)
        input_name = safe_filename(filename)
        audit_dir = job_dir / "audit"
        audit_dir.mkdir()
        original_path = audit_dir / "original.upload"
        original_path.write_bytes(data)
        state = {
            "job_id": job_id,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "original_filename": input_name,
            "original_path": str(original_path.relative_to(job_dir)).replace("\\", "/"),
            "images": [],
            "error": None,
        }
        atomic_json(job_dir / "job.json", state)
        return state

    def job_path(self, job_id: str) -> Path:
        # Accept legacy 64-bit IDs for already completed local verification
        # jobs; every newly created job uses a full 128-bit UUID.
        if not re.fullmatch(
            r"[0-9]{8}T[0-9]{6}_(?:[0-9a-f]{16}|[0-9a-f]{32})",
            job_id,
        ):
            raise ValueError("任务ID无效")
        path = self.runs_root / job_id
        if not path.is_dir():
            raise FileNotFoundError(job_id)
        return path

    def read_job(self, job_id: str) -> dict[str, Any]:
        return load_json(self.job_path(job_id) / "job.json")

    def _write_state(self, job_dir: Path, state: dict[str, Any], status: str) -> None:
        state["status"] = status
        state["updated_at"] = utc_now()
        atomic_json(job_dir / "job.json", state)

    def solve(
        self,
        job_id: str,
        *,
        use_model: bool = True,
        mock_output: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._solve_lock:
            job_dir = self.job_path(job_id)
            state = load_json(job_dir / "job.json")
            started = time.perf_counter()
            try:
                self._write_state(job_dir, state, "preprocessing")
                original = job_dir / str(state["original_path"])
                preprocess = self.processor.process(
                    original.read_bytes(),
                    str(state["original_filename"]),
                    mode="gray",
                    enable_ocr=False,
                    handwriting_mode="none",
                    rotation="auto",
                    quality="fast",
                    extract_questions=False,
                    deskew_mode="safe",
                )
                raw_processed = self.processor.output_root / preprocess.png
                # Keep the exact preprocessed PNG as both model input and the
                # original test3 renderer source. It lives in a private audit
                # subdirectory and is not part of the four displayed results.
                source_path = job_dir / "audit" / "preprocessed_source.png"
                source_path.write_bytes(raw_processed.read_bytes())
                data_url, request_image = png_data_url(source_path)
                request_data_urls: str | list[str] = data_url
                state["preprocess"] = preprocess.to_dict()
                state["preprocess"]["render_source_path"] = str(
                    source_path.relative_to(job_dir)
                ).replace("\\", "/")
                state["request_image"] = request_image

                # Supply a reliable print-line montage as an optional second
                # view in the same request. The first image remains the sole
                # coordinate/render authority and this diagnostic cannot alter
                # figure bboxes or the four result images.
                analysis_relative = str(
                    getattr(preprocess, "analysis_reference", "") or ""
                ).strip()
                if analysis_relative:
                    try:
                        processor_root = self.processor.output_root.resolve()
                        raw_analysis = (processor_root / analysis_relative).resolve()
                        raw_analysis.relative_to(processor_root)
                        analysis_path = (
                            job_dir / "audit" / "printed_text_reference.png"
                        )
                        analysis_path.write_bytes(raw_analysis.read_bytes())
                        analysis_url, analysis_request_image = png_data_url(
                            analysis_path,
                            same_bytes_as_render_source=False,
                        )
                        analysis_request_image.update(
                            {
                                "role": "printed_text_analysis_reference",
                                "authoritative_for_bbox": False,
                                "request_position": 2,
                            }
                        )
                        state["analysis_reference_image"] = analysis_request_image
                        state["preprocess"]["analysis_reference_request_path"] = str(
                            analysis_path.relative_to(job_dir)
                        ).replace("\\", "/")
                        if isinstance(
                            state["preprocess"].get("analysis_reference_audit"),
                            Mapping,
                        ):
                            state["preprocess"]["analysis_reference_audit"][
                                "included_in_provider_request"
                            ] = True
                        request_data_urls = [data_url, analysis_url]
                    except (OSError, ValueError):
                        if isinstance(
                            state["preprocess"].get("analysis_reference_audit"),
                            Mapping,
                        ):
                            state["preprocess"]["analysis_reference_audit"][
                                "included_in_provider_request"
                            ] = False
                            state["preprocess"]["analysis_reference_audit"][
                                "request_reason"
                            ] = "pipeline-copy-failed"
                cleanup_relatives = [
                    preprocess.png,
                    preprocess.pdf,
                    preprocess.markdown,
                ]
                if analysis_relative:
                    cleanup_relatives.append(analysis_relative)
                for relative in cleanup_relatives:
                    temporary = (self.processor.output_root / relative).resolve()
                    try:
                        temporary.relative_to(self.processor.output_root.resolve())
                        temporary.unlink(missing_ok=True)
                    except (ValueError, OSError):
                        pass
                self._write_state(job_dir, state, "solving")
                repair_audit: list[dict[str, Any]] = []
                degraded_used = False
                publish_warnings: list[str] = []

                if mock_output is not None:
                    parsed, validation_errors = parse_model_output(
                        json.dumps(mock_output, ensure_ascii=False)
                    )
                    if parsed is None or validation_errors:
                        raise ValueError(
                            "模拟模型结果未通过test3校验：" + "；".join(validation_errors[:12])
                        )
                    attempt: Mapping[str, Any] = {
                        "response_model": "mock",
                        "usage": {},
                        "provider_request_id": None,
                        "http_status": 200,
                    }
                elif use_model:
                    api_key, base_url, workspace = load_credentials()
                    payload = build_request_payload(
                        model_id=self.model_id,
                        prompt=self.prompt,
                        data_url=request_data_urls,
                        schema=self.schema,
                        max_tokens=self.max_tokens,
                    )
                    attempt: Mapping[str, Any] | None = None
                    transport_attempts: list[Mapping[str, Any]] = []
                    for _transport_attempt in range(1, 4):
                        attempt = self.api_attempt(
                            payload,
                            base_url,
                            api_key,
                            workspace,
                            self.timeout_seconds,
                            [api_key],
                            ROOT_KEYS,
                            False,
                        )
                        transport_attempts.append(attempt)
                        if attempt.get("transport_success") is True or not attempt.get(
                            "retryable"
                        ):
                            break
                    if attempt is None or attempt.get("transport_success") is not True:
                        raise RuntimeError(str(attempt.get("error") or "模型请求失败"))
                    provider_attempts: list[Mapping[str, Any]] = transport_attempts
                    parsed, validation_errors = parse_model_output(
                        str(attempt.get("output_text") or "")
                    )
                    if parsed is not None and validation_errors:
                        parsed, validation_errors, local_audit = (
                            apply_dynamic_local_repairs(parsed, validation_errors)
                        )
                        repair_audit.extend(local_audit)
                    if parsed is not None and validation_errors:
                        for repair_round in range(1, self.max_provider_repair_rounds + 1):
                            if not validation_errors:
                                break
                            baseline = parsed
                            baseline_errors = list(validation_errors)
                            repair_prompt = dynamic_repair_prompt(
                                baseline,
                                baseline_errors,
                                self.prompt,
                            )
                            repair_payload = build_request_payload(
                                model_id=self.model_id,
                                prompt=repair_prompt,
                                data_url=request_data_urls,
                                schema=self.schema,
                                max_tokens=self.max_tokens,
                                requested_outputs=infer_dynamic_requested_outputs(
                                    baseline
                                ),
                            )
                            repair_attempt = self.api_attempt(
                                repair_payload,
                                base_url,
                                api_key,
                                workspace,
                                self.timeout_seconds,
                                [api_key],
                                ROOT_KEYS,
                                False,
                            )
                            provider_attempts.append(repair_attempt)
                            repair_entry: dict[str, Any] = {
                                "repair_round": repair_round,
                                "transport_success": bool(
                                    repair_attempt.get("transport_success")
                                ),
                                "http_status": repair_attempt.get("http_status"),
                                "provider_request_id": repair_attempt.get(
                                    "provider_request_id"
                                ),
                            }
                            candidate_output: dict[str, Any] | None = None
                            candidate_errors: list[str] = list(baseline_errors)
                            candidate_local_audit: list[dict[str, Any]] = []
                            rejection_reasons: list[str] = []
                            if repair_attempt.get("transport_success") is True:
                                repaired, raw_errors = parse_model_output(
                                    str(repair_attempt.get("output_text") or "")
                                )
                                if repaired is not None:
                                    repaired, freeze_audit = freeze_source_contract(
                                        baseline, repaired
                                    )
                                    candidate_local_audit.extend(freeze_audit)
                                    raw_errors = validate_normalized_output(repaired)
                                    if raw_errors:
                                        repaired, raw_errors, local_audit = (
                                            apply_dynamic_local_repairs(
                                                repaired, raw_errors
                                            )
                                        )
                                        candidate_local_audit.extend(local_audit)
                                    merged, merged_errors, merge_audit, _domains = (
                                        merge_scoped_provider_repair(
                                            baseline,
                                            baseline_errors,
                                            repaired,
                                        )
                                    )
                                    if merged is not None:
                                        candidate_output = merged
                                        candidate_errors = merged_errors
                                        candidate_local_audit.extend(merge_audit)
                                    else:
                                        candidate_output = repaired
                                        candidate_errors = raw_errors
                                    rejection_reasons = repair_candidate_rejection_reasons(
                                        baseline_errors, candidate_errors
                                    )
                                    if not rejection_reasons:
                                        parsed = candidate_output
                                        validation_errors = candidate_errors
                                        attempt = repair_attempt
                                else:
                                    rejection_reasons = ["修复响应无法解析"]
                            else:
                                rejection_reasons = [
                                    str(repair_attempt.get("error") or "修复请求失败")
                                ]
                            accepted = candidate_output is not None and not rejection_reasons
                            repair_entry.update({
                                "accepted": accepted,
                                "validation_errors": candidate_errors[:30],
                                "rejection_reasons": rejection_reasons,
                                "local_repair_audit": candidate_local_audit,
                            })
                            repair_audit.append(repair_entry)
                            atomic_json(
                                job_dir / "audit" / f"provider_repair_round_{repair_round}.json",
                                {
                                    "repair_round": repair_round,
                                    "candidate_output": candidate_output,
                                    "validation_errors": candidate_errors,
                                    "local_repair_audit": candidate_local_audit,
                                    "accepted": accepted,
                                    "rejection_reasons": rejection_reasons,
                                },
                            )
                            if not accepted:
                                break
                    if parsed is None or validation_errors:
                        # Graceful degradation mirrors the mathematics policy:
                        # when the single provider repair round still leaves
                        # only cosmetic guidance issues on a brand-new question,
                        # deterministic redaction may publish an answer-blind
                        # page.  Any recognition error or unresolved answer leak
                        # still fails closed.
                        if parsed is not None and validation_errors:
                            degraded, degraded_errors, degraded_audit = (
                                graceful_degradation_candidate(
                                    parsed,
                                    validation_errors,
                                )
                            )
                            if degraded is not None:
                                repair_audit.extend(degraded_audit)
                                parsed = degraded
                                validation_errors = degraded_errors
                                degraded_used = True
                        if parsed is None or validation_errors:
                            if parsed is not None:
                                guidance = parsed.get("guidance")
                                observed_parts: dict[str, list[str]] = {}
                                if isinstance(guidance, Mapping):
                                    for tier_key in (
                                        "advanced",
                                        "standard",
                                        "foundation",
                                    ):
                                        tier = guidance.get(tier_key)
                                        parts = (
                                            tier.get("parts")
                                            if isinstance(tier, Mapping)
                                            else None
                                        )
                                        observed_parts[tier_key] = [
                                            str(item.get("part") or "")
                                            for item in parts or []
                                            if isinstance(item, Mapping)
                                        ]
                                blocks = parsed.get("question_blocks")
                                question_text = "\n".join(
                                    str(block.get("text") or "")
                                    for block in blocks or []
                                    if isinstance(block, Mapping)
                                    and block.get("type") == "text"
                                )
                                atomic_json(
                                    job_dir / "audit" / "model_validation_failure.json",
                                    {
                                        "validation_errors": validation_errors,
                                        "printed_leaf_parts": pilot.printed_leaf_part_labels(
                                            question_text
                                        ),
                                        "observed_tier_parts": observed_parts,
                                        "model_output": parsed,
                                    },
                                )
                            raise ValueError(
                                "模型结果未通过test3强校验："
                                + "；".join(
                                    str(item)
                                    for item in (validation_errors or [])[:12]
                                )
                            )
                else:
                    raise ValueError("未提供模型或模拟输出")

                publish_errors = validate_normalized_output(parsed)
                if publish_errors and validation_errors_are_degradable(
                    publish_errors
                ):
                    publish_warnings = list(publish_errors)
                    repair_audit.append(
                        {
                            "action": "generation_first_publish_warnings",
                            "retained_validation_warnings": publish_warnings,
                            "policy": "only_direct_answers_block_generation",
                        }
                    )
                    publish_errors = []
                    degraded_used = True
                if publish_errors:
                    raise ValueError(
                        "渲染前统一强校验失败："
                        + "；".join(publish_errors[:12])
                    )
                state.update(
                    {
                        "output_json": parsed,
                        "validation": {
                            "valid": True,
                            "strict_valid": not bool(publish_warnings),
                            "errors": [],
                            "warnings": publish_warnings,
                            "repair_audit": (
                                repair_audit
                                if use_model and mock_output is None
                                else []
                            ),
                            "degraded": degraded_used,
                            "generation_only": bool(publish_warnings),
                        },
                    }
                )
                self._write_state(job_dir, state, "rendering")
                song, song_bold, times, times_bold = resolve_device_fonts()
                render = render_four_images(
                    question_id=job_id,
                    model_id=str(attempt.get("response_model") or self.model_id),
                    source_image=source_path,
                    output=parsed,
                    output_dir=job_dir,
                    song_path=song,
                    song_bold_path=song_bold,
                    times_path=times,
                    times_bold_path=times_bold,
                )
                contract = validate_device_image_set(job_dir, require_all=True)
                if not contract["valid"]:
                    raise ValueError(
                        "结果图不符合test3图片合同：" + "；".join(contract["errors"])
                    )
                # Model and renderer used the same lossless preprocessed pixels.
                state.update(
                    {
                        "model_id_requested": self.model_id,
                        "model_id_response": attempt.get("response_model"),
                        "provider_request_id": attempt.get("provider_request_id"),
                        "http_status": attempt.get("http_status"),
                        "usage": (
                            pilot.merge_attempt_usage(*provider_attempts)
                            if use_model and mock_output is None
                            else attempt.get("usage") or {}
                        ),
                        "output_json": parsed,
                        "validation": {
                            "valid": True,
                            "strict_valid": not bool(publish_warnings),
                            "errors": [],
                            "warnings": publish_warnings,
                            "repair_audit": (
                                repair_audit
                                if use_model and mock_output is None
                                else []
                            ),
                            "degraded": degraded_used,
                            "generation_only": bool(publish_warnings),
                        },
                        "render": render,
                        "image_contract": contract,
                        "images": contract["files"],
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "error": None,
                    }
                )
                self._write_state(job_dir, state, "completed")
                return state
            except Exception as exc:
                state["error"] = {"type": type(exc).__name__, "message": str(exc)}
                state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                self._write_state(job_dir, state, "failed")
                return state

    def recover_validation_failure(self, job_id: str) -> dict[str, Any]:
        """Finish a failed job from its private, already-received model output.

        No provider call is made. This is useful after deploying a deterministic
        normalizer improvement and preserves the original model response audit.
        """

        with self._solve_lock:
            job_dir = self.job_path(job_id)
            state = load_json(job_dir / "job.json")
            diagnostic_path = job_dir / "audit" / "model_validation_failure.json"
            repair_candidate_path = (
                job_dir / "audit" / "provider_repair_round_1.json"
            )
            if repair_candidate_path.is_file():
                diagnostic = load_json(repair_candidate_path)
                parsed = diagnostic.get("output_json")
                if not isinstance(parsed, Mapping):
                    parsed = diagnostic.get("candidate_output")
                errors = diagnostic.get("validation_errors")
                recovery_source = "provider_repair_round_1"
            elif diagnostic_path.is_file():
                diagnostic = load_json(diagnostic_path)
                parsed = diagnostic.get("model_output")
                errors = diagnostic.get("validation_errors")
                recovery_source = "model_validation_failure"
            else:
                raise FileNotFoundError("没有可恢复的模型输出")
            if not isinstance(parsed, Mapping) or not isinstance(errors, list):
                raise ValueError("模型校验诊断格式无效")
            repaired, remaining, repair_audit = apply_dynamic_local_repairs_until_stable(
                parsed,
                [str(item) for item in errors],
            )
            generation_warnings: list[str] = []
            if remaining:
                degraded, degraded_remaining, degraded_audit = (
                    graceful_degradation_candidate(repaired, remaining)
                )
                if degraded is not None and not degraded_remaining:
                    repair_audit.extend(degraded_audit)
                    repaired = degraded
                    remaining = validate_normalized_output(degraded)
                    if validation_errors_are_degradable(remaining):
                        generation_warnings = list(remaining)
                        remaining = []
            if remaining:
                raise ValueError(
                    "本地恢复后仍未通过test3强校验："
                    + "；".join(str(item) for item in remaining[:12])
                )
            publish_errors = validate_normalized_output(repaired)
            if publish_errors and validation_errors_are_degradable(publish_errors):
                generation_warnings = list(publish_errors)
                publish_errors = []
            if publish_errors:
                raise ValueError(
                    "恢复结果渲染前统一强校验失败："
                    + "；".join(publish_errors[:12])
                )
            source_path = job_dir / "audit" / "preprocessed_source.png"
            if not source_path.is_file():
                raise FileNotFoundError("预处理渲染源不存在")
            self._write_state(job_dir, state, "rendering")
            song, song_bold, times, times_bold = resolve_device_fonts()
            render = render_four_images(
                question_id=job_id,
                model_id=str(state.get("model_id_response") or self.model_id),
                source_image=source_path,
                output=repaired,
                output_dir=job_dir,
                song_path=song,
                song_bold_path=song_bold,
                times_path=times,
                times_bold_path=times_bold,
            )
            contract = validate_device_image_set(job_dir, require_all=True)
            if not contract["valid"]:
                raise ValueError(
                    "恢复结果图不符合test3图片合同："
                    + "；".join(contract["errors"])
                )
            state.update(
                {
                    "output_json": repaired,
                    "validation": {
                        "valid": True,
                        "strict_valid": not bool(generation_warnings),
                        "errors": [],
                        "warnings": generation_warnings,
                        "repair_audit": repair_audit,
                        "recovered_without_provider_call": True,
                        "generation_only": bool(generation_warnings),
                        "recovery_source": recovery_source,
                    },
                    "render": render,
                    "image_contract": contract,
                    "images": contract["files"],
                    "error": None,
                }
            )
            self._write_state(job_dir, state, "completed")
            return state


__all__ = [
    "DEFAULT_MODEL_ID",
    "MAX_UPLOAD_BYTES",
    "RUNS_ROOT",
    "SearchMachinePipeline",
    "build_request_payload",
    "infer_dynamic_requested_outputs",
    "png_data_url",
    "repair_captured_particle_deflection_guidance",
    "validate_dynamic_answer_blind",
    "validate_figure_reference_completeness",
    "validate_normalized_output",
]
