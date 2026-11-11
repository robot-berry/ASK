from __future__ import annotations

import html
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .option_layout import has_standard_option_sequence, option_grid_columns
from .text import parse_question, reflow_text_for_render
from .types import QuestionDocument, VisualAsset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LAYOUT_MARKER_SIGNATURES = {
    794: (53, 187, "stem_min_internal_line_fill_ratio"),
    795: (73, 191, "stem_line_count"),
    796: (109, 197, "stem_min_line_fill_ratio"),
    797: (137, 211, "stem_text_wrap_applied_code"),
}
_TEXT_WRAP_CODES = {1: "balance", 2: "pretty", 3: "wrap", 4: "stable"}


def edge_render_enabled(document: QuestionDocument, config: dict[str, Any]) -> bool:
    rendering = config.get("rendering", {})
    return bool(
        rendering.get("engine") == "edge_katex"
        and (
            document.formula_specialist_used
            or rendering.get("edge_for_text_only", False)
        )
    )


def _decode_layout_markers(image: Image.Image, marker_y: int) -> dict[str, Any]:
    values: dict[str, int] = {}
    for x, (expected_green, expected_blue, name) in _LAYOUT_MARKER_SIGNATURES.items():
        red, green, blue = image.getpixel((x, marker_y))
        if green != expected_green or blue != expected_blue:
            raise RuntimeError(f"Rendered layout metric marker is missing: {name}")
        values[name] = red
    return {
        "stem_line_count": values["stem_line_count"],
        "stem_min_line_fill_ratio": round(
            values["stem_min_line_fill_ratio"] / 255.0,
            4,
        ),
        "stem_min_internal_line_fill_ratio": round(
            values["stem_min_internal_line_fill_ratio"] / 255.0,
            4,
        ),
        "stem_text_wrap_applied": _TEXT_WRAP_CODES.get(
            values["stem_text_wrap_applied_code"],
            "unknown",
        ),
    }


def _find_edge(config: dict[str, Any]) -> Path:
    candidates = [Path(value) for value in config.get("rendering", {}).get("edge_candidates", [])]
    candidates.extend(
        [
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Microsoft Edge headless executable was not found")


def _find_katex_root(config: dict[str, Any]) -> Path:
    configured = []
    for value in config.get("rendering", {}).get("katex_asset_roots", []):
        candidate = Path(value)
        configured.append(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)
    configured.extend(
        [
            PROJECT_ROOT / "vendor" / "katex",
            PROJECT_ROOT / "node_modules" / "katex" / "dist",
            PROJECT_ROOT.parents[1] / "01" / "web" / "static" / "katex",
        ]
    )
    for candidate in configured:
        if (candidate / "katex.min.js").is_file() and (candidate / "katex.min.css").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Local KaTeX assets were not found")


def _prepare_transcript(value: str) -> str:
    value = reflow_text_for_render(value)
    if len(list(re.finditer(r"(?<![A-Za-z])([A-D])[.．、]\s*", value))) < 2:
        return value
    return re.sub(r"(?<!\n)\s+(?=[A-D][.．、]\s*)", "\n", value)


def _stem_markup(value: str) -> str:
    """Mark safe structural rows for layout metrics without changing line flow."""

    # A multi-row aligned/cases expression is visually taller than the text
    # around it.  Letting prose continue to its right makes the browser output
    # compact, but destroys the apparent reading order and hurts final-PNG
    # readback.  Preserve the formula itself and add only a layout boundary.
    separated_lines: list[str] = []
    inline_math = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")
    for source_line in value.splitlines() or [value]:
        remaining = source_line
        while True:
            match = inline_math.search(remaining)
            if match is None:
                separated_lines.append(remaining)
                break
            formula = match.group(1)
            trailing = remaining[match.end() :]
            if r"\begin" in formula and r"\\" in formula and trailing.strip():
                separated_lines.append(remaining[: match.end()].rstrip())
                remaining = trailing.lstrip()
                continue
            separated_lines.append(remaining)
            break
    value = "\n".join(separated_lines)
    lines = value.splitlines()
    def visible_markup(line: str) -> str:
        escaped = html.escape(line)
        escaped = re.sub(
            r"^\((\d{1,2}|i{1,3}|iv|v|vi{0,3}|ix|x)[)）]",
            lambda match: f"（{match.group(1)}）",
            escaped,
            flags=re.IGNORECASE,
        )
        return re.sub(
            r"_{3,}",
            '<span class="answer-slot" role="img" aria-label="填空横线"></span>',
            escaped,
        )

    if len(lines) <= 1:
        return f'<span class="stem-segment">{visible_markup(value)}</span>'

    def self_contained_math(line: str) -> bool:
        dollars = len(re.findall(r"(?<!\\)\$", line))
        return (
            dollars % 2 == 0
            and line.count(r"\(") == line.count(r"\)")
            and line.count(r"\[") == line.count(r"\]")
        )

    if not all(self_contained_math(line) for line in lines):
        return f'<span class="stem-segment">{visible_markup(value)}</span>'
    return "<br>".join(
        f'<span class="stem-segment">{visible_markup(line)}</span>' for line in lines
    )


_VISUAL_VIEW_CAPTIONS = {
    "正面",
    "侧面",
    "左视图",
    "右视图",
    "主视图",
    "俯视图",
    "正视图",
}


def _visual_option_bundle(
    options: list[dict[str, str]],
    assets: list[VisualAsset],
) -> tuple[list[VisualAsset], VisualAsset, str] | None:
    """Recover a narrow, geometry-proven image-option strip without redrawing it."""

    if (
        len(options) != 4
        or [str(option.get("label") or "") for option in options] != list("ABCD")
        or len(assets) != 5
        or any(asset.extraction != "direct_source_crop" or asset.redraw_used for asset in assets)
    ):
        return None
    texts = [str(option.get("text") or "").strip() for option in options]
    if any(texts[:3]) or texts[3] not in _VISUAL_VIEW_CAPTIONS:
        return None
    ordered = sorted(assets, key=lambda asset: asset.source_bbox.x)
    if ordered != assets:
        return None
    centers = [asset.source_bbox.y + asset.source_bbox.height / 2 for asset in ordered]
    maximum_height = max(asset.source_bbox.height for asset in ordered)
    if max(centers) - min(centers) > maximum_height * 0.75:
        return None
    if ordered[4].source_bbox.x <= ordered[3].source_bbox.x + ordered[3].source_bbox.width / 2:
        return None
    return ordered[:4], ordered[4], texts[3]


def _source_visual_layout_html(
    assets: list[VisualAsset],
    maximum_width: int,
    *,
    enabled: bool,
) -> str:
    """Preserve a compact multi-panel figure's source-relative arrangement."""

    if not enabled or len(assets) < 2:
        return ""
    tallest = max(asset.source_bbox.height for asset in assets)

    def render_group(group: list[tuple[int, VisualAsset]]) -> str:
        left = min(asset.source_bbox.x for _index, asset in group)
        top = min(asset.source_bbox.y for _index, asset in group)
        right = max(asset.source_bbox.right for _index, asset in group)
        bottom = max(asset.source_bbox.bottom for _index, asset in group)
        source_width = right - left
        source_height = bottom - top
        if source_width <= 0 or source_height <= 0:
            return ""
        scale = min(1.0, maximum_width / source_width)
        layout_width = max(1, round(source_width * scale))
        layout_height = max(1, round(source_height * scale))
        figures: list[str] = []
        for index, asset in group:
            x = round((asset.source_bbox.x - left) * scale)
            y = round((asset.source_bbox.y - top) * scale)
            width = max(1, round(asset.source_bbox.width * scale))
            figures.append(
                '<figure class="source-visual-item" '
                f'style="left:{x}px;top:{y}px;width:{width}px" '
                f'data-source-x="{asset.source_bbox.x}" data-source-y="{asset.source_bbox.y}">'
                f'<img src="{html.escape(asset.crop_path, quote=True)}" '
                f'alt="原题裁图 {index}"></figure>'
            )
        return (
            '<div class="source-visual-layout" '
            f'style="width:{layout_width}px;height:{layout_height}px">'
            + "".join(figures)
            + "</div>"
        )

    top = min(asset.source_bbox.y for asset in assets)
    bottom = max(asset.source_bbox.bottom for asset in assets)
    if bottom - top <= max(450, round(tallest * 2.2)):
        return render_group(list(enumerate(assets, start=1)))

    # Figures from different paragraphs should not inherit the large blank
    # vertical distances of the photographed page.  Preserve source-relative
    # placement within each visual row, then stack those rows compactly.
    indexed = sorted(
        enumerate(assets, start=1),
        key=lambda item: (item[1].source_bbox.y, item[1].source_bbox.x),
    )
    groups: list[list[tuple[int, VisualAsset]]] = []
    row_gap = max(80, round(tallest * 0.35))
    for item in indexed:
        if not groups:
            groups.append([item])
            continue
        current_bottom = max(asset.source_bbox.bottom for _index, asset in groups[-1])
        if item[1].source_bbox.y - current_bottom > row_gap:
            groups.append([item])
        else:
            groups[-1].append(item)
    if len(groups) <= 1:
        return ""
    parts: list[str] = []
    for group in groups:
        if len(group) == 1:
            index, asset = group[0]
            parts.append(
                f'<figure><img src="{html.escape(asset.crop_path, quote=True)}" '
                f'alt="原题裁图 {index}"></figure>'
            )
        else:
            parts.append(render_group(group))
    return "\n".join(part for part in parts if part)


def _build_html(document: QuestionDocument, katex_root: Path, config: dict[str, Any]) -> str:
    output = config["output"]
    typography = config["typography"]
    prepared = _prepare_transcript(document.canonical_text)
    _, canonical_stem, canonical_options = parse_question(prepared)
    visual_option_bundle = _visual_option_bundle(document.options, document.visual_assets)
    expected_labels = [str(option.get("label") or "") for option in document.options]
    parsed_labels = [str(option.get("label") or "") for option in canonical_options]
    if visual_option_bundle is not None:
        canonical_stem = document.stem
        canonical_options = document.options
    elif (
        not has_standard_option_sequence(document.options)
        or parsed_labels != expected_labels
    ):
        canonical_stem = prepared
        canonical_options = []
    canonical = _stem_markup(canonical_stem)
    content_width = int(output["width_px"]) - int(output["margin_left_px"]) - int(output["margin_right_px"])
    option_columns = option_grid_columns(
        document.options,
        content_width,
        int(typography["font_size_px"]),
    )
    options_html = ""
    question_visual_html = ""
    asset_html = ""
    if visual_option_bundle is not None:
        visual_option_assets, question_visual, visual_caption = visual_option_bundle
        question_visual_html = (
            '<figure class="question-visual"><img src="'
            + html.escape(question_visual.crop_path, quote=True)
            + '" alt="原题题图"><figcaption>'
            + html.escape(visual_caption)
            + "</figcaption></figure>"
        )
        options_html = '<div id="visual-options">' + "".join(
            '<div class="visual-option"><span class="visual-option-label">'
            + html.escape(f"{option['label']}.")
            + '</span><img src="'
            + html.escape(asset.crop_path, quote=True)
            + '" alt="原题图片选项 '
            + html.escape(str(option["label"]), quote=True)
            + '"></div>'
            for option, asset in zip(canonical_options, visual_option_assets)
        ) + "</div>"
    else:
        if canonical_options:
            rows = "".join(
                '<div class="option">'
                + '<span class="option-label">'
                + html.escape(f"{option['label']}.")
                + '</span><span class="option-content">'
                + html.escape(f" {option['text']}")
                + "</span>"
                + "</div>"
                for option in canonical_options
            )
            options_html = f'<div id="options" class="options-{option_columns}">{rows}</div>'
        source_layout_html = _source_visual_layout_html(
            document.visual_assets,
            int(config["visual_assets"]["max_render_width_px"]),
            enabled=bool(
                config.get("rendering", {}).get(
                    "preserve_multi_visual_source_layout", True
                )
            ),
        )
        asset_html = source_layout_html or "\n".join(
            f'<figure><img src="{html.escape(asset.crop_path, quote=True)}" alt="原题裁图 {index}"></figure>'
            for index, asset in enumerate(document.visual_assets, start=1)
        )
    css_uri = (katex_root / "katex.min.css").as_uri()
    katex_uri = (katex_root / "katex.min.js").as_uri()
    auto_uri = (katex_root / "auto-render.min.js").as_uri()
    font_size = int(typography["font_size_px"])
    line_gap = int(typography["line_gap_px"])
    allowed_text_wrap = {"adaptive", "wrap", "pretty", "balance", "stable"}
    stem_text_wrap = str(typography.get("stem_text_wrap", "adaptive")).strip().lower()
    option_text_wrap = str(typography.get("option_text_wrap", "pretty")).strip().lower()
    if stem_text_wrap not in allowed_text_wrap:
        stem_text_wrap = "adaptive"
    if option_text_wrap not in allowed_text_wrap:
        option_text_wrap = "pretty"
    stem_css_text_wrap = "pretty" if stem_text_wrap == "adaptive" else stem_text_wrap
    adaptive_stem_wrap = "true" if stem_text_wrap == "adaptive" else "false"
    return f"""<!doctype html>
<html lang="zh-CN" data-doc-height="0" data-katex-error-count="-1" data-visible-latex-token-count="-1">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="{css_uri}">
<style>
*{{box-sizing:border-box}}html,body{{margin:0;padding:0;width:800px;background:#fff;color:#000}}
body{{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif}}
#page{{position:relative;width:800px;padding:{int(output['margin_top_px'])}px {int(output['margin_right_px'])}px {int(output['margin_bottom_px'])}px {int(output['margin_left_px'])}px;background:#fff}}
#render-marker{{position:absolute;right:0;bottom:0;width:2px;height:1px;background:rgb(7,8,249)}}
.layout-metric-marker{{position:absolute;bottom:0;width:1px;height:1px}}
#layout-internal-fill-marker{{right:5px;background:rgb(0,53,187)}}
#layout-line-count-marker{{right:4px;background:rgb(0,73,191)}}
#layout-min-fill-marker{{right:3px;background:rgb(0,109,197)}}
#layout-wrap-mode-marker{{right:2px;background:rgb(0,137,211)}}
#question{{font-size:{font_size}px;line-height:{font_size + line_gap}px;line-break:strict;word-break:normal;overflow-wrap:break-word}}
#stem{{white-space:pre-wrap;text-wrap:{stem_css_text_wrap}}}
.answer-slot{{display:inline-block;width:3.6em;height:.82em;border-bottom:2px solid currentColor;vertical-align:.08em;white-space:normal}}
#options{{display:grid;column-gap:24px;grid-template-columns:repeat(1,minmax(0,1fr))}}
#options.options-2{{grid-template-columns:repeat(2,minmax(0,1fr))}}#options.options-2 .option{{white-space:nowrap}}
.option{{min-width:0;white-space:pre-wrap;text-wrap:{option_text_wrap}}}
#options.options-2 .option-label{{font-weight:600}}
#visual-options{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin-top:{int(output['asset_gap_px'])}px;align-items:start}}
.visual-option{{min-width:0;text-align:center}}.visual-option-label{{display:block;font-weight:600;text-align:left}}
.visual-option img{{display:block;max-width:100%;height:auto;margin:4px auto 0}}
.question-visual figcaption{{font-size:.82em;margin-top:2px}}
.katex{{font-size:1.08em}}figure{{margin:{int(output['asset_gap_px'])}px auto 0;text-align:center;break-inside:avoid}}
figure img{{display:inline-block;max-width:{int(config['visual_assets']['max_render_width_px'])}px;height:auto}}
.source-visual-layout{{position:relative;margin:{int(output['asset_gap_px'])}px auto 0;break-inside:avoid}}
.source-visual-layout figure{{position:absolute;margin:0;text-align:left}}
.source-visual-layout img{{display:block;width:100%;height:auto;max-width:none}}
</style></head>
<body><main id="page"><div id="question"><div id="stem">{canonical}</div>{question_visual_html}{options_html}</div>{asset_html}<i id="layout-internal-fill-marker" class="layout-metric-marker"></i><i id="layout-line-count-marker" class="layout-metric-marker"></i><i id="layout-min-fill-marker" class="layout-metric-marker"></i><i id="layout-wrap-mode-marker" class="layout-metric-marker"></i><i id="render-marker"></i></main>
<script src="{katex_uri}"></script><script src="{auto_uri}"></script>
<script>
(()=>{{let errors=0;try{{renderMathInElement(document.getElementById('question'),{{
delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}},{{left:'\\\\(',right:'\\\\)',display:false}},{{left:'\\\\[',right:'\\\\]',display:true}}],
throwOnError:false,errorCallback:()=>{{errors+=1}}}});}}catch(_error){{errors+=1;}}
const applyAdaptiveStemWrap=()=>{{const stem=document.getElementById('stem');if(!{adaptive_stem_wrap}){{return '{stem_css_text_wrap}';}}
stem.style.textWrap='pretty';const lineHeight=parseFloat(getComputedStyle(stem).lineHeight)||1;
const lineCount=Math.max(1,Math.round(stem.getBoundingClientRect().height/lineHeight));
if(lineCount>=2&&lineCount<=6){{stem.style.textWrap='balance';return 'balance';}}return 'pretty';}};
const segmentLineMetrics=(segment,stemBounds,contentWidth,lineHeight)=>{{const fragments=[];
const addRect=rect=>{{if(rect.width>=0.25&&rect.height>=0.25)fragments.push({{left:rect.left,right:rect.right,center:(rect.top+rect.bottom)/2}});}};
const walker=document.createTreeWalker(segment,NodeFilter.SHOW_TEXT);let node;while((node=walker.nextNode())){{
const parent=node.parentElement;if(!node.nodeValue||!node.nodeValue.trim()||(parent&&parent.closest('.katex')))continue;
const range=document.createRange();range.selectNodeContents(node);for(const rect of range.getClientRects())addRect(rect);}}
for(const formula of segment.querySelectorAll('.katex')){{if(formula.parentElement.closest('.katex'))continue;
const katexHtml=formula.querySelector('.katex-html');const bases=katexHtml?Array.from(katexHtml.children).filter(child=>child.classList.contains('base')):[];
if(bases.length){{for(const base of bases)addRect(base.getBoundingClientRect());}}else{{addRect(formula.getBoundingClientRect());}}}}
fragments.sort((left,right)=>left.center-right.center);const lines=[];for(const fragment of fragments){{
let line=lines.reduce((best,candidate)=>Math.abs(candidate.center-fragment.center)<Math.abs((best?.center??Infinity)-fragment.center)?candidate:best,null);
if(!line||Math.abs(line.center-fragment.center)>lineHeight*0.45){{line={{left:fragment.left,right:fragment.right,center:fragment.center,count:1}};lines.push(line);}}
else{{line.left=Math.min(line.left,fragment.left);line.right=Math.max(line.right,fragment.right);line.center=(line.center*line.count+fragment.center)/(line.count+1);line.count+=1;}}}}
return lines.map(value=>Math.max(0,Math.min(1,(value.right-stemBounds.left)/Math.max(1,contentWidth))));}};
const stemLineMetrics=()=>{{const stem=document.getElementById('stem');const lineHeight=parseFloat(getComputedStyle(stem).lineHeight)||1;
const stemBounds=stem.getBoundingClientRect();const segmentFills=Array.from(stem.querySelectorAll('.stem-segment')).map(segment=>segmentLineMetrics(segment,stemBounds,stem.clientWidth,lineHeight));
const fills=segmentFills.flat();const internalFills=segmentFills.flatMap(values=>values.length>1?values.slice(0,-1):[]);
return {{lineCount:Math.max(1,Math.min(255,fills.length)),minFillRatio:fills.length?Math.min(...fills):0,
minInternalFillRatio:internalFills.length?Math.min(...internalFills):1}};}};
const setMetricMarker=(id,value,green,blue)=>{{document.getElementById(id).style.background=`rgb(${{value}},${{green}},${{blue}})`;}};
const visibleTextOutsideKatex=()=>{{const parts=[];const walker=document.createTreeWalker(document.getElementById('question'),NodeFilter.SHOW_TEXT);
let node;while((node=walker.nextNode())){{const parent=node.parentElement;if(!parent||!parent.closest('.katex')){{parts.push(node.nodeValue||'');}}}}return parts.join('');}};
const finish=()=>{{const appliedWrap=applyAdaptiveStemWrap();const layout=stemLineMetrics();
setMetricMarker('layout-internal-fill-marker',Math.round(layout.minInternalFillRatio*255),53,187);
setMetricMarker('layout-line-count-marker',layout.lineCount,73,191);
setMetricMarker('layout-min-fill-marker',Math.round(layout.minFillRatio*255),109,197);
setMetricMarker('layout-wrap-mode-marker',({{balance:1,pretty:2,wrap:3,stable:4}}[appliedWrap]||0),137,211);
const katexErrors=errors+document.querySelectorAll('.katex-error').length;
const visibleLatexTokens=(visibleTextOutsideKatex().match(/[\\$]/g)||[]).length;
document.documentElement.dataset.katexErrorCount=String(katexErrors);
document.documentElement.dataset.visibleLatexTokenCount=String(visibleLatexTokens);
document.documentElement.dataset.docHeight=String(Math.ceil(document.getElementById('page').scrollHeight));
document.getElementById('render-marker').style.background=(katexErrors+visibleLatexTokens)?'rgb(249,8,7)':'rgb(7,249,8)';}};
finish();if(document.fonts&&document.fonts.ready){{document.fonts.ready.then(finish);}}setTimeout(finish,200);}})();
</script></body></html>"""


def render_question_katex(
    document: QuestionDocument,
    sample_directory: Path,
    config: dict[str, Any],
    *,
    deadline_monotonic: float | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    edge = _find_edge(config)
    katex_root = _find_katex_root(config)
    html_path = sample_directory / "rendered.html"
    html_path.write_text(_build_html(document, katex_root, config), encoding="utf-8")

    maximum_height = int(config["output"].get("max_height_px", 30000))
    capture_height = min(maximum_height, int(config["rendering"].get("edge_capture_height_px", 10000)))
    edge_timeout_seconds = max(
        0.5,
        min(10.0, float(config["rendering"].get("edge_timeout_seconds", 2.0))),
    )
    edge_timeout_retries = max(
        0,
        min(2, int(config["rendering"].get("edge_timeout_retries", 1))),
    )
    edge_virtual_time_budget_ms = max(
        0,
        min(5000, int(config["rendering"].get("edge_virtual_time_budget_ms", 0))),
    )
    edge_attempt_timeouts = [edge_timeout_seconds] * (edge_timeout_retries + 1)
    if deadline_monotonic is not None:
        remaining_seconds = deadline_monotonic - time.perf_counter()
        if remaining_seconds <= 0:
            raise subprocess.TimeoutExpired(str(edge), 0)
        minimum_viable_attempt = max(
            0.5,
            min(
                edge_timeout_seconds,
                float(
                    config["rendering"].get(
                        "edge_minimum_viable_attempt_seconds", 1.2
                    )
                ),
            ),
        )
        attempt_count = edge_timeout_retries + 1
        while (
            attempt_count > 1
            and remaining_seconds / attempt_count < minimum_viable_attempt
        ):
            attempt_count -= 1
        bounded_timeout = min(
            edge_timeout_seconds,
            max(0.05, remaining_seconds / attempt_count),
        )
        edge_attempt_timeouts = [bounded_timeout] * attempt_count
    captured: Image.Image | None = None
    edge_attempts = 0
    for attempt, attempt_timeout in enumerate(edge_attempt_timeouts):
        edge_attempts = attempt + 1
        with tempfile.TemporaryDirectory(
            prefix="test7_edge_",
            dir=sample_directory,
            ignore_cleanup_errors=True,
        ) as profile:
            profile_path = Path(profile).resolve()
            common = [
                str(edge), "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                "--disable-features=RendererCodeIntegrity", "--disable-background-networking", "--no-first-run",
                "--allow-file-access-from-files", "--force-device-scale-factor=1",
                f"--user-data-dir={profile_path}",
            ]
            if edge_virtual_time_budget_ms:
                common.append(
                    f"--virtual-time-budget={edge_virtual_time_budget_ms}"
                )
            screenshot = (sample_directory / "output.edge.tmp.png").resolve()
            screenshot.unlink(missing_ok=True)
            try:
                subprocess.run(
                    [*common, f"--window-size=800,{capture_height}", f"--screenshot={screenshot}", html_path.resolve().as_uri()],
                    check=True, capture_output=True, timeout=attempt_timeout,
                )
            except subprocess.TimeoutExpired:
                screenshot.unlink(missing_ok=True)
                if attempt >= len(edge_attempt_timeouts) - 1:
                    raise
                continue
            with Image.open(screenshot) as opened:
                captured = opened.convert("RGB").copy()
            screenshot.unlink(missing_ok=True)
            break
    if captured is None:
        raise RuntimeError("Headless renderer did not produce a screenshot")
    if captured.width != 800:
        raise RuntimeError(f"Headless renderer produced {captured.width}px instead of 800px")

    marker_x = (798, 799)
    marker_y = None
    marker_error = False
    for y in range(captured.height):
        pixels = [captured.getpixel((x, y)) for x in marker_x]
        if all(g > 180 and r < 80 and b < 80 for r, g, b in pixels):
            marker_y = y
        elif all(r > 180 and g < 80 and b < 80 for r, g, b in pixels):
            marker_y = y
            marker_error = True
    if marker_y is None:
        raise RuntimeError(f"Rendered document exceeds the {capture_height}px capture window")
    if marker_error:
        raise RuntimeError("KaTeX errors or visible LaTeX source detected after rendering")
    layout_metrics = _decode_layout_markers(captured, marker_y)
    height = marker_y + 1
    if height <= 0 or height > maximum_height:
        raise RuntimeError(f"Invalid dynamic document height: {height}")
    rendered = captured.crop((0, 0, 800, height))
    for marker_x in range(794, 800):
        rendered.putpixel((marker_x, height - 1), (255, 255, 255))
    return rendered, {
        "engine": "edge-katex",
        "width": rendered.width,
        "height": rendered.height,
        "katex_error_count": 0,
        "visible_latex_token_count": 0,
        "katex_assets": str(katex_root),
        "edge_executable": str(edge),
        "edge_timeout_seconds": edge_timeout_seconds,
        "edge_attempt_timeouts_seconds": [
            round(value, 6) for value in edge_attempt_timeouts
        ],
        "edge_deadline_bounded": deadline_monotonic is not None,
        "edge_attempts": edge_attempts,
        "edge_virtual_time_budget_ms": edge_virtual_time_budget_ms,
        "stem_text_wrap": str(config["typography"].get("stem_text_wrap", "adaptive")),
        "option_text_wrap": str(config["typography"].get("option_text_wrap", "pretty")),
        **layout_metrics,
    }
