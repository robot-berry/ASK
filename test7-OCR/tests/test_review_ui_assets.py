from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ReviewUiAssetTests(unittest.TestCase):
    def test_output_draft_helper_and_app_javascript(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is unavailable")
        app_source = (PROJECT_ROOT / "review_ui" / "app.js").read_text(encoding="utf-8")
        helper_source = (PROJECT_ROOT / "review_ui" / "output_draft.js").read_text(
            encoding="utf-8"
        )
        logic_source = (PROJECT_ROOT / "review_ui" / "review_logic.js").read_text(
            encoding="utf-8"
        )
        mask_source = (PROJECT_ROOT / "review_ui" / "mask_editor.js").read_text(
            encoding="utf-8"
        )
        intake_source = (PROJECT_ROOT / "review_ui" / "intake_ui.js").read_text(
            encoding="utf-8"
        )
        boundary_source = (PROJECT_ROOT / "review_ui" / "boundary_editor.js").read_text(
            encoding="utf-8"
        )
        pair_source = (PROJECT_ROOT / "review_ui" / "capture_pair_review.js").read_text(
            encoding="utf-8"
        )
        index_source = (PROJECT_ROOT / "review_ui" / "index.html").read_text(
            encoding="utf-8"
        )
        product_source = (PROJECT_ROOT / "product_ui" / "app.js").read_text(
            encoding="utf-8"
        )
        product_index = (PROJECT_ROOT / "product_ui" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('attention === "render_readback_alert"', app_source)
        self.assertIn('attention === "reviewable"', app_source)
        self.assertIn('attention === "visual_alert"', app_source)
        self.assertIn('attention === "handwriting_alert"', app_source)
        self.assertIn('attention === "latency_alert"', app_source)
        self.assertIn("function outputRiskScore", app_source)
        self.assertIn("function renderReviewFocus", app_source)
        self.assertIn("Date.parse(right.modified_at", app_source)
        self.assertIn('output.naturalWidth === 800', app_source)
        self.assertIn("finalButton.disabled", app_source)
        self.assertIn("const outputCheckGroups", app_source)
        self.assertIn("function openCompareInspector", app_source)
        self.assertNotIn('querySelector("#output-verdict")', app_source)
        self.assertIn('const expectedReviewApiVersion = "review_workstation_v32"', app_source)
        self.assertIn('const expectedReviewUiVersion = "review_ui_v34"', app_source)
        self.assertIn('"UI 34 · 已连接"', app_source)
        self.assertIn("outputInheritedReview", app_source)
        self.assertIn("无需重复验收", app_source)
        self.assertIn('Boolean(right.default_eligible)', app_source)
        self.assertIn('run.online_latency_sla_applicable === false', app_source)
        self.assertIn('"，仅排版候选"', app_source)
        self.assertIn('measurement.online_latency_sla_applicable === false', app_source)
        self.assertIn('"纯排版耗时"', app_source)
        self.assertIn("此耗时不参与 5 秒在线指标", app_source)
        self.assertIn("function loadReviewImageWithRetry", app_source)
        self.assertIn("正在自动重连", app_source)
        self.assertIn('id="retry-initialize"', app_source)
        self.assertIn('window.location.protocol === "file:"', index_source)
        self.assertIn('class="product-entry" href="/ocr/">打开实际 OCR 工程', index_source)
        self.assertIn("测试与验收工作台（非 OCR 处理页）", index_source)
        self.assertIn('fetch("/api/product/process"', product_source)
        self.assertIn('fetch("/api/product/health"', product_source)
        self.assertIn("/api/product/feedback/", product_source)
        self.assertIn('id="download-button"', product_index)
        self.assertIn("只有自动校验通过的结果才能下载", product_index)
        self.assertIn("不会写入 Gold、训练集或 300 题基准测试", product_index)
        self.assertIn("只有发现实际问题时才反馈", product_index)
        self.assertIn("加入问题回归队列", product_index)
        self.assertIn('class="quality-link" href="/review/"', product_index)
        self.assertIn('item.same_output_previous_fail !== true', app_source)
        self.assertIn('function updateRunAgeWarning', app_source)
        self.assertIn('function renderReviewDisposition', app_source)
        self.assertIn('自动门禁已经拒绝本题', app_source)
        self.assertIn('setView("mask")', app_source)
        self.assertIn('setView("intake")', app_source)
        self.assertIn('setView("boundary")', app_source)
        self.assertIn('setView("pair")', app_source)
        self.assertIn("async function refreshHandwritingMasks", app_source)
        self.assertIn("function renderCompareReviewBar", app_source)
        self.assertIn("function showSaveReceipt", app_source)
        self.assertIn("/api/mask-task/", mask_source)
        self.assertIn("/api/mask-signoff/", mask_source)
        self.assertIn("mask_changed_signoffs_invalidated", (
            PROJECT_ROOT / "test7_ocr" / "handwriting_review.py"
        ).read_text(encoding="utf-8"))
        self.assertIn('id="mask-workspace"', index_source)
        self.assertIn("双来源 Mask 队列", index_source)
        self.assertIn('mask_editor.js?v=34', index_source)
        self.assertIn('intake_ui.js?v=34', index_source)
        self.assertIn('boundary_editor.js?v=34', index_source)
        self.assertIn('capture_pair_review.js?v=34', index_source)
        self.assertIn('id="collection-workspace-status"', index_source)
        self.assertIn('id="intake-guided-task"', index_source)
        self.assertIn('id="intake-drop-zone"', index_source)
        self.assertIn('id="intake-open-camera"', index_source)
        self.assertIn('id="intake-camera"', index_source)
        self.assertIn("/api/collection-workspace-refresh", intake_source)
        self.assertIn("/api/collection-guided-register", intake_source)
        self.assertIn('id="guided-task-select"', intake_source)
        self.assertIn("不会把计划行、测试集图片或合成图片计入自建数据", intake_source)
        self.assertIn("行可增量预检", intake_source)
        self.assertIn("report.ready_batch_path", intake_source)
        self.assertIn('正在载入首题图片', (PROJECT_ROOT / "review_ui" / "style.css").read_text(encoding="utf-8"))
        self.assertIn('/api/self-built-collection-plan.csv', index_source)
        self.assertIn('class="compare-review-bar"', index_source)
        self.assertIn("/api/intake-register", intake_source)
        self.assertIn("/api/boundary-task/", boundary_source)
        self.assertIn("/api/pair-task/", pair_source)
        self.assertIn("/api/pair-signoff/", pair_source)
        helper_probe = helper_source + r'''
const assert = require("node:assert/strict");
const draft = module.exports;
const context = {
  run_id: "run-v1", sample_id: "question-01",
  source_sha256: "source-hash", output_sha256: "output-hash", etag: '"etag"'
};
const value = draft.create(context, {
  reviewer_id: " reviewer-1 ", verdict: "fail",
  issues: {latex_source_visible: true, handwriting_visible: false},
  notes: [" 裸 LaTeX ", ""]
}, new Date("2026-09-06T12:00:00Z"));
assert.equal(draft.canRestore(value, context), true);
assert.equal(value.reviewer_id, "reviewer-1");
assert.deepEqual(value.notes, ["裸 LaTeX"]);
assert.equal(draft.canRestore(value, {...context, output_sha256: "new-output"}), false);
assert.equal(draft.canRestore(value, {...context, etag: '"new-etag"'}), false);
const conservative = draft.create(context, {
  verdict: "unknown",
  issues: {formula_render_error: "yes", visual_crop_incomplete: false}, notes: []
});
assert.equal(conservative.verdict, "pending");
assert.deepEqual(conservative.issues, {formula_render_error: null, visual_crop_incomplete: false});
assert.equal(draft.parse("{broken"), null);
'''
        logic_probe = logic_source + r'''
const assert = require("node:assert/strict");
const logic = module.exports;
assert.equal(logic.deriveVerdict({a: null, b: false}), "pending");
assert.equal(logic.deriveVerdict({a: false, b: false}), "pass");
assert.equal(logic.deriveVerdict({a: false, b: true}), "fail");
assert.equal(logic.finalizationBlocker({
  sourceLoaded: true, outputLoaded: true, outputWidth: 800,
  reviewerId: "reviewer-1", issues: {a: false, b: false}
}), null);
assert.match(logic.finalizationBlocker({
  sourceLoaded: true, outputLoaded: true, outputWidth: 800,
  reviewerId: "", issues: {a: false}
}), /验收人/);
assert.match(logic.finalizationBlocker({
  sourceLoaded: true, outputLoaded: true, outputWidth: 799,
  reviewerId: "reviewer-1", issues: {a: false}
}), /800px/);
assert.equal(logic.navigationIndex(3, 0, 1), 1);
assert.equal(logic.navigationIndex(3, 2, 1), 0);
assert.equal(logic.navigationIndex(3, 0, -1), 2);
assert.equal(logic.navigationIndex(3, -1, 1), 0);
assert.equal(logic.navigationIndex(0, -1, 1), -1);
'''
        for command, source in (
            ([node, "--check"], app_source),
            ([node, "--check"], mask_source),
            ([node, "--check"], intake_source),
            ([node, "--check"], boundary_source),
            ([node, "--check"], pair_source),
            ([node, "--check"], product_source),
            ([node], helper_probe),
            ([node], logic_probe),
        ):
            completed = subprocess.run(
                command,
                input=source,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"JavaScript verification failed:\n{completed.stdout}\n{completed.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
