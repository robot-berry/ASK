"""Run the portable subset. Three integration tests need the full project."""
import unittest
import test_isolation_answer_blanks as blanks
import test_isolation_linear_answer_guard as linear

reason = '需完整隔离项目及其依赖，本交接包仅含核心代码'
for cls, name in (
    (blanks.AnswerBlankTests, 'test_blank_survives_normalization_and_render_validation'),
    (blanks.AnswerBlankTests, 'test_placeholder_not_treated_as_a_delivered_fill_answer'),
    (linear.TestBoundary, 'test_fill_ocr_keeps_number_fragments_but_ordinary_filter_stays_conservative'),
):
    setattr(cls, name, unittest.skip(reason)(getattr(cls, name)))

names = ['test_isolation_quality_checks', 'test_isolation_answer_blanks',
         'test_isolation_linear_answer_guard', 'test_isolation_language']
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and len(result.skipped) == 3 else 1)
