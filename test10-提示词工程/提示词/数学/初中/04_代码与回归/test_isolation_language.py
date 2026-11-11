import unittest
from unittest.mock import patch
from compact_guidance import answer_blind_analysis_prose


class IsolationLanguageTests(unittest.TestCase):
    @patch.dict('os.environ', {'TEST6_ISOLATED_NON_STREAM':'1'})
    def test_boundary_check_does_not_become_nonsense_on_repeated_scrub(self):
        text='检查：当t=0时S=0，当t=10时S=0，符合实际；接下来在0≤t≤10内求二次函数的最大值，最后一步请你自己算一算。'
        for _ in range(3):
            text=answer_blind_analysis_prose(text)
        self.assertNotIn('所得参数',text)
        self.assertIn('请结合题目条件核对边界',text)
        self.assertIn('0≤t≤10',text)
        self.assertNotIn('符合实际',text)

    @patch.dict('os.environ', {'TEST6_ISOLATED_NON_STREAM':'1'})
    def test_completed_square_expression_is_retained(self):
        text='配方得S=-3(x-3)²+27，请检查顶点是否位于范围内。'
        self.assertEqual(answer_blind_analysis_prose(text),text)

    @patch.dict('os.environ', {'TEST6_ISOLATED_NON_STREAM':'1'})
    def test_partial_scrub_does_not_leave_broken_math(self):
        text='检查x=2时y=-4+8=4，面积S=1/2×4×4=8，最后一步请你自己算一算。'
        for _ in range(3):text=answer_blind_analysis_prose(text)
        self.assertNotIn('所得参数时所得参数',text)
        self.assertNotIn('S=1/2×4×4=8',text)

    @patch.dict('os.environ', {'TEST6_ISOLATED_NON_STREAM':'1'})
    def test_generic_removed_answer_hint_is_natural(self):
        text=answer_blind_analysis_prose('解这个方程得x=2。')
        self.assertNotIn('解参数关系后结合题设范围筛选',text)
        self.assertNotIn('x=2',text)

if __name__=='__main__':unittest.main()
