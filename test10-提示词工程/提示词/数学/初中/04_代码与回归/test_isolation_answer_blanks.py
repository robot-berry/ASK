import unittest
import json
from unittest.mock import patch
from isolation_answer_blanks import mask_supported_circle_answers,add_supported_circle_exercise,add_supported_linear_exercise,audit_supported_linear_work
from isolation_linear_answer_guard import protect_raw_compact
from isolation_quality_checks import prepare_local_quality
from compact_guidance import answer_blind_analysis_prose

CIRCLE='初三数学：直角三角形的两条直角边分别为6和8，求这个三角形内切圆和外接圆的半径。'
ENV={'TEST6_ISOLATED_NON_STREAM':'1','TEST6_ISOLATED_ANSWER_BLANK':'1'}

class AnswerBlankTests(unittest.TestCase):
    @patch.dict('os.environ',ENV)
    def test_linear_blank_preserves_given_digits_and_unsolved_work(self):
        doc={'l5':[{'s':['题目给出x+1=2','整理得x=2-1','因此$x=1$','方程的解为1']}]}
        fixed=protect_raw_compact(doc,'解方程x+1=2')['l5'][0]['s']
        self.assertIn('x+1=2',fixed[0]);self.assertIn('x=2-1',fixed[1]);self.assertIn('x=————',fixed[2]);self.assertIn('解为————',fixed[3])
        self.assertNotIn('x=1',fixed[2]);self.assertEqual(doc['l5'][0]['s'][2],'因此$x=1$')
        for s in fixed:self.assertNotIn('x=1',answer_blind_analysis_prose(s))

    @patch.dict('os.environ',ENV)
    def test_fill_blank_and_check_without_exposing(self):
        doc={'l5':[{'s':['2+x=3','所以$x=1$','括号内应填1','验算：2+1=3']}]}
        s=protect_raw_compact(doc,'填空：2+（ ）=3')['l5'][0]['s']
        self.assertIn('x=————',s[1]);self.assertIn('填————',s[2]);self.assertNotIn('2+1=3',s[3]);self.assertIn('2+x=3',s[0])

    @patch.dict('os.environ',ENV)
    def test_more_explicit_result_wordings_are_blanked(self):
        doc={'l5':[{'s':['该方程的解=1','x的值为1','方程解集为{1}','x∈{1}','原题常数1应保留']}]}
        steps=protect_raw_compact(doc,'解方程3x+2=5')['l5'][0]['s']
        for item in steps[:4]:self.assertIn('————',item)
        self.assertIn('常数1',steps[4])
        fill=protect_raw_compact({'l5':[{'s':['方框内填1','方格中应填1']}]},'填空：2+（ ）=3')
        for item in fill['l5'][0]['s']:self.assertIn('————',item)

    @patch.dict('os.environ',ENV)
    def test_coefficient_one_is_preserved_in_verification_guidance(self):
        doc={'l5':[{'s':['把系数化为1，并代回原方程检验左右两边是否相等。','把系数化为1，检验2+3=5。','检验：代入方程，左边3×1+2=5。']}]}
        steps=protect_raw_compact(doc,'解方程3x+2=5')['l5'][0]['s']
        self.assertIn('系数化为1',steps[0]);self.assertIn('系数化为1',steps[1]);self.assertNotIn('3×1+2=5',steps[2])

    @patch.dict('os.environ',ENV)
    def test_calculation_chain_keeps_original_work_and_blank(self):
        doc={'l5':[{'s':['计算得到$x=3÷3=1$']}]}
        s=protect_raw_compact(doc,'解方程3x+2=5')['l5'][0]['s'][0]
        self.assertIn('3÷3',s);self.assertIn('————',s);self.assertNotIn('=1',s)

    @patch.dict('os.environ',ENV)
    def test_add_blank_only_to_verified_numeric_expression(self):
        doc={'l5':[{'s':['根据乘除互逆，方框=6÷4，保留这个除法算式。','错误算式方框=6÷3。','检查方框=6÷4是否合理。']}]}
        steps=protect_raw_compact(doc,'填空：□×4=6')['l5'][0]['s']
        self.assertIn('方框=6÷4=————',steps[0])
        self.assertNotIn('————',steps[1]);self.assertNotIn('————',steps[2])
        equation=protect_raw_compact({'l5':[{'s':['可得x=12÷3+1。']}]},'解方程3(x-1)=12')
        self.assertIn('x=12÷3+1=————',equation['l5'][0]['s'][0])

    def test_both_circle_results_and_given_values(self):
        doc={'l5':[{'s':['直角边为6和8，斜边为10','r=(6+8-10)/2=2','R=10/2=5','内切圆半径为2，外接圆半径为5']}]}
        fixed,a=mask_supported_circle_answers(doc,CIRCLE);s=fixed['l5'][0]['s']
        self.assertEqual(a['masked'],4);self.assertEqual(a['issues'],[])
        self.assertIn('6和8',s[0]);self.assertIn('斜边为10',s[0]);self.assertIn('r=(6+8-10)/2=————',s[1]);self.assertIn('R=10/2=————',s[2]);self.assertNotIn('半径为2',s[3]);self.assertNotIn('半径为5',s[3])
        _,check=prepare_local_quality(fixed,CIRCLE);self.assertNotIn('模型直接写出所求圆的半径',check['issues'])
        self.assertIn('=2',doc['l5'][0]['s'][1])

    def test_incorrect_circle_results_are_not_just_blanked(self):
        for text in ('r=3','内切圆半径为3','r=2+1','R=10/2=6'):
            _,a=mask_supported_circle_answers({'l5':[{'s':[text]}]},CIRCLE)
            self.assertIn('所写圆半径与题设计算不一致',a['issues'],text)

    def test_incorrect_linear_candidate_is_not_just_blanked(self):
        q='解方程3x+2=5'
        for value in ('x=2','x=1+1','x=3÷3=2'):
            a=audit_supported_linear_work({'l5':[{'s':[value]}]},q)
            self.assertIn('所写未知数结果与题设计算不一致',a['issues'],value)
        good=audit_supported_linear_work({'l5':[{'s':['3x=3，所以x=3÷3','当x=0时左侧为2']}]},q)
        self.assertGreater(good['checked'],0);self.assertEqual(good['issues'],[])
        fill=audit_supported_linear_work({'l5':[{'s':['方框=6÷4']}]},'填空：□×4=6')
        self.assertGreater(fill['checked'],0);self.assertEqual(fill['issues'],[])

    def test_expression_and_unrecognized_questions_remain_unmodified(self):
        doc={'l5':[{'s':['r=(6+8-10)/2','R=10/2','直角边6和8','r=2c']}]}
        fixed,a=mask_supported_circle_answers(doc,CIRCLE);self.assertEqual(fixed,doc);self.assertEqual(a['issues'],[])
        unknown,a=mask_supported_circle_answers({'l5':[{'s':['r=2']}]},'一般三角形求内切圆半径')
        self.assertFalse(a['recognized']);self.assertEqual(unknown['l5'][0]['s'][0],'r=2')

    def test_circle_exercise_added_only_for_single_supported_part(self):
        doc={'l5':[{'n':'1','s':['先求斜边。','分别代入半径公式。']}]}
        fixed,added=add_supported_circle_exercise(doc,CIRCLE)
        self.assertTrue(added);last=fixed['l5'][0]['s'][-1]
        self.assertIn('c=10',last);self.assertIn('r=(6+8-10)÷2=————',last);self.assertIn('R=10÷2=————',last)
        _,quality=prepare_local_quality(fixed,CIRCLE);self.assertEqual(quality['issues'],[])
        again,added=add_supported_circle_exercise(fixed,CIRCLE);self.assertFalse(added);self.assertEqual(again,fixed)
        _,added=add_supported_circle_exercise({'l5':[{'n':'1','s':['第一问']},{'n':'2','s':['第二问']}]},CIRCLE);self.assertFalse(added)
        _,added=add_supported_circle_exercise(doc,CIRCLE+'另求面积。');self.assertFalse(added)

    @patch.dict('os.environ',ENV)
    def test_single_equation_can_get_a_blank_when_model_stops_early(self):
        doc={'l5':[{'n':'1','s':['先两边减去2。','两边再除以3，检查是否符合原方程。']}]}
        cleaned=protect_raw_compact(doc,'解方程3x+2=5')
        fixed,added=add_supported_linear_exercise(cleaned,'解方程3x+2=5')
        self.assertTrue(added);self.assertIn('x=————',fixed['l5'][0]['s'][-1])
        again,added=add_supported_linear_exercise(fixed,'解方程3x+2=5');self.assertFalse(added);self.assertEqual(again,fixed)
        _,added=add_supported_linear_exercise(doc,'解方程x²=4');self.assertFalse(added)

    @patch.dict('os.environ',ENV)
    def test_blank_survives_normalization_and_render_validation(self):
        from compact_guidance import expand_compact_output
        from search_machine_pipeline import parse_model_output,validate_normalized_output
        doc={'l1':[{'n':'1','s':['两条直角边能先确定什么？','两个半径与哪条边有关？']}],
             'l2':['勾股定理：确定斜边','内切圆：使用半径公式','外接圆：半径与斜边有关'],
             'l3':[{'n':'1','s':['已知两条直角边，先看斜边。','确定斜边后分别建立两个半径关系。']}],
             'l4':[{'n':'1','s':['由两直角边可建立平方关系。','内切圆公式要求区分直角边和斜边。','外接圆半径与斜边对应。']}],
             'l5':[{'n':'1','s':['两条直角边为6和8。','斜边c满足c²=6²+8²。','两个公式用于直角三角形。','r=(6+8-10)/2=2，R=10/2=5。','核对公式适用条件。'],'pit':'不要把直角边当作斜边。'}],
             'f':[{'x':'a^2+b^2=c^2','c':'直角三角形'}],'pat':'先求斜边再分别代入。','s':'c','u':[]}
        masked,check=mask_supported_circle_answers(doc,CIRCLE);self.assertEqual(check['issues'],[])
        masked,quality=prepare_local_quality(masked,CIRCLE);self.assertEqual(quality['issues'],[])
        expanded,_=expand_compact_output(masked,subject='mathematics',fallback_question_text=CIRCLE,curriculum_stage='junior_high',part_labels=[])
        parsed,errors=parse_model_output(json.dumps(expanded,ensure_ascii=False))
        self.assertEqual(errors,[]);self.assertEqual(validate_normalized_output(parsed),[])
        display=json.dumps(parsed['display_guidance'],ensure_ascii=False)
        self.assertIn('————',display);self.assertNotIn('r=2',display);self.assertNotIn('R=5',display)

    @patch.dict('os.environ',ENV)
    def test_placeholder_not_treated_as_a_delivered_fill_answer(self):
        from search_machine_pipeline import validate_generic_answer_blind
        def check(text):
            return validate_generic_answer_blind({'question_blocks':[{'type':'text','text':'填空：2+（ ）=3'}], 'guidance':{'foundation':{'parts':[{'part':'1','steps':[text]}]}}})
        self.assertEqual(check('方格中应填————'),[])
        self.assertTrue(check('方格中应填1'))
        self.assertTrue(check('方格中应填————，答案是1'))

if __name__=='__main__':unittest.main()
