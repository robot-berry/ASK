import unittest
from fractions import Fraction
from isolation_quality_checks import prepare_local_quality, parse_time_polynomial, motion_expected, refine_tangent_explanation

MOTION='初三数学：直角三角形ABC中，A(0,0)、B(6,0)、C(0,4)。P从A向B以每秒1个单位运动，Q从A向C以每秒0.5个单位运动，同时出发，当P到B时停止。求运动过程中三角形BPQ面积的最大值及对应时间。'
PARABOLA='初三数学：抛物线y=-x²+4x与x轴交于A(0,0)、B(4,0)。点P在A、B之间的抛物线上运动，求三角形PAB面积的最大值及此时P的坐标。'
CIRCLE='初三数学：直角三角形的两条直角边分别为6和8，求这个三角形内切圆和外接圆的半径。'
TANGENT='初三数学：圆O的半径为5，弦AB的长度为8。过A、B分别作圆O的切线，交于点P。求线段PA的长度。'

class LocalQualityTests(unittest.TestCase):
    def test_target_triangle_subscript_is_checked_without_confusing_other_areas(self):
        _,ok=prepare_local_quality({'steps':['S_{△BPQ}=0.25t(6-t)']},MOTION)
        self.assertEqual(ok['issues'],[])
        _,bad=prepare_local_quality({'steps':['S_{△BPQ}=0.5t(6-t)']},MOTION)
        self.assertIn('动点面积展开与底乘高不一致',bad['issues'])
        _,other=prepare_local_quality({'steps':['S_{△APQ}=0.25t²']},MOTION)
        self.assertIn('未能验证动点面积表达式',other['issues'])
    def test_tangent_named_intersection_needs_actual_midpoint_relation(self):
        for definition in ('由PA=PB和OA=OB，OP垂直平分AB，设交点为H。',
                           '设H为OP与AB的交点，由PA=PB、OA=OB可知OP垂直平分AB。'):
            _,a=prepare_local_quality({'steps':[definition,'OH/OA=AH/PA。']},TANGENT)
            self.assertTrue(a['tangent_proof_checked'])
        _,a=prepare_local_quality({'steps':['设H为OP与AB的交点。','OH/OA=AH/PA。']},TANGENT)
        self.assertFalse(a['tangent_proof_checked'])
    def test_observed_tangent_wording_is_repaired_only_for_recognized_question(self):
        text='由OA⊥PA可知△OAP是直角三角形，接下来应把半径、半弦和切线长放进同一个直角三角形中。'
        doc={'steps':['设H为AB中点，OH⊥AB。','OH/OA=AH/PA。',text]}
        fixed,a=prepare_local_quality(doc,TANGENT)
        self.assertIn('相似关系',fixed['steps'][-1]);self.assertIn('澄清切线与半弦的几何依据',a['repairs'])
        unchanged,_=prepare_local_quality(doc,'一个普通圆的题目')
        self.assertEqual(unchanged,doc)
    def test_exact_circle_question_is_clarified_but_false_claim_still_blocked(self):
        sentence='内切圆半径公式中的分母2与直角边、斜边是什么关系？外接圆半径又对应哪条边？'
        fixed,a=prepare_local_quality({'l1':[sentence]},CIRCLE)
        self.assertNotIn('内切圆半径公式的分母表述有误',a['issues'])
        self.assertIn('分母是什么',fixed['l1'][0])
        _,bad=prepare_local_quality({'l1':[sentence,'内切圆半径公式中的分母和斜边有关']},CIRCLE)
        self.assertIn('内切圆半径公式的分母表述有误',bad['issues'])
    def test_motion_detects_bad_expansion_and_keeps_correct(self):
        original={'l5':[{'s':['面积S = ½ × (6 - t) × 0.5t','化简得S = ½ × (6t - 0.5t²) = 3t - 0.25t²']}], 'f':[{'x':'S = 1/2ab'}]}
        _,audit=prepare_local_quality(original,MOTION)
        self.assertIn('动点面积展开与底乘高不一致',audit['issues']);self.assertGreaterEqual(audit['area_expressions_checked'],2)
        valid={'l5':[{'s':['面积S = ½ × (6 - t) × 0.5t','展开得S=1.5t-0.25t²，最后一步请自己计算']}]}
        _,ok=prepare_local_quality(valid,MOTION);self.assertEqual(ok['issues'],[]);self.assertGreaterEqual(ok['area_expressions_checked'],2)
    def test_motion_direct_target_blocked_but_intermediate_is_allowed(self):
        _,a=prepare_local_quality({'l5':[{'s':['S=1.5t-0.25t²','最大值在t=3时取得']}]},MOTION)
        self.assertIn('模型直接写出所求的最值时间',a['issues'])
        _,b=prepare_local_quality({'l5':[{'s':['S=1.5t-0.25t²','顶点横坐标为t=3÷1，请自行计算']}]},MOTION)
        self.assertEqual(b['issues'],[])
        _,c=prepare_local_quality({'l5':[{'s':['S=1.5t-0.25t²','顶点t=-1.5/(2×(-0.25))=3，此时面积最大']}]},MOTION)
        self.assertIn('模型直接写出所求的最值时间',c['issues'])
    def test_motion_each_requested_result_is_checked_separately(self):
        for phrase in ('最大面积为2.25', '面积的最大值是2.25', '最大值=4'):
            _, audit = prepare_local_quality({'steps':['S=1.5t-0.25t²', phrase]}, MOTION)
            self.assertIn('模型直接写出所求的最大面积', audit['issues'])
        _, time_audit = prepare_local_quality({'steps':['S=1.5t-0.25t²', 't=3']}, MOTION)
        self.assertIn('模型直接写出所求的最值时间', time_audit['issues'])
        _, wrong_time = prepare_local_quality({'steps':['S=1.5t-0.25t²', '最大值在t=6时取得']}, MOTION)
        self.assertIn('模型直接写出所求的最值时间', wrong_time['issues'])
        _, warning = prepare_local_quality({'steps':['S=1.5t-0.25t²', '不要认为最大值出现在t=6外']}, MOTION)
        self.assertNotIn('模型直接写出所求的最值时间', warning['issues'])
        _, safe = prepare_local_quality({'steps':['S=1.5t-0.25t²', '最大面积为9÷4，请自行计算']}, MOTION)
        self.assertNotIn('模型直接写出所求的最大面积', safe['issues'])
    def test_motion_varied_values_and_unknown_geometry(self):
        q=MOTION.replace('B(6,0)','B(8,0)').replace('C(0,4)','C(0,8)').replace('0.5个单位','1个单位')
        self.assertEqual(motion_expected(q),(Fraction(0),Fraction(4),Fraction(-1,2)))
        _,a=prepare_local_quality({'l5':[{'s':['S=4t-0.5t²']}]},q);self.assertEqual(a['issues'],[])
        self.assertIsNone(motion_expected(q.replace('P从A向B','P从B向A')))
        self.assertIsNone(motion_expected(q.replace('当P到B时停止','当Q到C时停止')))
        self.assertIsNone(motion_expected(q.replace('同时出发，','')))
    def test_motion_unsupported_continuation_is_not_invented(self):
        _,a=prepare_local_quality({'steps':['S=1.5t-0.25t²','P到B就停止，Q还在运动。']},MOTION)
        self.assertIn('终止条件与继续运动的描述不一致，需核对',a['issues'])
        _,b=prepare_local_quality({'steps':['S=1.5t-0.25t²','P到B时，Q尚未到达C。']},MOTION)
        self.assertNotIn('终止条件与继续运动的描述不一致，需核对',b['issues'])
    def test_unparseable_motion_equation_is_not_treated_as_verified(self):
        _,a=prepare_local_quality({'l5':[{'s':['面积由底乘高再除以二得到，请自行化简']}]},MOTION)
        self.assertIn('未能验证动点面积表达式',a['issues'])
    def test_motion_function_notation_is_checked(self):
        _,a=prepare_local_quality({'l5':[{'s':['展开得S(t)=-0.25t²+1.5t']}]},MOTION)
        self.assertEqual(a['issues'],[]);self.assertEqual(a['area_expressions_checked'],1)
        _,b=prepare_local_quality({'l5':[{'s':['展开得S(t)=-0.25t²+3t']}]},MOTION)
        self.assertIn('动点面积展开与底乘高不一致',b['issues'])
    def test_repair_right_triangle_formula_scope(self):
        raw={'l2':['内切圆半径公式：r=(a+b-c)/2，适用于任意三角形'], 'l5':[{'s':['检查：内切圆半径公式适用于任意三角形']}]}
        fixed,a=prepare_local_quality(raw,CIRCLE)
        self.assertEqual(len(a['repairs']),1);self.assertIn('直角三角形',fixed['l2'][0]);self.assertNotIn('任意三角形',fixed['l2'][0]);self.assertIn('任意三角形',raw['l2'][0])
        same,_=prepare_local_quality(raw,'一般三角形ABC，求内切圆半径');self.assertEqual(same,raw)
    def test_circle_final_values_are_blocked_but_work_can_remain(self):
        _,a=prepare_local_quality({'l5':[{'s':['外接圆半径R = c/2 = 10/2 = 5','内切圆半径r=4/2=2']}]},CIRCLE)
        self.assertIn('模型直接写出所求圆的半径',a['issues'])
        _,b=prepare_local_quality({'l5':[{'s':['斜边c=10','外接圆半径R=10/2','内切圆半径r=4/2']}]},CIRCLE)
        self.assertEqual(b['issues'],[])
    def test_tangent_direct_answer_and_incomplete_path(self):
        _,a=prepare_local_quality({'l5':[{'s':['切线长PA=20/3']}]},TANGENT);self.assertIn('模型直接写出所求切线长度',a['issues'])
        _,b=prepare_local_quality({'l5':[{'s':['可得OC=3，但需结合△OCP关系进一步推导']}]},TANGENT);self.assertIn('切线题尚未建立可计算的关系',b['issues'])
        _,c=prepare_local_quality({'l5':[{'s':['设H为弦AB的中点，OH垂直AB。','相似三角形得到OH/OA=AH/PA。','相似后PA=5×4÷3，请自行计算']}]},TANGENT);self.assertEqual(c['issues'],[]);self.assertTrue(c['tangent_proof_checked'])
    def test_tangent_rejects_false_geometry_not_just_final_leaks(self):
        for statement in ('OP的长度等于半径5与弦心距d的和或差', 'OP=OA+OH', 'PA²=OA²+OP²', 'PA是直角三角形OAP的斜边'):
            raw={'l5':[{'s':['设H为弦AB的中点，OH⊥AB。',statement]}]}
            _,a=prepare_local_quality(raw,TANGENT)
            self.assertIn('切线题存在与题设矛盾的几何关系',a['issues'],statement)
    def test_tangent_unknown_proof_fails_closed_without_mislabeling(self):
        raw={'l5':[{'s':['PA不是斜边，不能认为OP等于半径与弦心距之和或差。','可进一步作辅助线。']}]}
        _,a=prepare_local_quality(raw,TANGENT)
        self.assertNotIn('切线题存在与题设矛盾的几何关系',a['issues'])
        self.assertIn('切线题关键几何关系尚未得到独立核对',a['issues'])
    def test_tangent_pythagorean_equation_and_variant(self):
        doc={'l5':[{'s':['设H为AB中点，OH⊥AB。','根据勾股定理，OA² + PA² = OP²。','由相似三角形得OH/OA=AH/PA。']}]}
        _,a=prepare_local_quality(doc,TANGENT)
        self.assertNotIn('切线题存在与题设矛盾的几何关系',a['issues']);self.assertTrue(a['tangent_proof_checked'])
        variant=TANGENT.replace('半径为5','半径为13').replace('长度为8','长度为24')
        _,b=prepare_local_quality(doc,variant)
        self.assertNotIn('切线题存在与题设矛盾的几何关系',b['issues']);self.assertTrue(b['tangent_proof_checked'])
    def test_tangent_reversed_segment_labels_and_false_position(self):
        good={'l5':[{'s':['设OP与AB交于H，OH⊥AB，AH=4。','由相似得OH/OA=AH/AP。']}]}
        _,ok=prepare_local_quality(good,TANGENT)
        self.assertTrue(ok['tangent_proof_checked']);self.assertEqual(ok['issues'],[])
        bad={'l5':[{'s':['设OP与AB交于H，OH⊥AB。','由相似得OH/OA=AH/AP。'],'pit':'注意OH是弦心距，不是OP的一部分。'}]}
        _,a=prepare_local_quality(bad,TANGENT)
        self.assertIn('切线题存在与题设矛盾的几何关系',a['issues'])
        warning={'l5':[{'s':['设H为中点，OH⊥AB。','OH/OA=AH/AP。'],'pit':'不要认为OH不是OP的一部分。'}]}
        _,b=prepare_local_quality(warning,TANGENT)
        self.assertNotIn('切线题存在与题设矛盾的几何关系',b['issues'])
        _,c=prepare_local_quality({'l5':[{'s':['H是AB的中点。','OH/OA=AH/AP。','AP=20/3']}]},TANGENT)
        self.assertIn('模型直接写出所求切线长度',c['issues'])
        _,d=prepare_local_quality({'l5':[{'s':['H是AB的中点。','OH/OA=AP/AH。']}]},TANGENT)
        self.assertIn('切线题存在与题设矛盾的几何关系',d['issues'])
    def test_tangent_filler_replaced_only_with_verified_construction(self):
        part={'steps':['作OH⊥AB于H。','检查：相似比对应边是否写对？OH对应OA，AH对应AP，所以OH/OA=AH/AP。','先整理已知条件、所求对象和限制范围。']}
        output={'display_guidance':{'novice':{'parts':[part]}}}
        quality={'issues':[],'tangent_proof_checked':True}
        fixed,count=refine_tangent_explanation(output,TANGENT,quality)
        steps=fixed['display_guidance']['novice']['parts'][0]['steps']
        self.assertEqual(count,1);self.assertIn('△OHA∽△OAP',steps[-2]);self.assertNotIn('检查：',steps[-1])
        self.assertEqual(output['display_guidance']['novice']['parts'][0]['steps'][-1],'先整理已知条件、所求对象和限制范围。')
        _,count=refine_tangent_explanation(output,TANGENT,{'issues':['有误'],'tangent_proof_checked':True})
        self.assertEqual(count,0)
        _,count=refine_tangent_explanation(output,TANGENT.replace('过A、B分别作圆O的切线','只过A作圆O的切线'),quality)
        self.assertEqual(count,0)
        unrelated={'display_guidance':{'novice':{'parts':[{'steps':['H是CD中点。','OH对应OA，AH对应AP，OH/OA=AH/AP。','先整理已知条件、所求对象和限制范围。']}]}}}
        _,count=refine_tangent_explanation(unrelated,TANGENT,quality)
        self.assertEqual(count,0)
    def test_circle_denominator_incorrect_claim_is_blocked(self):
        for phrase in ('内切圆半径公式中的分母和斜边有关', '内切圆半径公式中的分母和分子分别对应直角三角形的哪些边？', '内切圆半径公式中的分母2对应哪条边？','内切圆半径公式中的分母是常数2，它对应的是哪条边？'):
            _,a=prepare_local_quality({'l1':[phrase]},CIRCLE)
            self.assertIn('内切圆半径公式的分母表述有误',a['issues'])
        _,b=prepare_local_quality({'l1':['内切圆半径公式中的分母是2，斜边在分子中。']},CIRCLE)
        self.assertNotIn('内切圆半径公式的分母表述有误',b['issues'])
    def test_parabola_equivalent_absolute_value_and_vertex_form(self):
        raw={'l5':[{'pit':'点P在A、B之间时纵坐标非负，高就是y，不要误加绝对值或把高写成横坐标。','s':['配方得S=-2(x-2)²+8。']}]}
        fixed,a=prepare_local_quality(raw,PARABOLA)
        self.assertIn('高可以写成y，也可以写成|y|',fixed['l5'][0]['pit'])
        self.assertNotIn('模型直接写出所求动点坐标或最大面积',a['issues'])
        self.assertIn('S=-2(x-2)²+8',fixed['l5'][0]['s'][0])
        self.assertGreater(a['parabola_area_expressions_checked'],0)
        outside,_=prepare_local_quality(raw,PARABOLA.replace('B(4,0)','B(5,0)'))
        self.assertEqual(outside,raw)
    def test_parabola_area_formula_is_independently_checked(self):
        _,correct=prepare_local_quality({'steps':['S=1/2×4×(-x²+4x)=-2x²+8x']},PARABOLA)
        self.assertGreaterEqual(correct['parabola_area_expressions_checked'],2)
        self.assertNotIn('抛物线动点面积表达式与题设不一致',correct['issues'])
        _,bad=prepare_local_quality({'steps':['S=1/2×4×(-x²+4x)=-2x²+4x']},PARABOLA)
        self.assertIn('抛物线动点面积表达式与题设不一致',bad['issues'])
        _,unsupported=prepare_local_quality({'steps':['S=1/2×4×(-x²+4x)=-2x²+4x']},PARABOLA.replace('三角形PAB','三角形PAC'))
        self.assertEqual(unsupported['parabola_area_expressions_checked'],0)
        self.assertNotIn('抛物线动点面积表达式与题设不一致',unsupported['issues'])
    def test_parabola_blocks_direct_area_but_not_vertex_form(self):
        _,a=prepare_local_quality({'steps':['S=-2(x-2)²+8']},PARABOLA)
        self.assertNotIn('模型直接写出所求动点坐标或最大面积',a['issues'])
        _,b=prepare_local_quality({'steps':['检查S=1/2×4×4=8，结论留给学生。']},PARABOLA)
        self.assertIn('模型直接写出所求动点坐标或最大面积',b['issues'])
        _,c=prepare_local_quality({'steps':['顶点横坐标为x=2','此时点的纵坐标y=4']},PARABOLA)
        self.assertIn('模型直接写出所求动点坐标或最大面积',c['issues'])
        _,d=prepare_local_quality({'steps':['顶点横坐标x=2在0到4之间']},PARABOLA)
        self.assertIn('模型直接写出所求动点坐标或最大面积',d['issues'])
        _,e=prepare_local_quality({'steps':['顶点横坐标2在范围内，所以S在x=2时取得最大值']},PARABOLA)
        self.assertIn('模型直接写出所求动点坐标或最大面积',e['issues'])
    def test_parabola_wrong_warning_and_correct_target(self):
        raw={'l5':[{'pit':'忽略y的正负性，误将面积表达式写成2y而非2|y|','s':['S=2(-x²+4x)']}]}
        fixed,a=prepare_local_quality(raw,PARABOLA);self.assertIn('y非负',fixed['l5'][0]['pit']);self.assertTrue(a['parabola_target_recognized'])
        _,b=prepare_local_quality({'l5':[{'s':['动点P的坐标为(2,4)']}]},PARABOLA);self.assertIn('模型直接写出所求动点坐标或最大面积',b['issues'])
        _,c=prepare_local_quality({'l5':[{'s':['顶点在x=-8/(2×-2)=2，此时y=-4+8=4']}]},PARABOLA);self.assertIn('模型直接写出所求动点坐标或最大面积',c['issues'])
        unchanged,_=prepare_local_quality(raw,PARABOLA.replace('B(4,0)','B(5,0)'));self.assertEqual(unchanged,raw)
    def test_polynomial_parser_rejects_code(self):
        with self.assertRaises(ValueError):parse_time_polynomial('__import__("os")')
        self.assertEqual(parse_time_polynomial('½×(6-t)×0.5t'),(Fraction(0),Fraction(3,2),Fraction(-1,4)))

if __name__=='__main__':unittest.main()
