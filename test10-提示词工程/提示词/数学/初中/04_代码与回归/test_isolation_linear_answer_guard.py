import unittest
from fractions import Fraction
from isolation_linear_answer_guard import linear_integer_solution,linear_rational_solution,arithmetic_fill_solution,protect_linear_target,protect_raw_compact,simple_reciprocal_solution,incomplete_fill_in_ocr

def item(q,steps):
 return {'question_blocks':[{'type':'text','text':q}],'display_guidance':{'novice':{'parts':[{'steps':steps}]}}}
class TestBoundary(unittest.TestCase):
 def test_answer_one_and_intermediate_numbers(self):
  d=item('解方程$3x+2=5$',['由等式性质得3x=3','再除以3得到$x=1$','检查：将所得参数代入原方程，左边3×1+2=5','系数化为1是常用步骤'])
  z=protect_linear_target(d);s=z['display_guidance']['novice']['parts'][0]['steps']
  self.assertEqual(s[0],'由等式性质得3x=3');self.assertNotIn('x=1',s[1]);self.assertNotIn('3×1',s[2]);self.assertIn('系数化为1',s[3]);self.assertIn('3x+2=5',z['question_blocks'][0]['text'])
 def test_computed_result_and_equivalent_workings(self):
  d=item('解方程3(x-1)=12',['已知括号含x-1','两边同除得到x-1=4','此时x=4+1','计算得到x=5','求得3x=15'])
  s=protect_linear_target(d)['display_guidance']['novice']['parts'][0]['steps']
  self.assertIn('x-1=4',s[1]);self.assertIn('x=4+1',s[2]);self.assertNotIn('x=5',s[3]);self.assertIn('3x=15',s[4])
 def test_skip_unsupported_and_ambiguous(self):
  for q in ('解方程1/(x-1)=2','解方程x*x=1','计算3+2','解方程x=1与x=2'):
   self.assertIsNone(linear_integer_solution(q))
  d=item('解方程1/(x-1)=2',['分母为零时x=1不是方程的解'])
  self.assertIn('x=1',protect_linear_target(d)['display_guidance']['novice']['parts'][0]['steps'][0])
 def test_negative_and_decimal(self):
  self.assertEqual(linear_integer_solution('解方程 2x+4=0'),-2)
  self.assertIsNone(linear_integer_solution('解方程2x=1'))
  d=item('解方程 2x+4=0',['化简得到x=-2','x=-2+1不是本题的一步'])
  s=protect_linear_target(d)['display_guidance']['novice']['parts'][0]['steps'];self.assertNotIn('x=-2',s[0]);self.assertIn('x=-2+1',s[1])
 def test_equal_digit_can_remain_as_given_value(self):
  d=item('解方程x+1=2',['原方程中的常数1保留','检查原方程里的常数1是否抄对','把1移到右边，得到x=2-1'])
  s=protect_linear_target(d)['display_guidance']['novice']['parts'][0]['steps']
  self.assertIn('常数1',s[0]);self.assertIn('常数1',s[1]);self.assertIn('x=2-1',s[2])
 def test_reciprocal_equation_and_fraction_answer(self):
  self.assertEqual(simple_reciprocal_solution(r'解方程 $\frac{1}{(x-1)}=2$'),Fraction(3,2))
  d=item(r'解方程 $\frac{1}{(x-1)}=2$',['分母x-1≠0，即x≠1','两边同乘得1=2(x-1)','得到x=3/2','检验x=3/2时分母不为零'])
  s=protect_linear_target(d)['display_guidance']['novice']['parts'][0]['steps']
  self.assertIn('x≠1',s[0]);self.assertIn('1=2(x-1)',s[1]);self.assertNotIn('x=3/2',s[2]);self.assertNotIn('3/2',s[3])
 def test_unsupported_reciprocal_forms_are_not_guessed(self):
  for q in ['解方程1/(x-1)=0','解方程0/(x-1)=2','解方程1/(2*x-1)=2']:
   self.assertIsNone(simple_reciprocal_solution(q))
 def test_fill_in_verification_and_known_numbers(self):
  d=item('填空: +',['已知数为2和3','由2+x=3得到x=3-2','检查：将所得参数代入原式，2+1=3，等式成立','括号内应填1','题设中的1要照常保留'])
  s=protect_linear_target(d)['display_guidance']['novice']['parts'][0]['steps']
  self.assertIn('2+x=3',s[1]);self.assertNotIn('2+1=3',s[2]);self.assertNotIn('填1',s[3]);self.assertIn('题设中的1',s[4])
 def test_incomplete_fill_in_ocr_is_blocked(self):
  self.assertTrue(incomplete_fill_in_ocr('填空: +'))
  for damaged in ('填空: 2+', '填空：2+3=5', '填空：2+（）', '填空：+（）='):
   self.assertTrue(incomplete_fill_in_ocr(damaged), damaged)
  self.assertFalse(incomplete_fill_in_ocr('填空：2+（ ）=3'))
  self.assertFalse(incomplete_fill_in_ocr('填空：2+□=3'))
  self.assertFalse(incomplete_fill_in_ocr('填空：比2大3的数是____'))
  self.assertFalse(incomplete_fill_in_ocr('解方程3x+2=5'))
 def test_fraction_solution_equivalent_decimal_and_prose(self):
  self.assertEqual(linear_rational_solution('解方程2x=1'),Fraction(1,2))
  d=item('解方程2x=1',['两边除以2，得x=1/2','x的值为0.5','方程的解是\\frac{1}{2}','题目里的常数1仍保留','x=1÷2'])
  s=protect_linear_target(d)['display_guidance']['novice']['parts'][0]['steps']
  self.assertNotIn('x=1/2',s[0]);self.assertNotIn('0.5',s[1]);self.assertNotIn('\\frac{1}{2}',s[2]);self.assertIn('常数1',s[3]);self.assertIn('x=1÷2',s[4])
 def test_chained_solution_keeps_work_and_removes_result(self):
  d=item('解方程3x+2=5',['由此x=3÷3=1','x=3/3=1，继续检查','x=3÷3=1+0','原方程的解为1'])
  s=protect_linear_target(d)['display_guidance']['novice']['parts'][0]['steps']
  self.assertIn('x=3÷3',s[0]);self.assertNotIn('=1',s[0]);self.assertNotIn('=1',s[1]);self.assertIn('=1+0',s[2]);self.assertNotIn('解为1',s[3])
 def test_hint_replaces_direct_solution_but_retains_known_and_work(self):
  doc={'l5':[{'s':['原方程是2x=1','两边除以2得到$x=1÷2$','因此得到 $x=\\frac{1}{2}$','可得$x=1÷2=0.5$']}]}
  s=protect_raw_compact(doc,'解方程2x=1')['l5'][0]['s']
  self.assertEqual(s[0],'原方程是2x=1');self.assertIn('x=1÷2',s[1]);self.assertIn('自己算一算',s[2]);self.assertNotIn('\\frac{1}{2}',s[2]);self.assertIn('x=1÷2',s[3]);self.assertNotIn('0.5',s[3]);self.assertEqual(s[3].count('$'),2)
 def test_single_blank_can_identify_target_without_guessing_multiple_blanks(self):
  self.assertEqual(arithmetic_fill_solution('填空：2+（ ）=3'),Fraction(1))
  self.assertEqual(arithmetic_fill_solution('填空：□×4=2'),Fraction(1,2))
  self.assertIsNone(arithmetic_fill_solution('填空：□+□=3'))
  self.assertIsNone(arithmetic_fill_solution('填空：2+（）'))
  doc={'l5':[{'s':['由题意2+x=3','所以$x=1$','括号内应填1','验算：2+1=3']}]}
  s=protect_raw_compact(doc,'填空：2+（ ）=3')['l5'][0]['s']
  self.assertIn('2+x=3',s[0]);self.assertIn('自己算',s[1]);self.assertNotIn('应填1',s[2]);self.assertNotIn('2+1=3',s[3])
 def test_fill_result_and_mistake_are_explained_without_broken_sentence(self):
  doc={'l5':[{'pit':'容易误认为括号内填2，因为2+2=4，但题目要求结果是3','s':['计算3减2，得到结果为1，这是括号内应填的数']}]}
  part=protect_raw_compact(doc,'填空：2+（ ）=3')['l5'][0]
  self.assertIn('计算3减2',part['s'][0]);self.assertIn('自己算',part['s'][0]);self.assertNotIn('结果为1',part['s'][0]);self.assertIn('核对等号',part['pit']);self.assertNotIn('误认为请',part['pit'])
 def test_fill_ocr_keeps_number_fragments_but_ordinary_filter_stays_conservative(self):
  import numpy as np
  from test5_preprocess import OCRItem,filter_inline_annotations
  def fragment(text,x):
   box=np.array([[x,0],[x+20,0],[x+20,20],[x,20]],dtype=np.float32)
   return OCRItem(box,text,0.99,20,20,x+10,10,printed=True)
  items=[fragment('填空：',0),fragment('2',30),fragment('+',60),fragment('3',90)]
  self.assertEqual([i.text for i in filter_inline_annotations(items,20,protect_fill_numerals=True)],['填空：','2','+','3'])
  self.assertNotIn('2',[i.text for i in filter_inline_annotations(items,20)])
 def test_keep_intermediate_step_before_generic_cleanup(self):
  from compact_guidance import answer_blind_analysis_prose
  doc={'l5':[{'s':['展开后得1=2x-2，移项得2x=3，解得x=3/2']}]}
  pre=protect_raw_compact(doc,'解方程1/(x-1)=2')
  cleaned=answer_blind_analysis_prose(pre['l5'][0]['s'][0])
  self.assertIn('1=2x-2',cleaned);self.assertIn('2x=3',cleaned);self.assertNotIn('x=3/2',cleaned)
  self.assertIn('x=3/2',doc['l5'][0]['s'][0])
if __name__=='__main__':unittest.main()
