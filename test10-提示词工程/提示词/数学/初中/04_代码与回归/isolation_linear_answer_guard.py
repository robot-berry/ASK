"""Local-only prototype: protect answers for a narrow class of equations.

Deliberately narrow: one explicitly stated equation, one variable x, a rational
linear solution or a simple constant/(x+b)=c equation; fill-in questions get
only limited verification/direct-phrase filtering. Other inputs are untouched
and must not be interpreted as safe. Original online project is not modified.
"""
from __future__ import annotations
import ast
import copy
import os
import re
from fractions import Fraction
from typing import Any

QUESTION = re.compile(r'^\s*解(?:下列)?方程\s*[:：]?\s*\$?\s*([0-9xX+\-−*/×÷().=（）\s]+?)\s*\$?\s*[。.!！]?\s*$')
NUMBER = r'[+\-−]?(?:\\(?:d)?frac\{\d+\}\{\d+\}|\d+(?:\.\d+)?(?:\s*/\s*\d+)?)'
ASSIGN = re.compile(r'(?<![A-Za-z0-9_])(?P<var>[xX])\s*[=＝]\s*(?P<num>'+NUMBER+r')(?![A-Za-z0-9_.])')
OPS = re.compile(r'^\s*(?:[+\-−*/×÷^]|\\(?:frac|times|cdot))')
FILL_DIRECT = re.compile(r'(?P<lead>(?:(?:空格|括号|横线|空白处|方框|方格)(?:内|中|里)?(?:应当|应该|应|可)?(?:填写|填入|填|是|为)|(?:应该|应当|应)填))(?:的数(?:字)?是|的数(?:字)?为|数字|数)?\s*(?P<num>'+NUMBER+r')(?![A-Za-z0-9_.])')
SOLUTION_PROSE = re.compile(r'(?P<lead>(?:方程的解|这个方程的解|该方程的解|所求的解|方程解|[xX]的值|[xX]的解)(?:为：|是：|是|为|等于|[=＝])\s*)(?P<num>'+NUMBER+r')(?![A-Za-z0-9_.])')
SOLUTION_SET = re.compile(r'(?P<lead>(?:(?:方程的)?解集(?:是|为|等于|[=＝])?|[xX]\s*[∈∊])\s*[\{（(]\s*)(?P<num>'+NUMBER+r')(?P<end>\s*[\}）)])')
ASSIGNMENT_CHAIN = re.compile(r'(?<![A-Za-z0-9_])(?P<var>[xX])\s*[=＝](?P<work>[^。；;\n$=＝]+?)[=＝]\s*(?P<num>'+NUMBER+r')(?![A-Za-z0-9_.])')
MATH_ASSIGN = re.compile(r'(?:(?:所以|因此|故|计算可得|计算得|解得|得到|即)\s*)?\$\s*(?P<var>[xX])\s*[=＝]\s*(?P<num>'+NUMBER+r')\s*\$')
FILL_RESULT = re.compile(r'(?:得到|得出|计算得)?(?:结果|答案|所填的数|括号内的数)(?:是|为|等于)\s*(?P<num>'+NUMBER+r')(?![A-Za-z0-9_.])')
HINT = '最后一步，请你自己算一算，x的值是多少？'
BLANK = '————'

def _blank_mode() -> bool:
    return os.environ.get('TEST6_ISOLATED_NON_STREAM') == '1' and os.environ.get('TEST6_ISOLATED_ANSWER_BLANK') == '1'

class IncompleteFillInError(ValueError):
    """A fill-in problem could not be read completely in the isolated test."""

def incomplete_fill_in_ocr(text: str) -> bool:
    """Block visibly incomplete arithmetic fill-in transcripts in local tests.

    This detects only obvious damage, not substituted or missing digits in an
    otherwise well-formed transcript. Never treat a False result as OCR proof.
    """
    text=str(text or '').strip()
    if not re.match(r'^填空\s*[:：]', text):return False
    if not re.search(r'[+×÷=＝]', text):return False
    has_blank = bool(re.search(r'[（(]\s*[）)]|_{2,}|□|▢|\[\s*\]', text))
    return not re.search(r'\d', text) or not re.search(r'[=＝]', text) or not has_blank

def _expr(node: ast.AST) -> tuple[Fraction, Fraction]:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(0), Fraction(str(node.value))
    if isinstance(node, ast.Name) and node.id == 'x':
        return Fraction(1), Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        a,b=_expr(node.operand);s=-1 if isinstance(node.op,ast.USub) else 1
        return s*a,s*b
    if isinstance(node, ast.BinOp):
        a,b=_expr(node.left);c,d=_expr(node.right)
        if isinstance(node.op, ast.Add):return a+c,b+d
        if isinstance(node.op, ast.Sub):return a-c,b-d
        if isinstance(node.op, ast.Mult):
            if a and c:raise ValueError('nonlinear')
            return a*d+c*b,b*d
        if isinstance(node.op, ast.Div) and not c and d:
            return a/d,b/d
    raise ValueError('unsupported expression')

def linear_rational_solution(question: str) -> Fraction | None:
    m=QUESTION.fullmatch(str(question or ''))
    if not m:return None
    text=m.group(1).replace('−','-').replace('×','*').replace('÷','/').replace('（','(').replace('）',')').replace('X','x')
    text=re.sub(r'\s+','',text)
    if text.count('=')!=1 or text.count('x')==0 or len(text)>160:return None
    text=re.sub(r'(?<=[0-9)])(?=[x(])','*',text)
    try:
        left,right=text.split('=');a,b=_expr(ast.parse(left,mode='eval').body);c,d=_expr(ast.parse(right,mode='eval').body)
        if a==c:return None
        solution=(d-b)/(a-c)
        return solution if abs(solution)<=1000000 else None
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError,RecursionError):return None

def linear_integer_solution(question: str) -> int | None:
    solution = linear_rational_solution(question)
    return int(solution) if solution is not None and solution.denominator == 1 else None

def arithmetic_fill_solution(question: str) -> Fraction | None:
    """Solve only a single explicitly blank, linear arithmetic equality."""
    text = re.sub(r'^\s*填空\s*[:：]\s*', '', str(question or ''))
    if text == str(question or ''):return None
    blank = re.compile(r'[（(]\s*[）)]|_{2,}|□|▢|\[\s*\]')
    if len(blank.findall(text)) != 1:return None
    text = blank.sub('x', text)
    if not re.fullmatch(r'[\d\s+\-−×÷*/().（）=＝]+|[\d\s+\-−×÷*/().（）=＝x]+', text):return None
    return linear_rational_solution('解方程' + text.replace('＝', '='))

def simple_reciprocal_solution(question: str) -> Fraction | None:
    text=re.sub(r'\s+','',str(question or '')).replace('$','').replace('−','-').replace('（','(').replace('）',')')
    text=re.sub(r'\\(?:d)?frac\{([+\-]?\d+)\}\{\(?x([+\-]\d+)\)?\}',r'\1/(x\2)',text)
    m=re.fullmatch(r'解(?:下列)?方程[:：]?([+\-]?\d+)/\(x([+\-]\d+)\)=([+\-]?\d+)[。.!！]?',text)
    if not m:return None
    numerator,shift,right=map(int,m.groups())
    if numerator==0 or right==0:return None
    result=Fraction(numerator,right)-shift
    return result if abs(result)<=1000000 else None

def _number_value(text: str) -> Fraction:
    text=re.sub(r'\s+','',text).replace('−','-')
    m=re.fullmatch(r'([+\-]?)\\(?:d)?frac\{(\d+)\}\{(\d+)\}',text)
    if m:return (-1 if m.group(1)=='-' else 1)*Fraction(int(m.group(2)),int(m.group(3)))
    return Fraction(text)

def _direct_value(match: re.Match[str], text: str, expected: Fraction) -> bool:
    try:value=_number_value(match.group('num'))
    except (ValueError,ZeroDivisionError):return False
    return value==expected and not OPS.match(text[match.end():])

def _append_arithmetic_blank(text: str, answer: Fraction, fill_in: bool) -> str:
    """Attach a blank only to a recognized, exact, numeric target expression."""
    if not _blank_mode() or BLANK in text or re.search(r'检查|核对|验算|检验',text):return text
    label=r'(?:[xX]|方框|空格(?:里的数)?|括号里的数)' if fill_in else r'[xX]'
    pattern=re.compile(r'(?<![A-Za-z0-9_])(?P<label>'+label+r')\s*[=＝]\s*(?P<expr>[0-9\s+\-−×÷*/().]{1,100})')
    def repl(match: re.Match[str]) -> str:
        if match.end()<len(text) and (text[match.end()].isalnum() or text[match.end()] in '=＝^'):
            return match.group(0)
        expression=match.group('expr').strip().replace('−','-').replace('×','*').replace('÷','/')
        if not any(op in expression for op in ('+','-','*','/')):return match.group(0)
        try:a,b=_expr(ast.parse(expression,mode='eval').body)
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):return match.group(0)
        if a or b!=answer:return match.group(0)
        return match.group(0).rstrip()+'='+BLANK
    return pattern.sub(repl,text)

def _clean_string(text: str, answer: Fraction) -> str:
    hint = 'x='+BLANK if _blank_mode() else HINT
    has_answer=any(_direct_value(m,text,answer) for m in ASSIGN.finditer(text))
    if has_answer and re.search(r'代入|代回|检验|验算|检查',text):
        return '将所得结果代回原方程，检验等式两边是否相等。'
    chain_changed = False
    def chain_repl(m: re.Match[str]) -> str:
        nonlocal chain_changed
        if not _direct_value(m, text, answer):return m.group(0)
        chain_changed = True
        return m.group('var')+'='+m.group('work').rstrip()
    cleaned=ASSIGNMENT_CHAIN.sub(chain_repl,text)
    def math_repl(m: re.Match[str]) -> str:
        try: value = _number_value(m.group('num'))
        except (ValueError, ZeroDivisionError): return m.group(0)
        return hint if value == answer else m.group(0)
    cleaned = MATH_ASSIGN.sub(math_repl, cleaned)
    def math_block_repl(m: re.Match[str]) -> str:
        block = m.group(0)
        if any(_direct_value(found, block, answer) for found in ASSIGN.finditer(block)):
            return hint
        return block
    cleaned = re.sub(r'\$[^$\n]+\$', math_block_repl, cleaned)
    before_assignments = cleaned
    def repl(m: re.Match[str]) -> str:
        if not _direct_value(m,before_assignments,answer):return m.group(0)
        return hint
    cleaned=ASSIGN.sub(repl,cleaned)
    def prose_repl(m: re.Match[str]) -> str:
        if not _direct_value(m,cleaned,answer):return m.group(0)
        return m.group('lead')+BLANK if _blank_mode() else hint
    cleaned=SOLUTION_PROSE.sub(prose_repl,cleaned)
    def set_repl(m: re.Match[str]) -> str:
        try:value=_number_value(m.group('num'))
        except (ValueError,ZeroDivisionError):return m.group(0)
        if value!=answer:return m.group(0)
        return m.group('lead')+BLANK+m.group('end') if _blank_mode() else hint
    cleaned=SOLUTION_SET.sub(set_repl,cleaned)
    cleaned = re.sub(r'(?:所以|因此|故|计算可得|计算得|解得|得到|即)\s*(?=最后一步，请你)', '', cleaned)
    if chain_changed and hint not in cleaned:
        cleaned = cleaned.rstrip('。；; ') + ('，所以 ' if _blank_mode() else '。') + hint
    # Remove the final-result cue as well, so the project's later generic
    # sentence filter does not discard preceding legitimate calculations.
    cleaned=re.sub(r'(?:解(?:这个方程)?得|求得|计算得)(?=\s*\$?[xX]\s*[=＝]\s*(?:____|—{2,}|\\underline))','最后一步：',cleaned)
    if _blank_mode():
        cleaned=re.sub(r'(?<![A-Za-z0-9_])_{3,}',BLANK,cleaned)
    # A numeric substitution in a verification sentence can reveal the result
    # even if the adjacent x=value has been hidden. Be deliberately conservative
    # on that sentence only; retain ordinary numerals elsewhere.
    num=re.compile(r'(?<![\d.])'+re.escape(str(answer)).replace(r'\-',r'[\-−]')+r'(?![\d.])')
    if re.search(r'代入|代回|检验|验算|检查',cleaned) and num.search(cleaned):
        # Preserve ordinary constants such as “系数化为1”. Only remove a
        # numerical substitution equality containing the actual answer.
        numeric_equality=re.finditer(r'\d\s*[+\-−×*÷/]\s*\d[^。；;\n]*[=＝]\s*[+\-−]?\d+(?:\.\d+)?',cleaned)
        if any(num.search(match.group(0)) for match in numeric_equality):
            return '将所得结果代回原方程，检验等式两边是否相等。'
    return cleaned

def protect_raw_compact(document: dict[str, Any], question: str) -> dict[str, Any]:
    """Protect supported answers before generic sanitization can discard steps."""
    answer=linear_rational_solution(question)
    if answer is None:answer=simple_reciprocal_solution(question)
    fill_in = bool(re.match(r'^\s*填空', str(question or '')))
    if answer is None:answer=arithmetic_fill_solution(question)
    if answer is None:return document
    result=copy.deepcopy(document)
    def walk(value: Any) -> Any:
        if isinstance(value,str):
            cleaned = _clean_string(value,answer)
            cleaned = _clean_fill_in_string(cleaned, answer) if fill_in else cleaned
            return _append_arithmetic_blank(cleaned,answer,fill_in)
        if isinstance(value,list):return [walk(x) for x in value]
        if isinstance(value,dict):return {k:walk(x) for k,x in value.items()}
        return value
    for name in ('l1','l2','l3','l4','l5','f','pat','u'):
        if name in result:result[name]=walk(result[name])
    return result

def _clean_fill_in_string(text: str, answer: Fraction | None = None) -> str:
    if re.search(r'代入|代回|检验|验算|检查',text) and re.search(r'\d[^。；;]*[=＝][^。；;]*\d',text):
        return '将求出的内容代回原式，检验等式是否成立。'
    if re.search(r'误认为|误以为|误填|容易误|错误地',text) and FILL_DIRECT.search(text):
        return '注意核对等号两边的关系，算完后代回原式检验。'
    if answer is not None:
        for match in FILL_RESULT.finditer(text):
            if _direct_value(match,text,answer):
                prefix = text[:match.start()].rstrip('，,：: ')
                return (prefix + '。' if prefix else '') + ('空格里应填'+BLANK+'。' if _blank_mode() else '最后一步，请你自己算一算。')
    def repl(match: re.Match[str]) -> str:
        if answer is not None and not _direct_value(match,text,answer):
            return match.group(0)
        if OPS.match(text[match.end():]):return match.group(0)
        return match.group('lead')+BLANK if _blank_mode() else '请你算一算，空格里应该填什么？'
    return FILL_DIRECT.sub(repl,text)

def protect_linear_target(output: dict[str, Any]) -> dict[str, Any]:
    blocks=output.get('question_blocks')
    if not isinstance(blocks,list):return output
    texts=[str(x.get('text','')) for x in blocks if isinstance(x,dict) and x.get('type')=='text']
    if len(texts)!=1:return output
    answer=linear_rational_solution(texts[0])
    if answer is None:answer=simple_reciprocal_solution(texts[0])
    fill_in=bool(re.match(r'^\s*填空',texts[0]))
    if answer is None:answer=arithmetic_fill_solution(texts[0])
    if answer is None and not fill_in:return output
    def walk(v: Any) -> Any:
        if isinstance(v,str):
            cleaned = _clean_string(v,answer) if answer is not None else v
            cleaned = _clean_fill_in_string(cleaned, answer) if fill_in else cleaned
            return _append_arithmetic_blank(cleaned,answer,fill_in) if answer is not None else cleaned
        if isinstance(v,list):return [walk(x) for x in v]
        if isinstance(v,dict):return {k:walk(x) for k,x in v.items()}
        return v
    for name in ('knowledge_points','base_formulas','visual_evidence','guidance','display_guidance'):
        if name in output:output[name]=walk(output[name])
    return output
