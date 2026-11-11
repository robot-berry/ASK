"""Small, local-only answer blanks for one independently solved question family.

No generic question understanding, difficulty classification, or leakage
guarantee. The caller must still run the other mathematical quality checks.
"""
from __future__ import annotations
import copy
import ast
import math
import re
from fractions import Fraction
from typing import Any
from isolation_quality_checks import right_triangle_radii, _eval_geometry
from isolation_linear_answer_guard import linear_rational_solution, simple_reciprocal_solution, arithmetic_fill_solution, _expr

BLANK = '————'
NUM = r'[+\-−]?(?:\\(?:d)?frac\{\d+\}\{\d+\}|\d+(?:\.\d+)?(?:\s*/\s*\d+)?)'
TAIL = re.compile(r'\s*[+\-−×÷*/^]')
CHAIN = re.compile(r'(?<![A-Za-z0-9_])(?P<var>[rR])\s*[=＝]\s*(?P<work>[A-Za-z0-9\\{}().+\-−×÷*/^²\s]{1,120}?[=＝])\s*(?P<num>'+NUM+r')(?![A-Za-z0-9_.])')
ASSIGN = re.compile(r'(?<![A-Za-z0-9_])(?P<var>[rR])\s*[=＝]\s*(?P<num>'+NUM+r')(?![A-Za-z0-9_.])')
PROSE = re.compile(r'(?P<label>内切圆|外接圆)半径(?:是|为|等于)\s*(?P<num>'+NUM+r')(?![A-Za-z0-9_.])')
NUMERIC_RHS = re.compile(r'(?<![A-Za-z0-9_])(?P<var>[rR])\s*[=＝]\s*(?P<rhs>[0-9.\s+\-−×÷*/()²^=＝]+)')
LINEAR_RHS = re.compile(r'(?<![A-Za-z0-9_])(?P<var>x|方框|空格|括号里的数)\s*[=＝]\s*(?P<rhs>[0-9.\s+\-−×÷*/()=＝]+)',re.I)

def audit_supported_linear_work(document: dict[str,Any], question: str) -> dict[str,Any]:
    """Check explicitly assigned, numeric candidate answers for supported tasks."""
    answer=linear_rational_solution(question)
    if answer is None:answer=simple_reciprocal_solution(question)
    fill=bool(re.match(r'^\s*填空',str(question or '')))
    if answer is None:answer=arithmetic_fill_solution(question)
    if answer is None:return {'recognized':False,'checked':0,'issues':[]}
    checked=0;issues=[]
    def walk(v: Any):
        if isinstance(v,str):yield v
        elif isinstance(v,list):
            for x in v:yield from walk(x)
        elif isinstance(v,dict):
            for x in v.values():yield from walk(x)
    for key in ('l1','l2','l3','l4','l5','f','pat','u'):
        for text in walk(document.get(key)):
            for match in LINEAR_RHS.finditer(text):
                if not fill and match.group('var').lower()!='x':continue
                if match.end()<len(text) and (text[match.end()].isalnum() or text[match.end()] in '\\^'):
                    continue
                before=text[max(0,match.start()-8):match.start()]
                after=text[match.end():]
                if re.search(r'(?:当|假设|例如)\s*$',before) and re.match(r'\s*时',after):continue
                for item in re.split(r'[=＝]',match.group('rhs')):
                    expression=item.strip().replace('−','-').replace('×','*').replace('÷','/')
                    if not expression or len(expression)>120:continue
                    try:a,b=_expr(ast.parse(expression,mode='eval').body)
                    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):continue
                    if a:continue
                    checked+=1
                    if b!=answer:issues.append('所写未知数结果与题设计算不一致')
    return {'recognized':True,'checked':checked,'issues':list(dict.fromkeys(issues))}

def _number(s: str) -> Fraction:
    s=re.sub(r'\s+','',s).replace('−','-')
    m=re.fullmatch(r'([+\-]?)\\(?:d)?frac\{(\d+)\}\{(\d+)\}',s)
    if m:return (-1 if m.group(1)=='-' else 1)*Fraction(int(m.group(2)),int(m.group(3)))
    return Fraction(s)

def mask_supported_circle_answers(document: dict[str,Any], question: str) -> tuple[dict[str,Any],dict[str,Any]]:
    radii=right_triangle_radii(question)
    if radii is None:return document,{'recognized':False,'masked':0,'issues':[]}
    result=copy.deepcopy(document);masked=0;issues=[]
    def walk(v: Any) -> Any:
        nonlocal masked
        if isinstance(v,str):
            text=re.sub(r'(?<![A-Za-z0-9_])_{3,}',BLANK,v)
            # Check only arithmetic expressions we can evaluate exactly; an
            # unsupported expression is unknown, never evidence of correctness.
            for expression in NUMERIC_RHS.finditer(text):
                if expression.end() < len(text) and (text[expression.end()].isalpha() or text[expression.end()] == '\\'):
                    continue
                expected=radii[0] if expression.group('var')=='r' else radii[1]
                for segment in re.split(r'[=＝]',expression.group('rhs')):
                    try:value=_eval_geometry(segment.strip(),{})
                    except ValueError:continue
                    if value!=expected:issues.append('所写圆半径与题设计算不一致')
            def candidate(match: re.Match[str], expected: Fraction, prefix: str) -> str:
                nonlocal masked
                if TAIL.match(text[match.end():]):return match.group(0)
                raw=match.group('num')
                try:value=_number(raw)
                except (ValueError,ZeroDivisionError):return match.group(0)
                # An arithmetic division expression may be left for students;
                # existing rules treat a reduced fractional final as direct.
                if ('/' in raw or '\\frac' in raw or '\\dfrac' in raw) and expected.denominator==1:
                    return match.group(0)
                if value != expected:
                    issues.append('所写圆半径与题设计算不一致')
                    return match.group(0)
                masked+=1
                return prefix+BLANK
            def chain(match: re.Match[str]) -> str:
                expected=radii[0] if match.group('var')=='r' else radii[1]
                prefix=match.group(0)[:match.start('num')-match.start()]
                return candidate(match,expected,prefix)
            text=CHAIN.sub(chain,text)
            def assign(match: re.Match[str]) -> str:
                expected=radii[0] if match.group('var')=='r' else radii[1]
                prefix=match.group(0)[:match.start('num')-match.start()]
                return candidate(match,expected,prefix)
            text=ASSIGN.sub(assign,text)
            def prose(match: re.Match[str]) -> str:
                expected=radii[0] if match.group('label')=='内切圆' else radii[1]
                prefix=match.group(0)[:match.start('num')-match.start()]
                return candidate(match,expected,prefix)
            return PROSE.sub(prose,text)
        if isinstance(v,list):return [walk(x) for x in v]
        if isinstance(v,dict):return {k:walk(x) for k,x in v.items()}
        return v
    for key in ('l1','l2','l3','l4','l5','f','pat','u'):
        if key in result:result[key]=walk(result[key])
    return result,{'recognized':True,'masked':masked,'issues':list(dict.fromkeys(issues))}


def add_supported_circle_exercise(document: dict[str,Any], question: str) -> tuple[dict[str,Any],bool]:
    """Add a verified final expression only to a single-part Pythagorean case."""
    if right_triangle_radii(question) is None:return document,False
    q=re.sub(r'\s+','',question)
    match=re.fullmatch(r'(?:初三数学[:：])?直角三角形的两条直角边分别为(\d+)和(\d+)[，,]求这个三角形内切圆和外接圆的半径[。.!！]?',q)
    if not match:return document,False
    a,b=map(int,match.groups());c=math.isqrt(a*a+b*b)
    if c*c!=a*a+b*b:return document,False
    result=copy.deepcopy(document);parts=result.get('l5')
    if not isinstance(parts,list) or len(parts)!=1 or not isinstance(parts[0],dict):return document,False
    steps=parts[0].get('s')
    if not isinstance(steps,list) or not steps or not all(isinstance(x,str) for x in steps):return document,False
    if any(BLANK in x or re.search(r'_{3,}',x) for x in steps):return document,False
    original=steps[-1].rstrip()
    separator='' if original.endswith(('。','！','？')) else '。'
    steps[-1]=original+separator+f'斜边c={c}，代入得r=({a}+{b}-{c})÷2={BLANK}，R={c}÷2={BLANK}，请你自己算一算。'
    return result,True


def add_supported_linear_exercise(document: dict[str,Any], question: str) -> tuple[dict[str,Any],bool]:
    """Add one visible blank to a supported single equation or one-blank task."""
    fill=bool(re.match(r'^\s*填空',str(question or '')))
    answer=arithmetic_fill_solution(question) if fill else linear_rational_solution(question)
    if answer is None:return document,False
    result=copy.deepcopy(document);parts=result.get('l5')
    if not isinstance(parts,list) or len(parts)!=1 or not isinstance(parts[0],dict):return document,False
    steps=parts[0].get('s')
    if not isinstance(steps,list) or not steps or not all(isinstance(x,str) for x in steps):return document,False
    if any(BLANK in x or re.search(r'_{3,}',x) for x in steps):return document,False
    original=steps[-1].rstrip()
    separator='' if original.endswith(('。','！','？')) else '。'
    hint='请在横线上填出所求的数：————。' if fill else '请在横线上填出x的值：x=————。'
    steps[-1]=original+separator+hint
    return result,True
