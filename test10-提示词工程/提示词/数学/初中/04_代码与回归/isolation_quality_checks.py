"""Conservative, local-only checks for the four reviewed math failures.

These are narrow prototypes, not a general mathematics or answer verifier.
They must not be used as evidence that unrecognized questions are safe.
"""
from __future__ import annotations

import ast
import copy
import math
import re
from fractions import Fraction
from typing import Any


class IsolatedQualityError(ValueError):
    """A locally recognized mathematical inconsistency prevents publication."""


def _poly(node: ast.AST) -> tuple[Fraction, Fraction, Fraction]:
    zero = Fraction(0)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = Fraction(str(node.value))
        if abs(value) > 1000000: raise ValueError('constant too large')
        return value, zero, zero
    if isinstance(node, ast.Name) and node.id == 't': return zero, Fraction(1), zero
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        p = _poly(node.operand); sign = -1 if isinstance(node.op, ast.USub) else 1
        return tuple(sign*x for x in p)
    if isinstance(node, ast.BinOp):
        a, b = _poly(node.left), _poly(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = -1 if isinstance(node.op, ast.Sub) else 1
            return tuple(x+sign*y for x,y in zip(a,b))
        if isinstance(node.op, ast.Mult):
            full = [sum(a[i]*b[k-i] for i in range(3) if 0 <= k-i < 3) for k in range(5)]
            if full[3] or full[4]: raise ValueError('degree > 2')
            return tuple(full[:3])
        if isinstance(node.op, ast.Div) and b[0] and not b[1] and not b[2]:
            return tuple(x/b[0] for x in a)
        if isinstance(node.op, ast.Pow) and b[1] == b[2] == 0 and b[0] in (0,1,2):
            if b[0] == 0:return Fraction(1),zero,zero
            if b[0] == 1:return a
            return _poly(ast.BinOp(left=node.left,op=ast.Mult(),right=node.left))
    raise ValueError('unsupported expression')


def parse_time_polynomial(text: str) -> tuple[Fraction, Fraction, Fraction]:
    """Parse a small arithmetic-only polynomial, without eval or model code."""
    text = text.strip().replace('−','-').replace('×','*').replace('÷','/')
    text = text.replace('½','(1/2)').replace('¼','(1/4)').replace('²','**2').replace('^','**')
    text = re.sub(r'\\(?:d)?frac\{(\d+)\}\{(\d+)\}',r'(\1/\2)',text)
    text = text.replace('\\times','*').replace('\\cdot','*').replace(' ','')
    text = re.sub(r'(?<=[0-9)])(?=t|\()', '*', text)
    text = re.sub(r'(?<=t)(?=\()', '*', text)
    if not text or len(text)>160 or not re.fullmatch(r'[0-9t+*/().-]+',text):raise ValueError('unsupported token')
    try: return _poly(ast.parse(text,mode='eval').body)
    except (SyntaxError,RecursionError,ZeroDivisionError,OverflowError) as exc: raise ValueError('invalid polynomial') from exc


def motion_expected(question: str) -> tuple[Fraction, Fraction, Fraction] | None:
    """Recognize only the verified axial BPQ motion described in the test.

    No inference is made for other geometries, speeds or wordings.
    """
    q=re.sub(r'\s+','',question).replace('，',',').replace('（','(').replace('）',')')
    m=re.search(r'A\(0,0\)[、,]B\((\d+(?:\.\d+)?),0\)[、,]C\(0,(\d+(?:\.\d+)?)\).*?P从A向B以每秒(\d+(?:\.\d+)?)个单位运动[，,]Q从A向C以每秒(\d+(?:\.\d+)?)个单位运动',q)
    # The reference polynomial is only valid over the stated common motion.
    # Do not infer a stopping condition from speeds or from the question type.
    if not m or not re.search(r'三角形BPQ面积',q) or '同时出发' not in q or '当P到B时停止' not in q:return None
    length,height,vp,vq=map(Fraction,m.groups())
    if min(length,height,vp,vq)<=0 or vq*length>height*vp:return None
    return Fraction(0),length*vq/2,-vp*vq/2


def _strings(value: Any):
    if isinstance(value,str): yield value
    elif isinstance(value,list):
        for item in value: yield from _strings(item)
    elif isinstance(value,dict):
        for item in value.values():yield from _strings(item)


def _area_candidates(text: str):
    # Expressions immediately following an explicit S=; unsupported parts
    # are skipped, never evaluated as Python. In an equality chain, each
    # parsable numeric polynomial must equal the reference expression.
    cleaned=text.replace('$','').replace('\\times','×').replace('\\cdot','×')
    cleaned=re.sub(r'\\(?:d)?frac\{(\d+)\}\{(\d+)\}',r'(\1/\2)',cleaned)
    # Only this exact subscript denotes the target triangle. Never turn an
    # area of a different triangle into the target area.
    cleaned=re.sub(r'S_\{\s*(?:△|\\triangle)\s*BPQ\s*\}', 'S', cleaned)
    for match in re.finditer(r'(?<![A-Za-z])S\s*(?:\(\s*t\s*\))?\s*[=＝]\s*([0-9tT\s+\-−×÷*/().²½¼^=＝]+)',cleaned):
        if match.end() < len(cleaned) and cleaned[match.end()].isascii() and cleaned[match.end()].isalpha():
            continue
        for expression in re.split(r'[=＝]',match.group(1)):
            try:yield parse_time_polynomial(expression.strip().lower())
            except ValueError:continue


def _right_circle_claim(text: str) -> bool:
    return bool(re.search(r'内切圆半径.*(?:适用于|适用|适合)任意三角形',text))


def right_triangle_radii(question: str) -> tuple[Fraction,Fraction] | None:
    q=re.sub(r'\s+','',question)
    m=re.search(r'直角三角形.*?两条直角边分别为(\d+)和(\d+)',q)
    if not m or '内切圆' not in q or '外接圆' not in q:return None
    a,b=map(int,m.groups());c=math.isqrt(a*a+b*b)
    if not a or not b or c*c!=a*a+b*b:return None
    return Fraction(a+b-c,2),Fraction(c,2)


def _number(text: str) -> Fraction:
    text=text.strip().replace('−','-')
    match=re.fullmatch(r'\\(?:d)?frac\{(\d+)\}\{(\d+)\}',text)
    if match:return Fraction(int(match.group(1)),int(match.group(2)))
    return Fraction(text)


def _radius_leak(text: str, radii: tuple[Fraction,Fraction]) -> bool:
    for m in re.finditer(r'(?<![A-Za-z0-9_])([rR])\s*[=＝]\s*([^，。；;\n$]+)',text):
        expected=radii[0] if m.group(1)=='r' else radii[1]
        for segment in re.split(r'[=＝]',m.group(2)):
            match=re.match(r'\s*([+\-−]?(?:\\(?:d)?frac\{\d+\}\{\d+\}|\d+(?:\.\d+)?(?:\s*/\s*\d+)?))(?![\d.])',segment)
            if not match or re.match(r'\s*[+\-−×÷*/^]',segment[match.end():]):continue
            try:
                number=match.group(1)
                # An uncalculated division such as 10/2 is allowed when the
                # final value is an integer; a reduced fractional final value
                # remains a direct result and must be blocked.
                if ('/' in number or '\\frac' in number or '\\dfrac' in number) and expected.denominator==1:continue
                if _number(number)==expected:return True
            except (ValueError,ZeroDivisionError):pass
    for label,expected in (('内切圆',radii[0]),('外接圆',radii[1])):
        for m in re.finditer(label+r'半径(?:是|为|等于)\s*(\d+(?:\.\d+)?(?:/\d+)?)',text):
            try:
                if _number(m.group(1))==expected:return True
            except (ValueError,ZeroDivisionError):pass
    return False


def tangent_length(question: str) -> Fraction | None:
    q=re.sub(r'\s+','',question)
    m=re.search(r'圆O的半径为(\d+).*?弦AB的长度为(\d+).*?切线.*?求(?:线段)?PA的长度',q)
    if not m:return None
    radius,chord=map(int,m.groups());half=Fraction(chord,2)
    square=Fraction(radius*radius)-half*half
    if radius<=0 or square<=0:return None
    top,bottom=math.isqrt(square.numerator),math.isqrt(square.denominator)
    if top*top!=square.numerator or bottom*bottom!=square.denominator:return None
    return Fraction(radius)*half/Fraction(top,bottom)


def _tangent_geometry_values(question: str) -> dict[str, Fraction] | None:
    """Private geometry for the same limited circle/tangent question family."""
    length=tangent_length(question)
    if length is None:return None
    q=re.sub(r'\s+','',question)
    m=re.search(r'圆O的半径为(\d+).*?弦AB的长度为(\d+)',q)
    if not m:return None
    radius=Fraction(int(m.group(1)));half=Fraction(int(m.group(2)),2)
    square=radius*radius-half*half
    offset=Fraction(math.isqrt(square.numerator),math.isqrt(square.denominator))
    op=radius*radius/offset
    return {'OA':radius,'OB':radius,'AH':half,'BH':half,'OH':offset,
            'PA':length,'PB':length,'OP':op,'HP':op-offset}


def _geometry_expression(node: ast.AST, values: dict[str,Fraction]) -> Fraction:
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):
        number=Fraction(str(node.value))
        if abs(number)>1000000:raise ValueError('constant too large')
        return number
    if isinstance(node,ast.Name) and node.id in values:return values[node.id]
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        value=_geometry_expression(node.operand,values)
        return -value if isinstance(node.op,ast.USub) else value
    if isinstance(node,ast.BinOp):
        a,b=_geometry_expression(node.left,values),_geometry_expression(node.right,values)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Sub):return a-b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Div) and b:return a/b
        if isinstance(node.op,ast.Pow) and b in (0,1,2):return a**int(b)
    raise ValueError('unsupported geometry expression')


def _eval_geometry(text: str, values: dict[str,Fraction]) -> Fraction:
    text=text.strip().replace('−','-').replace('×','*').replace('÷','/')
    text=text.replace('²','**2').replace('^','**').replace(' ','')
    text=text.replace('\\times','*').replace('\\cdot','*')
    text=re.sub(r'\\(?:d)?frac\{([A-Za-z0-9]+)\}\{([A-Za-z0-9]+)\}',r'(\1/\2)',text)
    text=re.sub(r'(?<=[0-9)])(?=[A-Z(])','*',text)
    if not text or len(text)>120 or not re.fullmatch(r'[A-Z0-9+*/().-]+',text):raise ValueError('unsupported tokens')
    try:return _geometry_expression(ast.parse(text,mode='eval').body,values)
    except (SyntaxError,RecursionError,ZeroDivisionError,OverflowError) as exc:raise ValueError('invalid geometry expression') from exc


def _tangent_geometry_audit(document: dict[str,Any], question: str) -> dict[str,Any]:
    """Check explicit recognized line equations; mark missing proof unverified.

    H is only assigned midpoint values when the model defines it accordingly.
    Failure to parse a claim is not evidence that the claim is correct.
    """
    values=_tangent_geometry_values(question)
    if values is None:return {'recognized':False,'checked':0,'contradiction':False,'proof_checked':False}
    strings=list(_strings(document)); all_text='\n'.join(strings)
    h_defined=bool(re.search(r'(?:H.{0,12}(?:中点|垂足)|(?:中点|垂足).{0,12}H|OH\s*⊥\s*AB(?:于H)?|OP垂直平分AB[，,]设交点为H|设H为OP与AB的交点.{0,40}OP垂直平分AB)',all_text))
    if not h_defined:
        values={k:v for k,v in values.items() if 'H' not in k}
    # Segment lengths are direction-independent. Keep this local to lengths;
    # do not normalize directed vectors or angular expressions this way.
    values.update({key[::-1]:value for key,value in list(values.items())})
    contradiction=False;checked=0;proof=False
    for text in strings:
        compact=re.sub(r'\s+','',text)
        # In this narrowly recognized nondegenerate two-tangent construction,
        # H is on segment OP. Do not confuse an explicit warning against the
        # false assertion with the false assertion itself.
        false_part=re.search(r'(?<!认为)(?<!以为)(?<!说成)OH(?:是弦心距[，,]?)?(?:并)?(?:不是|不属于)OP(?:的)?(?:一部分|部分)',compact)
        if h_defined and false_part:
            contradiction=True
        # These are the observed incorrect claims. Do not block a warning
        # which explicitly says the relation is wrong or cannot be used.
        if not re.search(r'(?:不是|不能|不可|不等于|不能认为|不可认为|误以为|误认为|错误地认为)',compact):
            if re.search(r'PA(?:是|为|作为)(?:直角三角形OAP的|三角形OAP的|其中的)?斜边|斜边(?:为|是)PA',compact):contradiction=True
            if 'OP' in compact and '弦心距' in compact and '半径' in compact and re.search(r'和或差|之和或差|半径.{0,12}(?:加|减)弦心距',compact):contradiction=True
        normalized=text.replace('$','').replace('\\times','×').replace('\\cdot','×')
        normalized=re.sub(r'\\(?:d)?frac\{([A-Za-z0-9]+)\}\{([A-Za-z0-9]+)\}',r'(\1/\2)',normalized)
        # A whole equality must consist of arithmetic and recognized segment
        # labels. Avoid substring checks that would misread OA²+PA²=OP².
        pattern=r'(?<![A-Za-z0-9_])([A-Z0-9\s+\-−×÷*/().²^]+?)\s*[=＝]\s*([A-Z0-9\s+\-−×÷*/().²^]+)'
        for match in re.finditer(pattern,normalized):
            left,right=match.group(1).strip(),match.group(2).strip()
            labels=re.findall(r'[A-Z]+',left+' '+right)
            canonical={name if name in ('OA','OB','OH','AH','BH','HP','OP','PA','PB') else name[::-1] for name in labels}
            if not canonical.intersection(('OA','OB','OH','AH','BH','HP','OP','PA','PB')):continue
            try:a,b=_eval_geometry(left,values),_eval_geometry(right,values)
            except ValueError:continue
            checked+=1
            if a!=b:contradiction=True
            if {'PA','OA','AH','OH'}.issubset(canonical):proof=True
    return {'recognized':True,'checked':checked,'contradiction':contradiction,'proof_checked':proof}


def _tangent_leak(text: str, expected: Fraction) -> bool:
    pat=r'(?:(?:PA|AP)\s*[=＝]|(?:PA|AP)的长度(?:为|是|等于))\s*([+\-]?(?:\\(?:d)?frac\{\d+\}\{\d+\}|\d+(?:\.\d+)?(?:\s*/\s*\d+)?))'
    for match in re.finditer(pat,text,re.I):
        if re.match(r'\s*[+\-−×÷*/^]',text[match.end():]):continue
        try:
            raw=match.group(1)
            if ('/' in raw or '\\frac' in raw or '\\dfrac' in raw) and expected.denominator==1:continue
            if _number(raw)==expected:return True
        except (ValueError,ZeroDivisionError):pass
    return False


def _valid_nonnegative_parabola(question: str) -> bool:
    q=re.sub(r'\s+','',question).replace('，',',')
    m=re.search(r'y=-x[²2]\+(\d+)x.*?A\(0,0\).*?B\((\d+),0\)',q)
    return bool(m and m.group(1)==m.group(2) and '之间' in q and '抛物线' in q)


def _parabola_target(question: str) -> tuple[Fraction,Fraction,Fraction] | None:
    if not _valid_nonnegative_parabola(question) or '面积' not in question or '坐标' not in question:return None
    match=re.search(r'y\s*=\s*-x[²2]\s*\+\s*(\d+)x',question)
    if not match:return None
    b=Fraction(match.group(1))
    return b/2,b*b/4,b*b*b/8


def _parabola_area_expected(question: str) -> tuple[Fraction,Fraction,Fraction] | None:
    """Independent area polynomial for one explicitly described PAB family."""
    if not _valid_nonnegative_parabola(question):return None
    q=re.sub(r'\s+','',question)
    if '三角形PAB面积' not in q or 'P在A、B之间的抛物线上' not in q:return None
    m=re.search(r'y=-x[²2]\+(\d+)x',q)
    if not m or int(m.group(1))<=0:return None
    b=Fraction(m.group(1))
    return Fraction(0),b*b/2,-b/2


def _parabola_leak(text: str, target: tuple[Fraction,Fraction,Fraction]) -> bool:
    x,y,area=target
    for m in re.finditer(r'[（(]\s*(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\s*[）)]',text):
        if Fraction(m.group(1))==x and Fraction(m.group(2))==y:return True
    for m in re.finditer(r'(?:最大面积|面积最大值|最大值)(?:为|是|等于|[=＝])\s*(\d+(?:\.\d+)?)',text):
        if Fraction(m.group(1))==area:return True
    for match in re.finditer(r'(?:顶点横坐标|最大值对应的横坐标|面积最大时的横坐标)\s*(?:为|是|等于)?\s*(?:x\s*[=＝]\s*)?(\d+(?:\.\d+)?)',text,re.I):
        if Fraction(match.group(1))==x:return True
    for match in re.finditer(r'(?:S|面积).{0,10}?(?:当|在|于)\s*x\s*[=＝]\s*(\d+(?:\.\d+)?)\s*时.{0,10}(?:最大|顶点)',text,re.I):
        if Fraction(match.group(1))==x:return True
    # A completed numerical equality S=...=value is a direct area answer.
    # A correct vertex-form function with x remains allowed, even when its
    # constants permit the student to infer the maximum.
    for match in re.finditer(r'(?<![A-Za-z0-9_])S\s*[=＝]\s*([^,，。；;\n$]+)',text):
        final=re.split(r'[=＝]',match.group(1))[-1].strip()
        number=re.fullmatch(r'([+\-−]?\d+(?:\.\d+)?)(?:\s*(?:平方单位|平方厘米|平方分米|平方毫米))?',final)
        if number and Fraction(number.group(1).replace('−','-'))==area:return True
    # In one sentence, giving both requested coordinates separately is still
    # an explicit result, even if the model never writes the ordered pair.
    def assigned(var: str, expected: Fraction) -> bool:
        for match in re.finditer(r'(?<![A-Za-z0-9_])'+var+r'\s*[=＝]\s*([^,，。；;\n$]+)',text):
            last=re.split(r'[=＝]',match.group(1))[-1].strip()
            number=re.match(r'([+\-−]?\d+(?:\.\d+)?)(?![\d.])',last)
            if number and not re.match(r'\s*[+\-−×÷*/^]',last[number.end():]):
                if Fraction(number.group(1).replace('−','-'))==expected:return True
        return False
    if assigned('x',x) and assigned('y',y):return True
    return False


def refine_tangent_explanation(output: dict[str,Any], question: str, quality: dict[str,Any]) -> tuple[dict[str,Any],int]:
    """Repair one observed filler pattern only for the exact verified family.

    This doesn't validate arbitrary geometry prose or other parts of a reply.
    """
    q=re.sub(r'\s+','',question)
    if not re.search(r'过A、B分别作圆O的切线[，,]交于点P',q):return output,0
    if quality.get('issues') or not quality.get('tangent_proof_checked'):return output,0
    result=copy.deepcopy(output);changed=0
    groups=[]
    for section,key in (('display_guidance','novice'),('guidance','foundation')):
        block=result.get(section)
        if isinstance(block,dict) and isinstance(block.get(key),dict):groups.append(block[key])
    for group in groups:
        for part in group.get('parts',[]):
            steps=part.get('steps') if isinstance(part,dict) else None
            if not isinstance(steps,list) or len(steps)<2:continue
            if steps[-1] != '先整理已知条件、所求对象和限制范围。':continue
            prev=str(steps[-2]);prior='\n'.join(str(x) for x in steps[:-2])
            if not all(x in prev for x in ('OH','OA','AH')) or not ('AP' in prev or 'PA' in prev):continue
            if not re.search(r'OH\s*⊥\s*AB\s*于H|H(?:是|为)(?:弦)?AB的中点',prior) or '相似' in prior:continue
            explanation='由OA=OB、PA=PB可知OP垂直平分AB，所以H在OP上；再结合OA⊥PA、OH⊥AB，可得△OHA∽△OAP。'
            previous=re.sub(r'^检查[：:]\s*相似比对应边是否写对[？?]\s*','',prev)
            steps[-2:]=[explanation,previous]
            changed+=1
    return result,changed


def prepare_local_quality(document: dict[str,Any], question: str) -> tuple[dict[str,Any],dict[str,Any]]:
    """Repair narrow wording errors and reject recognized wrong motion algebra."""
    # One observed question is ambiguous and pedagogically misleading: it
    # suggests a relation between the constant denominator and the sides.
    # Replace only this exact sentence, preserving independent bad-claim checks.
    document=copy.deepcopy(document)
    ambiguous=('内切圆半径公式中的分母2与直角边、斜边是什么关系？外接圆半径又对应哪条边？',
               '内切圆半径公式中的分母2与直角边或斜边有什么关系？外接圆半径又对应哪条边？')
    replacement='内切圆半径公式中的分母是什么？分子如何由两条直角边和斜边组成？外接圆半径又对应哪条边？'
    changed=False
    def clarify(v: Any) -> Any:
        nonlocal changed
        if isinstance(v,str) and v in ambiguous:
            changed=True
            return replacement
        if isinstance(v,list):return [clarify(x) for x in v]
        if isinstance(v,dict):return {k:clarify(x) for k,x in v.items()}
        return v
    if right_triangle_radii(question) is not None:document=clarify(document)
    tangent_wording_changes=[]
    wording={
        '垂径定理：圆心与切线交点的连线垂直平分弦，联系弦长与半弦。':
            '由OA=OB和切线长PA=PB，O与P都在AB的垂直平分线上，因此OP垂直平分AB。',
        '由OA⊥PA可知△OAP是直角三角形，接下来应把半径、半弦和切线长放进同一个直角三角形中。':
            '由OA⊥PA可知△OAP是直角三角形；再通过另一个直角三角形与它的相似关系，联系半径、半弦和切线长。',
    }
    def clarify_tangent(v: Any) -> Any:
        if isinstance(v,str) and v in wording:
            tangent_wording_changes.append('澄清切线与半弦的几何依据')
            return wording[v]
        if isinstance(v,list):return [clarify_tangent(x) for x in v]
        if isinstance(v,dict):return {k:clarify_tangent(x) for k,x in v.items()}
        return v
    if tangent_length(question) is not None:document=clarify_tangent(document)
    expected=motion_expected(question)
    issues=[];checked=0
    radii=right_triangle_radii(question)
    if radii is not None and any(_radius_leak(value,radii) for value in _strings(document)):
        issues.append('模型直接写出所求圆的半径')
    if radii is not None and any(re.search(r'内切圆半径.{0,20}分母(?:和|与|跟)斜边(?:有关|相关)|内切圆半径.{0,20}分母和分子分别对应.{0,15}(?:哪些边|各边)|内切圆半径.{0,25}分母.{0,20}(?<!不)对应(?:的是|的|是)?(?:哪条|什么)边',value) for value in _strings(document)):
        issues.append('内切圆半径公式的分母表述有误')
    tangent=tangent_length(question)
    if tangent is not None and any(_tangent_leak(value,tangent) for value in _strings(document)):
        issues.append('模型直接写出所求切线长度')
    if tangent is not None and any(re.search(r'需另法|但需.{0,24}进一步推导|仍需另寻',value) for value in _strings(document)):
        issues.append('切线题尚未建立可计算的关系')
    geometry=_tangent_geometry_audit(document,question)
    if geometry['contradiction']:
        issues.append('切线题存在与题设矛盾的几何关系')
    elif geometry['recognized'] and not geometry['proof_checked']:
        issues.append('切线题关键几何关系尚未得到独立核对')
    parabola_target=_parabola_target(question)
    if parabola_target is not None and _parabola_leak('\n'.join(_strings(document)),parabola_target):
        issues.append('模型直接写出所求动点坐标或最大面积')
    parabola_expected=_parabola_area_expected(question);parabola_checked=0
    if parabola_expected is not None:
        for value in _strings(document):
            for candidate in _area_candidates(re.sub(r'[xX]','t',value)):
                if not candidate[1] and not candidate[2]:continue
                parabola_checked+=1
                if candidate != parabola_expected:
                    issues.append('抛物线动点面积表达式与题设不一致')
    if expected is not None:
        if '当P到B时停止' in question and any(re.search(r'P.{0,12}(?:到|达)B.{0,20}Q(?:还|仍|继续)(?:在)?运动',value) for value in _strings(document)):
            issues.append('终止条件与继续运动的描述不一致，需核对')
        for value in _strings(document):
            for candidate in _area_candidates(value):
                if not candidate[1] and not candidate[2]:continue
                checked+=1
                if candidate != expected:issues.append('动点面积展开与底乘高不一致')
        if checked==0:
            issues.append('未能验证动点面积表达式')
        target_t=-expected[1]/(2*expected[2])
        for value in _strings(document):
            # This detects explicitly labelled scalar maximum answers only.
            # It does not establish that free text or other answer forms are safe.
            if re.search(r'最大值|最大面积', question):
                maximum = r'(?:最大面积|面积的最大值|面积最大值|最大值|S_?max)\s*(?:为|是|等于|[=＝])\s*([+\-−]?\d+(?:\.\d+)?)'
                for match in re.finditer(maximum, value, re.I):
                    if not re.match(r'\s*[\d.+\-−×÷*/^]', value[match.end():]):
                        issues.append('模型直接写出所求的最大面积')
            if re.search(r'时间|时刻', question):
                time_result = r'(?:最大值|最大面积|面积最大)(?:时|在|对应|取得|发生在|出现在|于)*\s*(?:时刻|时间)?\s*t\s*[=＝]\s*([+\-−]?\d+(?:\.\d+)?)\s*(?:时取得|时达到|时获得|秒时取得|秒时达到)'
                if re.search(time_result, value, re.I):
                    issues.append('模型直接写出所求的最值时间')
            for match in re.finditer(r'(?<![A-Za-z0-9_])t\s*[=＝]\s*(\d+(?:\.\d+)?)(?![\d.])',value):
                if Fraction(match.group(1))==target_t and not re.match(r'\s*[+\-−×÷*/^]',value[match.end():]):
                    issues.append('模型直接写出所求的最值时间')
            for match in re.finditer(r'(?<![A-Za-z0-9_])t\s*[=＝]\s*([^,，。；;\n$]+)',value):
                final=re.split(r'[=＝]',match.group(1))[-1].strip()
                number=re.match(r'([+\-−]?\d+(?:\.\d+)?)(?![\d.])',final)
                if number and not re.match(r'\s*[+\-−×÷*/^]',final[number.end():]):
                    if Fraction(number.group(1).replace('−','-'))==target_t:
                        issues.append('模型直接写出所求的最值时间')
    result=copy.deepcopy(document);actions=(['澄清内切圆公式分母的引导问句'] if changed else [])+tangent_wording_changes
    right_triangle='直角三角形' in question
    parabola=_valid_nonnegative_parabola(question)
    def walk(v: Any) -> Any:
        if isinstance(v,str):
            if right_triangle and _right_circle_claim(v):
                actions.append('修正直角三角形内切圆公式适用条件')
                return '直角三角形中，a、b为直角边，c为斜边时，内切圆半径r=(a+b-c)/2。'
            if parabola and re.search(r'误将.*2\s*y.*(?:2\s*\|\s*y\s*\||绝对值)',v,re.I):
                actions.append('修正非负区间内面积表达式的易错提示')
                return '本题区间内y非负，可以用y表示高；若点可能位于x轴下方，要根据位置取绝对值。'
            if parabola and re.search(r'(?:不要|不能|不应)(?:误)?(?:加|使用|写|取|保留)绝对值',v):
                actions.append('修正把绝对值等价写法误判为错误的提示')
                return '本题区间内y非负，高可以写成y，也可以写成|y|。'
            return v
        if isinstance(v,list):return [walk(x) for x in v]
        if isinstance(v,dict):return {k:walk(x) for k,x in v.items()}
        return v
    return walk(result),{'motion_question_recognized':expected is not None,'right_triangle_radii_recognized':radii is not None,'tangent_question_recognized':tangent is not None,'tangent_geometry_checked':geometry['checked'],'tangent_proof_checked':geometry['proof_checked'],'parabola_target_recognized':parabola_target is not None,'parabola_area_expressions_checked':parabola_checked,'area_expressions_checked':checked,'issues':list(dict.fromkeys(issues)),'repairs':list(dict.fromkeys(actions))}
