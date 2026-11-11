from pathlib import Path
from docx import Document
import json,re
root=Path(r'D:\robot\Documents\ASK\picture\test6\txt')
src=root/'专利初稿_题图结构恢复与分阶段发布.docx'
out=root/'专利初稿_题图结构恢复与分阶段发布_精简稿.docx'
d=Document(src)
changes=[]
pairs=[
('Word整理日期：2026年10月14日','修订日期：2026年10月29日'),
('上述背景描述用于说明技术问题，不代表已检索确认的最接近现有技术，具体对比文件应在专利检索后补充。',''),
('预处理操作按图像情况选择，不要求每张图像均发生透视变换。',''),
('，而非单纯采用字符更多的候选',''),
('，因此早期发布不等于所有引导图片永久不可改变',''),
('是否降低总体耗时及降低幅度取决于输入图像分布、模型实现与部署条件。',''),
('但保留原图并不表示文字识别完全正确，也不自动保证模型理解正确。',''),
('；这些名称用于说明具体实现，技术方案不依赖某一供应商名称',''),
('该调度为系统实施的配套方式，不作为本申请核心空间融合机制成立的前提。',''),
('该数值为工程参数示例，并不表示对所有设备和图像均为最优阈值。',''),
('相关符号仅用于触发检查，不直接证明当前识别错误，也不作为推导题目答案的依据。','相关符号用于触发检查。'),
('，不是全部候选均按同一长度比例处理',''),
('，因此默认拒绝条件不解释为所有路径的绝对限制',''),
('该估计是候选定位方法，不等价于逐字检测所得的真实边界。',''),
('，不能声称完成了正常OCR重排',''),
('这些参数为实施例中的终端规格，不限制核心方法只能在该宽度运行。',''),
('该内容组织是应用实施方式，技术方案关注字段完整性检测及图片发布过程。',''),
('这些阈值是发布校验参数，不等同于对教学内容的普遍限定。',''),
('，不把正常的逐档增长示例解释为所有情况下均严格连续',''),
('规则校验只能提供对已定义模式的检测能力，不能据此宣称任意生成内容都绝不会泄露答案或包含错误。',''),
('识别质量、内容质量和图片文件规格属于不同检查层次，规格通过不等于题意识别或引导推理完全正确。',''),
('实施例七 工程记录及验证边界','实施例七 性能记录与验证方法'),
('[0058] 上述数据是已有报告记录，不是本次重新实验结果，也不是当前6.3.6全部新增机制的独立对照证明。报告明确区分自动门槛和人工语义完整率。不能将300/300解释为文字识别准确率100%，也不能把历史整体时延改善全部归因于本草稿中的某一个特征。','[0058] 上述时延数据对应报告所记录的整轮优化；298/300和300/300分别表示文字自动结构门槛和文字加必要配图的语义包自动门槛通过数。人工语义完整率及各项机制的独立贡献需分别通过人工复核和对照实验测定。'),
('本次撰写仅查阅相关代码和记录，未执行这些测试。',''),
]
for old,new in pairs:
    found=False
    for p in d.paragraphs:
        if old in p.text:
            before=p.text
            p.text=before.replace(old,new)
            changes.append({'before':old,'after':new})
            found=True
    if not found: raise ValueError('Missing text: '+old)
# Keep substantive limitation and scope language in claims unchanged.
a=Document(src)
def claims(doc):
    ps=doc.paragraphs; start=next(i for i,p in enumerate(ps) if p.text=='权利要求书'); end=next(i for i,p in enumerate(ps) if p.text=='说明书')
    return [p.text for p in ps[start:end]]
assert claims(a)==claims(d)
assert len(d.inline_shapes)==4
assert len(re.findall(r'^\[\d{4}\]', '\n'.join(p.text for p in d.paragraphs),re.M))==59
for rootel in [d.element,d.styles.element]:
    for node in rootel.xpath('.//w:pBdr'):node.getparent().remove(node)
d.save(out)
work=root/'word_work'/'concise_20261029';work.mkdir(exist_ok=True)
(work/'edits.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
print('Edited passages:',len(changes))
print('Removed characters:',sum(len(x['before'])-len(x['after']) for x in changes))
print('Claims unchanged; 59 numbered paragraphs; 4 figures retained')
