from pathlib import Path
import importlib.util,os
P=Path(__file__).resolve().parent
R=Path(r'C:/Users/robot/.cache/codex-runtimes/codex-primary-runtime')
os.environ['PATH']=str(R/'dependencies/native/poppler/Library/bin')+os.pathsep+os.environ['PATH']
s=importlib.util.spec_from_file_location('packaged_renderer',R/'plugins/openai-primary-runtime/plugins/documents/skills/documents/render_docx.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.convert_to_pdf=lambda *a,**k:(str(P/'word_qa/preview.pdf'),'Native Microsoft Word PDF export; LibreOffice unavailable on this host')
m.rasterize(str(P/'搜题工程三轮模型测试汇总报告_宋体版.docx'),str(P/'word_qa'),120,False,False)
