"""Use the packaged DOCX rasterizer with an already exported native Word PDF."""
from pathlib import Path
import os,sys,importlib.util
P=Path(__file__).resolve().parent
R=Path(r'C:/Users/robot/.cache/codex-runtimes/codex-primary-runtime')
os.environ['PATH']=str(R/'dependencies/native/poppler/Library/bin')+os.pathsep+os.environ['PATH']
s=importlib.util.spec_from_file_location('packaged_renderer',R/'plugins/openai-primary-runtime/plugins/documents/skills/documents/render_docx.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.convert_to_pdf=lambda *a,**k:(str(P/'word_qa/preview.pdf'),'Native Microsoft Word ExportAsFixedFormat; LibreOffice unavailable')
m.rasterize(sys.argv[1] if len(sys.argv)>1 else str(P/'搜题工程六图流程模型测试报告.docx'),str(P/'word_qa'),120,False,False)
