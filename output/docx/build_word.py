from pathlib import Path
import re, runpy
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPM
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
root=Path.cwd(); out=root/'output/docx'; assets=out/'assets'; assets.mkdir(exist_ok=True)
SimpleDocTemplate.multiBuild=lambda *a,**kw: None
ns=runpy.run_path(str(root/'output/pdf/fingerprint_report/build_report.py'))
doc=Document(); sec=doc.sections[0]; sec.page_width= Inches(8.27); sec.page_height= Inches(11.69); sec.left_margin= Inches(1.22); sec.right_margin= Inches(.98)
for st in ['Normal','Title','Heading 1','Heading 2','Heading 3','Caption']:
 s=doc.styles[st]; s.font.name='Times New Roman'; s.font.color.rgb=RGBColor(0,0,0)
doc.styles['Normal'].font.size=Pt(11); doc.styles['Normal'].paragraph_format.line_spacing=1.25

def plain(t):
 import html
 return html.unescape(re.sub('<[^>]+>','',t.replace('<br/>','\n')))
def para(t,style=None): return doc.add_paragraph(plain(t),style)
def picture(p,caption):
 doc.add_picture(str(p),width= Inches(5.95)); para(caption,'Caption')
fig=0
for item in ns['story']:
 if isinstance(item,PageBreak): doc.add_page_break()
 elif isinstance(item,Paragraph):
  name=item.style.name; st={'ChapterX':'Heading 1','SectionX':'Heading 2','SubX':'Heading 3','CaptionX':'Caption','CoverX':'Title'}.get(name)
  para(item.text,st)
 elif isinstance(item,Table):
  t=doc.add_table(rows=0,cols=len(item._cellvalues[0])); t.style='Table Grid'
  for row in item._cellvalues:
   cells=t.add_row().cells
   for c,v in zip(cells,row): c.text=plain(v.text if hasattr(v,'text') else str(v))
 elif isinstance(item,Drawing):
  fig+=1
  try:
   p=assets/f'diagram_{fig}.png'; renderPM.drawToFile(item,str(p),fmt='PNG',dpi=180); doc.add_picture(str(p),width= Inches(5.9))
  except Exception:
   from PIL import Image, ImageDraw, ImageFont
   im=Image.new('RGB',(int(item.width*3),int(item.height*3)), 'white'); dr=ImageDraw.Draw(im); font=ImageFont.truetype('C:/Windows/Fonts/times.ttf',26)
   for v in item.contents:
    if type(v).__name__=='Rect': dr.rectangle((v.x*3,(item.height-v.y-v.height)*3,(v.x+v.width)*3,(item.height-v.y)*3),fill='#edf2f5',outline='#374e5b',width=2)
    elif type(v).__name__=='Line': dr.line((v.x1*3,(item.height-v.y1)*3,v.x2*3,(item.height-v.y2)*3),fill='black',width=2)
   for v in item.contents:
    if hasattr(v,'text'): dr.text((v.x*3,(item.height-v.y)*3),v.text,font=font,fill='black',anchor='ms')
   p=assets/f'diagram_{fig}.png'; im.save(p); doc.add_picture(str(p),width= Inches(5.9))
 elif hasattr(item,'_content'):
  for sub in item._content:
   if isinstance(sub,Paragraph): para(sub.text,'Caption')
   elif isinstance(sub,Table):
    t=doc.add_table(rows=0,cols=len(sub._cellvalues[0])); t.style='Table Grid'
    for row in sub._cellvalues:
     for c,v in zip(t.add_row().cells,row): c.text=plain(v.text if hasattr(v,'text') else str(v))
 elif type(item).__name__=='TableOfContents':
  p=doc.add_paragraph(); fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'TOC \\o "1-2" \\h' if item._notifyKind=='TOCEntry' else 'TOC \\t "Caption,1"'); p._p.append(fld)
# Detailed architecture supplement
para('Appendix C Model architecture diagrams and implementation references','Heading 1')
para('The following diagrams are original redrawings of the architectures used by the project. They show the repository adaptations rather than reproducing published figures. Input is a batch of three channel 128 by 128 images. Source for every projection and normalisation step is src/models/backbone.py. The backbone concepts are attributed to references [4], [5] and [6].')
for title,steps,ref in [
 ('DenseNet121',['Input 3 x 128 x 128','7 x 7 convolution stride 2 and max pooling','Dense block 1 with 6 layers then transition','Dense block 2 with 12 layers then transition','Dense block 3 with 24 layers then transition','Dense block 4 with 16 layers then batch norm and ReLU','Global average pool gives 1024 features','Linear 1024 to 512 then L2 normalisation'], '[4] Huang et al. https://arxiv.org/abs/1608.06993'),
 ('ResNet18',['Input 3 x 128 x 128','7 x 7 convolution stride 2 and max pooling','Residual stage 1 two basic blocks 64 channels','Residual stage 2 two basic blocks 128 channels','Residual stage 3 two basic blocks 256 channels','Residual stage 4 two basic blocks 512 channels','Adaptive average pool gives 512 features','Linear 512 to 512 then L2 normalisation'],'[5] He et al. https://arxiv.org/abs/1512.03385'),
 ('VGG16',['Input 3 x 128 x 128','Two convolutions 64 channels then max pool','Two convolutions 128 channels then max pool','Three convolutions 256 channels then max pool','Three convolutions 512 channels then max pool','Three convolutions 512 channels then max pool','Adaptive pool 7 x 7 then flatten 25088 values','FC 4096 ReLU dropout then FC 4096 ReLU dropout','Linear 4096 to 512 then L2 normalisation'],'[6] Simonyan and Zisserman https://arxiv.org/abs/1409.1556')]:
 doc.add_page_break(); para(title+' architecture','Heading 2')
 # Editable Word table represents flow diagram
 t=doc.add_table(rows=0,cols=1); t.style='Table Grid'
 for i,s in enumerate(steps): t.add_row().cells[0].text=('%02d  '%(i+1))+s+ ('  ↓' if i<len(steps)-1 else '')
 para(title+' architecture adapted for fingerprint embeddings. Source: '+ref+'; implementation: src/models/backbone.py. Diagram labels are editable Word content.','Caption')
 para('The original 1000 class ImageNet output is removed. The project learns a 512 dimensional projection and returns unit vectors. ArcFace is attached only during supervised training in src/models/losses.py and src/training/train.py [7]; search compares the embeddings using cosine similarity.')
para('Appendix D Repository evaluation plots','Heading 1')
para('These are saved repository plots. Each caption identifies its experiment directory. Historical baseline curves are distinct from the sampled checkpoint comparison and from the person retrieval experiment; they must be interpreted within their own saved evaluation protocol.')
for folder in ['densenet121_arcface_e25','resnet18_arcface_e25','vgg16_arcface_e14']:
 for name in ['roc','histogram','cmc']:
  p=root/'outputs/results'/folder/(name+'.png')
  if p.exists():
   doc.add_page_break(); para(folder.replace('_',' ')+' '+name,'Heading 2'); picture(p, f'Saved {name} plot. Source: outputs/results/{folder}/{name}.png; associated numerical evidence: report.json in the same directory.')
para('Appendix E Configuration and evidence provenance','Heading 1')
para('The current src/training/config.yaml selects vgg16 with a 512 dimensional embedding, ArcFace scale 64 and configured margin 28.6, batch size 8, learning rate 0.001, backbone rate multiplier 0.1 and ten frozen epochs within a 25 epoch budget. This editable configuration is not proof of the historical settings of every checkpoint. Use checkpoint metadata and run logs when reproducing a particular result.')
para('Primary evidence paths: outputs/eer_comparison/comparison.csv; outputs/person_evaluation/densenet121_arcface/summary.json; outputs/results/*/report.json. System implementation: src/data/preprocessing.py, src/data/dataset.py, src/models/backbone.py, src/models/losses.py, src/retrieval/core.py and src/retrieval/pinecone_store.py. The supplied thesis is used only for report organisation and front matter, not as a source of fingerprint experimental findings.')
f=sec.footer.paragraphs[0]; f.alignment=1; fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); f._p.append(fld)
doc.save(out/'Fingerprint_Project_Report.docx'); print('SAVED',len(doc.paragraphs))
