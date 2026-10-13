"""Draw an authoring preview from the deck's shapes and embedded fonts."""
from io import BytesIO
import json
from pathlib import Path
import posixpath
import struct
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parents[2]
PATH = ROOT / "docs/presentations/FreeToken_Weekly_Progress_08-10-2026_with_static_pruning.pptx"
NS = {"a":"http://schemas.openxmlformats.org/drawingml/2006/main", "p":"http://schemas.openxmlformats.org/presentationml/2006/main", "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
SCALE = 120/914400
WIDTH, HEIGHT = 1600, 900
with ZipFile(PATH) as z:
    package = {name:z.read(name) for name in z.namelist()}
font_names = ["Geist", "Geist SemiBold", "Newsreader", "Newsreader Medium"]
font_data = {}
for i, name in enumerate(font_names, 1):
    b=package[f"ppt/fonts/font{i}.fntdata"]
    font_data[name] = b[-struct.unpack_from('<I', b, 4)[0]:]
overflows = []


def get_font(style):
    latin = style.find("a:latin", NS) if style is not None else None
    family = latin.get('typeface') if latin is not None else 'Geist'
    if style is not None and style.get('b')=='1' and family=='Geist':
        family='Geist SemiBold'
    size=round(int(style.get('sz','1500'))/100/72*120) if style is not None else 25
    return ImageFont.truetype(BytesIO(font_data.get(family, font_data['Geist'])), size)


def color(element, default='#2B3139'):
    node=element.find('a:solidFill/a:srgbClr',NS) if element is not None else None
    return '#'+node.get('val') if node is not None else default


def box(shape):
    xf=shape.find('.//a:xfrm',NS)
    if xf is None:
        xf=shape.find('p:xfrm',NS)
    off=xf.find('a:off',NS); ext=xf.find('a:ext',NS)
    return tuple(float(v)*SCALE for v in (off.get('x'),off.get('y'),ext.get('cx'),ext.get('cy')))


def draw_body(draw, body, bbox, number, margins=(0,0,0,0), anchor='t'):
    x,y,w,h=bbox
    bodypr=body.find('a:bodyPr',NS)
    if bodypr is not None:
        margins=tuple(float(bodypr.get(k,str(v/SCALE)))*SCALE for k,v in zip(('lIns','tIns','rIns','bIns'),margins))
        anchor=bodypr.get('anchor',anchor)
    left,top,right,bottom=margins
    lines=[]
    for p in body.findall('a:p',NS):
        value=''.join(node.text or '' for node in p.findall('.//a:t',NS))
        if not value:
            continue
        style=p.find('a:r/a:rPr',NS)
        if style is None:
            style=p.find('a:pPr/a:defRPr',NS)
        font=get_font(style)
        spacing=int(style.get('spc','0'))/100/72*120 if style is not None else 0
        def length(text):
            return font.getlength(text)+max(0,len(text)-1)*spacing
        wrapped=[]; current=''
        for word in value.split():
            candidate=(current+' '+word).strip()
            if current and length(candidate)>w-left-right:
                wrapped.append(current); current=word
            else:
                current=candidate
        if current:
            wrapped.append(current)
        nowrap=bodypr is not None and bodypr.get('wrap')=='none'
        if nowrap:
            wrapped=[value]
        linepct=p.find('a:pPr/a:lnSpc/a:spcPct',NS)
        lineheight=font.size*(int(linepct.get('val'))/100000 if linepct is not None else 1.18)
        paragraph_align=p.find('a:pPr',NS)
        align=paragraph_align.get('algn','l') if paragraph_align is not None else 'l'
        for text in wrapped:
            lines.append((text,font,color(style),spacing,lineheight,align,length(text)))
    total=sum(line[4] for line in lines)
    if total>h-top-bottom+3 or any(line[6]>w-left-right+3 for line in lines):
        overflows.append(dict(slide=number,text=' | '.join(line[0] for line in lines),height=total,available_height=h-top-bottom,width=max((line[6] for line in lines),default=0),available_width=w-left-right))
    yy=y+top+(max(0,(h-top-bottom-total)/2) if anchor=='ctr' else 0)
    for value,font,ink,spacing,lineheight,align,length in lines:
        xx=x+left
        if align=='ctr':
            xx+=(w-left-right-length)/2
        elif align=='r':
            xx+=w-left-right-length
        if abs(spacing)<.03:
            draw.text((xx,yy),value,font=font,fill=ink,anchor='lt')
        else:
            for character in value:
                draw.text((xx,yy),character,font=font,fill=ink,anchor='lt'); xx+=font.getlength(character)+spacing
        yy+=lineheight


def render(number):
    slide=ET.fromstring(package[f'ppt/slides/slide{number}.xml'])
    relationships=ET.fromstring(package[f'ppt/slides/_rels/slide{number}.xml.rels'])
    targets={r.get('Id'):posixpath.normpath('ppt/slides/'+r.get('Target')) for r in relationships}
    canvas=Image.new('RGB',(WIDTH,HEIGHT),'#FAF7F0'); draw=ImageDraw.Draw(canvas)
    for shape in slide.find('p:cSld/p:spTree',NS):
        kind=shape.tag.split('}')[-1]
        if kind=='pic':
            b=shape.find('.//a:blip',NS); x,y,w,h=box(shape)
            image=Image.open(BytesIO(package[targets[b.get('{'+NS['r']+'}embed')]])).convert('RGBA').resize((round(w),round(h)),Image.Resampling.LANCZOS)
            canvas.paste(image,(round(x),round(y)),image)
        elif kind=='sp':
            x,y,w,h=box(shape)
            properties=shape.find('p:spPr',NS)
            fill=properties.find('a:solidFill',NS)
            if fill is not None:
                draw.rectangle((x,y,x+w,y+h),fill=color(properties))
            body=shape.find('p:txBody',NS)
            if body is not None:
                draw_body(draw,body,(x,y,w,h),number)
        elif kind=='graphicFrame':
            x,y,w,h=box(shape)
            table=shape.find('.//a:tbl',NS)
            if table is None:
                continue
            widths=[float(c.get('w'))*SCALE for c in table.find('a:tblGrid',NS)]
            yy=y
            for row in table.findall('a:tr',NS):
                height=float(row.get('h'))*SCALE; xx=x
                for cell,width in zip(row.findall('a:tc',NS),widths):
                    properties=cell.find('a:tcPr',NS)
                    draw.rectangle((xx,yy,xx+width,yy+height),fill=color(properties,'#F7F4ED'),outline='#DDDAD4',width=1)
                    margins=tuple(float(properties.get(k,'91440'))*SCALE for k in ('marL','marT','marR','marB'))
                    draw_body(draw,cell.find('a:txBody',NS),(xx,yy,width,height),number,margins,properties.get('anchor','ctr'))
                    xx+=width
                yy+=height
    target=ASSETS/f'preview_slide{number}.png';canvas.save(target)
    return canvas


pages=[render(number) for number in range(21,31)]
sheet=Image.new('RGB',(1200,5*365),'white')
label_font=ImageFont.truetype(BytesIO(font_data['Geist']),18)
for i,page in enumerate(pages):
    x=(i%2)*600; y=(i//2)*365
    sheet.paste(page.resize((600,338),Image.Resampling.LANCZOS),(x,y+27))
    ImageDraw.Draw(sheet).text((x+12,y+3),f'Slide {i+3}'+(' - reserved' if i>=8 else ''),font=label_font,fill='#2B3139')
sheet.save(ASSETS/'contact_sheet.png')
(ASSETS/'preview_layout_review.json').write_text(json.dumps({'kind':'Pillow authoring preview with original embedded fonts; not an Office rendering','overflows':overflows},indent=2)+'\n')
print(json.dumps(overflows,indent=2))
