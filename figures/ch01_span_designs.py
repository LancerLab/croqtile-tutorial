"""Render three proposals for the original first (span) figure, without changing the book.

Run from the repository root: python3 figures/ch01_span_designs.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import fitz
from manim import Scene, Text, Rectangle, Line, Arrow, Brace, VGroup, LEFT, RIGHT, UP, DOWN, tempconfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'programming-guide/figure-designs/span'
INK = '#24323D'
GRAY = '#63717B'
LIGHT = '#DBE2E6'
BLUE = '#245E8A'
TEAL = '#167467'
OCHRE = '#A26A21'
FONT = 'Noto Sans CJK SC'
MONO = 'DejaVu Sans Mono'


def txt(s, x, y, size=25, color=INK, mono=False, left=False):
    t = Text(s, font=MONO if mono else FONT, font_size=size,
             color=color, disable_ligatures=True)
    t.move_to([x, y, 0])
    if left:
        t.shift(RIGHT * t.width / 2)
    return t


def rule(x1, x2, y, color=LIGHT, width=1):
    return Line([x1, y, 0], [x2, y, 0], stroke_color=color, stroke_width=width)


class SpanProposal(Scene):
    def __init__(self, variant, **kwargs):
        self.variant = variant
        super().__init__(**kwargs)

    def grid(self, x, y, cw=.48, highlight=False):
        cells = VGroup()
        for r in range(4):
            for c in range(8):
                fill = '#EFF4F7'
                if highlight and c == 0: fill = '#E1F0EC'
                if highlight and r == 0: fill = '#F6EDDD'
                cell = Rectangle(width=cw-.035, height=cw-.035,
                                 stroke_color='#BCCBD4', stroke_width=.85,
                                 fill_color=fill, fill_opacity=1)
                cell.move_to([x+(c-3.5)*cw,y+(1.5-r)*cw,0])
                cells.add(cell)
        self.add(cells)
        return cells

    def braces(self, grid, mode='full'):
        row = Brace(grid, LEFT, buff=.13, color=TEAL, stroke_width=1)
        col = Brace(grid, UP, buff=.13, color=OCHRE, stroke_width=1)
        self.add(row,col)
        r = txt('lhs.span(0)\n= 4' if mode=='full' else '4 行',0,0,21,TEAL,mode=='full')
        c = txt('lhs.span(1) = 8' if mode=='full' else '8 列',0,0,23,OCHRE,mode=='full')
        r.next_to(row,LEFT,buff=.18)
        c.next_to(col,UP,buff=.18)
        self.add(r,c)

    def common(self, y=-1.3, compact=False):
        self.add(rule(-6.25,6.25,y), txt('形状如何用于后续代码',-6.25,y-.36,23,GRAY,left=True))
        rows = [
            ('s32 [lhs.span] output;', '完整形状  →  output 为 [4, 8]', BLUE),
            ('s32 [lhs.span(0), rhs.span(1)] result;', '取两个输入的维度  →  [4, 16]*', TEAL),
            ('parallel tile by lhs.span(0)', '第一维大小  →  4 个并行实例', OCHRE),
        ]
        for i,(code,desc,color) in enumerate(rows):
            yy=y-.94-i*.64
            self.add(txt(code,-6.25,yy,20,color,True,True),
                     txt(desc,1.25,yy,21,INK,left=True))
        self.add(txt('* 此处第二个输入 rhs 的形状为 [8, 16]。',-6.25,y-2.68,19,GRAY,left=True))

    def header(self,title,subtitle):
        self.add(txt(title,-6.25,3.63,31,INK,left=True),
                 txt(subtitle,-6.25,3.02,23,GRAY,left=True))

    def construct(self):
        self.camera.background_color = '#FFFFFF'
        getattr(self,self.variant)()

    def a(self):
        # Faithful to the original: dimension braces on the actual matrix,
        # full-shape extraction on the right, reusable code patterns underneath.
        self.header('span：从矩阵读出形状', 'lhs 为 4 × 8 矩阵；span 读取完整形状，span(i) 读取第 i 维。')
        grid=self.grid(-1.55,.72,.46)
        self.braces(grid)
        self.add(txt('lhs',-1.55,-.57,24,GRAY,True))
        self.add(Arrow([.40,.72,0],[1.25,.72,0],buff=.08,color=BLUE,stroke_width=1.7,
                       max_tip_length_to_length_ratio=.13))
        self.add(txt('lhs.span',1.45,1.19,26,BLUE,True,True),
                 txt('[4, 8]',1.45,.54,36,INK,True,True),
                 txt('保留两个维度及其顺序',1.45,-.11,23,GRAY,left=True))
        self.common()

    def b(self):
        # Syntax-first: three concrete access forms above their visual meanings.
        self.header('span 与 span(i)：对应到哪个维度？', '先读表达式，再对照矩阵和返回结果。维度编号从 0 开始。')
        columns=[(-4.35,'lhs.span','完整形状',BLUE),(0,'lhs.span(0)','第 0 维 · 行',TEAL),
                 (4.35,'lhs.span(1)','第 1 维 · 列',OCHRE)]
        for x,code,meaning,c in columns:
            self.add(txt(code,x,2.16,29,c,True),txt(meaning,x,1.61,22,GRAY))
            g=self.grid(x,.30,.30,highlight=False)
            # Emphasize dimensions directly, with a thin bracket/brace.
            if x==0:
                brace=Brace(g,LEFT,buff=.10,color=c)
                self.add(brace,txt('4',0,0,25,c,True).next_to(brace,LEFT,buff=.1))
            elif x>0:
                brace=Brace(g,UP,buff=.08,color=c)
                # The result below gives 8; keep the brace free of a duplicate label.
                self.add(brace)
            self.add(txt('[4, 8]' if x<0 else ('4' if x==0 else '8'),x,-.68,31,c,True))
        self.add(Line([-2.175,1.18,0],[-2.175,-.89,0],stroke_color=LIGHT,stroke_width=1),
                 Line([2.175,1.18,0],[2.175,-.89,0],stroke_color=LIGHT,stroke_width=1))
        self.common()

    def c(self):
        # A compact book figure: a single matrix and a literal shape tuple;
        # connector lines tie axis measurements to the tuple's components.
        self.header('从二维矩阵到形状表达式', '形状记录各维大小；读取形状不会访问或复制矩阵元素。')
        g=self.grid(-3.7,.65,.44,highlight=True)
        self.braces(g,'short')
        self.add(txt('lhs',-3.7,-.51,24,GRAY,True))
        self.add(Arrow([-1.65,.65,0],[-.5,.65,0],buff=.12,color=GRAY,stroke_width=1.6,
                       max_tip_length_to_length_ratio=.12))
        self.add(txt('lhs.span  =',.02,1.6,29,BLUE,True,True))
        bracket=txt('[ 4, 8 ]',2.37,.76,43,INK,True)
        self.add(bracket)
        self.add(txt('第 0 维',.35,-.03,21,TEAL,left=True),
                 txt('第 1 维',3.7,-.03,21,OCHRE,left=True))
        self.add(txt('lhs.span(0) = 4',.35,-.53,20,TEAL,True,True),
                 txt('lhs.span(1) = 8',3.7,-.53,20,OCHRE,True,True))
        # Lines align with the two numbers in the shape tuple.
        self.add(Line([1.85,.38,0],[1.85,.10,0],stroke_color=TEAL,stroke_width=1.5),
                 Line([1.85,.10,0],[1.17,.10,0],stroke_color=TEAL,stroke_width=1.5),
                 Line([2.93,.38,0],[2.93,.10,0],stroke_color=OCHRE,stroke_width=1.5),
                 Line([2.93,.10,0],[4.4,.10,0],stroke_color=OCHRE,stroke_width=1.5))
        self.common()


def package_comparison():
    out = OUT
    labels=[('a','A · 保留原图结构','矩阵 + 维度大括号 + 代码示例'),
            ('b','B · 三种表达式对照','完整形状、行数、列数并排阅读'),
            ('c','C · 形状分量连线','矩阵维度直接对应 [4, 8] 的分量')]
    fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(fontpath,36)
    small=ImageFont.truetype(fontpath,25)
    canvas=Image.new('RGB',(3300,805),'white')
    d=ImageDraw.Draw(canvas)
    for i,(name,title,desc) in enumerate(labels):
     x=i*1100
     d.text((x+35,20),title,font=font,fill='#24323D')
     d.text((x+35,77),desc,font=small,fill='#63717B')
     im=Image.open(out/f'span-{name}.png').convert('RGB')
     im.thumbnail((1100,668))
     canvas.paste(im,(x,127))
     if i: d.line((x,15,x,790),fill='#DBE2E6',width=2)
    canvas.save(out/'comparison.png')
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>第一张 span 图：三个设计方案</title><style>
    * { box-sizing: border-box; } body { margin: 28px; font-family: "Noto Sans CJK SC", sans-serif; color: #24323d; background: white; }
    h1 { font-size: 24px; font-weight: 600; margin-bottom: 8px; } p { color:#63717b; }
    .compare { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:24px; margin-top:30px; }
    section { min-width:0; } h2 { font-size:19px; margin-bottom:8px; } section p { font-size:14px; }
    img { display:block; width:100%; height:auto; } a { color:#245e8a; } .full { display:block; margin-top:12px; }
    details { margin-top:38px; border-top:1px solid #dbe2e6; padding-top:18px; } summary { cursor:pointer; }
    details img { max-width:1000px; margin:20px auto; }
    @media(max-width:850px){ .compare { grid-template-columns:1fr; } }
    </style><h1>第一张 span 图 · 三个设计方案</h1>
    <p>正文已恢复原图；三份方案均保留完整形状、单维大小和三条代码示例。点击图片查看原尺寸。</p>
    <div class="compare">''']
    for name,title,desc in labels:
     parts.append(f'<section><h2>{title}</h2><p>{desc}</p><a href="span-{name}.png"><img src="span-{name}.png" alt="{title}"></a><a class="full" href="span-{name}.png">打开 {name.upper()} 原尺寸图</a></section>')
    parts.append('''</div><p><a href="comparison.pdf">下载三页 PDF</a> · <a href="comparison.png">打开并排比较图</a></p>
    <details><summary>对照：已经恢复到正文的原图</summary><img src="../../figures/f04-fig_span_light.png" alt="原始 span 图"></details></html>''')
    (out/'comparison.html').write_text('\n'.join(parts))
    doc=fitz.open()
    for name,title,desc in labels:
     page=doc.new_page(width=840,height=550)
     page.insert_text((25,26),f'Design {name.upper()}',fontsize=16,color=(.14,.2,.24))
     page.insert_image(fitz.Rect(0,40,840,550),filename=str(out/f'span-{name}.png'))
    doc.set_metadata({'title':'Croqtile first figure: designs A, B and C'})
    doc.save(out/'comparison.pdf')
    (out/'README.md').write_text('''# First figure proposals\n\nThe book retains its original figures and the expanded text. These three\nproposals are separate from the published chapter. The original first figure\nis `../../figures/f04-fig_span_light.png`.\n\n- A: matrix, dimension braces and common code patterns; follows the original.\n- B: three access expressions compared through three matrix diagrams.\n- C: matrix axes connected to the components of the shape tuple.\n\nOpen `comparison.html` for a side-by-side comparison, `comparison.png` for\na single comparison sheet, or `comparison.pdf` for three full-size pages.\n\nRender the proposals from the repository root:\n\n```bash\npython3 figures/ch01_span_designs.py\n```\n''')
    print('Packaged comparison HTML, PNG and three-page PDF')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('a','b','c'):
        with tempconfig({'background_color':'#FFFFFF','frame_width':14,'frame_height':8.5,
                         'pixel_width':2100,'pixel_height':1275,'save_last_frame':True,
                         'write_to_movie':False,'disable_caching':True,
                         'media_dir':str(ROOT/'figures/media/span-proposals'),
                         'output_file':'span-'+name,'verbosity':'ERROR'}):
            scene=SpanProposal(name)
            scene.render()
            src=Path(scene.renderer.file_writer.image_file_path)
            (OUT/f'span-{name}.png').write_bytes(src.read_bytes())
            print('Rendered',name.upper(),flush=True)
    package_comparison()

if __name__=='__main__':main()
