"""Shared layouts for the guide's static diagrams.

Use restrained accent fills, readable labels, and explicit schematic examples.
All geometry is kept inside the 16:9 frame; publication rendering crops margins.
"""
from manim import *
from theme import parse_theme

C, THEME = parse_theme()


def label(text, x=0, y=0, size=20, color=None, width=None, mono=False,
          weight=NORMAL, align='center'):
    obj = Text(text, font='DejaVu Sans Mono' if mono else 'DejaVu Sans',
               font_size=size, color=color or C['fg'], line_spacing=0.8,
               weight=weight, disable_ligatures=True)
    if width and obj.width > width:
        obj.scale_to_fit_width(width)
    obj.move_to([x, y, 0])
    if align == 'left':
        obj.shift(RIGHT * obj.width / 2)
    return obj


def heading(scene, title, subtitle):
    scene.camera.background_color = C['bg']
    scene.add(label(title, x=-6.15, y=3.25, size=28, width=12.3,
                    weight=SEMIBOLD, align='left'),
              label(subtitle, x=-6.15, y=2.65, size=18,
                    color=C['fg2'], width=12.3, align='left'),
              Line([-6.15, 2.32, 0], [6.15, 2.32, 0],
                   color=C['separator'], stroke_width=1))


def foot(scene, text):
    scene.add(Line([-6.15, -2.94, 0], [6.15, -2.94, 0],
                   color=C['separator'], stroke_width=1),
              label(text, x=-6.15, y=-3.25, size=17,
                    color=C['fg2'], width=12.3, align='left'))


def panel(scene, title, lines, x, y=0, w=4.9, h=3.5, accent='blue'):
    color = C[accent]
    rect = RoundedRectangle(width=w, height=h, corner_radius=0.10,
                            stroke_color=C['stroke'], stroke_width=1.2,
                            fill_color=C['fill'], fill_opacity=0.6).move_to([x,y,0])
    left = x-w/2+0.3
    scene.add(rect, label(title, left, y+h/2-0.45, 22, color, w-0.6,
                          weight=SEMIBOLD, align='left'),
              Line([left, y+h/2-0.8, 0], [x+w/2-0.3, y+h/2-0.8, 0],
                   color=C['separator'], stroke_width=1))
    step = min(0.65, (h-1.3)/max(1,len(lines)-1))
    for i, line in enumerate(lines):
        scene.add(label(line, left, y+h/2-1.15-i*step, 19, width=w-0.6,
                        align='left'))
    return rect


def connect(scene, a, b, text='', color='blue'):
    arrow = Arrow(a.get_right(),b.get_left(),buff=0.14,stroke_width=2.5,
                  color=C[color],max_tip_length_to_length_ratio=0.15)
    scene.add(arrow)
    if text:
        scene.add(label(text, *arrow.get_center()[:2], size=16, color=C[color]).shift(UP*0.3))
    return arrow


def grid(scene, rows, cols, x, y, cell=0.55, highlight=None, accent='blue', values=False):
    result=[]
    for r in range(rows):
        row=[]
        for c in range(cols):
            active = highlight(r,c) if highlight else False
            color=C[accent] if active else C['stroke']
            box=Square(side_length=cell-0.02,stroke_color=color,stroke_width=1.2,
                       fill_color=C[accent] if active else C['fill'],
                       fill_opacity=0.13 if active else 0.6)
            box.move_to([x+(c-(cols-1)/2)*cell,y+((rows-1)/2-r)*cell,0])
            scene.add(box)
            if values:
                txt=str(r*cols+c) if values is True else values(r,c)
                scene.add(label(txt,*box.get_center()[:2],size=16,width=cell-0.08))
            row.append(box)
        result.append(row)
    return result


def tiled_add(scene):
    heading(scene,'Tiled addition through shared storage','lhs and rhs: [64, 128]  |  parallel {tr, tc} by [4, 8]')
    for x,name,accent in [(-3.3,'lhs','blue'),(3.3,'rhs','orange')]:
        scene.add(label(name+'.chunkat(1, 3)',x,1.95,21,C[accent]))
        grid(scene,4,8,x,0.65,0.55,lambda r,c:(r,c)==(1,3),accent)
        scene.add(label('[16, 16] tile',x,-0.65,19,C[accent]))
        scene.add(Arrow([x,-0.88,0],[x,-1.45,0],buff=0.05,color=C[accent]))
        scene.add(label('dma.copy => shared',x,-1.78,20,C[accent]))
    scene.add(label('lhs_load.data.at(i, j) + rhs_load.data.at(i, j)',y=-2.4,size=21,width=12))
    foot(scene,'output.at(tr # i, tc # j)  |  tile (1, 3): rows 16–31, columns 48–63')


def future_data(scene):
    heading(scene,'Copy result: .data and .span','f = dma.copy input => shared;')
    a=panel(scene,'f: transfer handle',['f.data: copied tensor','f.span: copied tensor shape'], -3.2,h=2.8,accent='purple')
    b=panel(scene,'Shared destination tile',['Read values with f.data.at(idx)','Iterate over the shape f.span'],3.2,h=2.8,accent='blue')
    connect(scene,a,b,'.data','purple')
    scene.add(label('foreach idx in [f.span]',y=-2.0,size=21,mono=True),
              label('output.at(idx) = f.data.at(idx);',y=-2.55,size=21,mono=True))
    foot(scene,'Synchronous copy: use the result directly. Async copy: wait for completion before reading.')


def mma_syntax(scene):
    heading(scene,'MMA: initialize, load, accumulate, store','The load and multiply steps repeat for each K slice.')
    rows=[('1  Initialize','mc = mma.fill 0.0f;','Set the accumulator','purple'),
          ('2  Load','ma = mma.load lhs_tile;','Prepare both operands','blue'),
          ('3  Accumulate','mma.row.row mc, ma, mb;','C += A × B','green'),
          ('4  Store','mma.store mc, output_s;','Write the result tile','orange')]
    for i,(name,code,desc,accent) in enumerate(rows):
        y=1.65-i*1.1
        box=RoundedRectangle(width=12.3,height=0.87,corner_radius=0.1,stroke_color=C['stroke'],
                             fill_color=C['fill'],fill_opacity=0.7).move_to([0,y,0])
        scene.add(box,label(name,-4.65,y,20,C[accent],2.6),
                  label(code,-0.35,y,20,width=5.6,mono=True),label(desc,4.55,y,19,width=3.0))
    foot(scene,'mma.load prepares ma and mb; layout declarations must match the staged operand layout.')


def group_scope(scene):
    heading(scene,'MMA cooperation scope','The space specifier changes the cooperating group; the four operations stay the same.')
    for x,title,lines,accent,nrows in [(-3.2,': group',['1 warp = 32 threads','Warp-scoped cooperation'],'blue',1),
                                      (3.2,': group-4',['4 warps = 128 threads','Warpgroup-scoped cooperation'],'purple',4)]:
        panel(scene,title,lines,x,y=0.1,h=4.0,accent=accent)
        for r in range(nrows):
            for c in range(8):
                scene.add(Dot([x+(c-3.5)*0.35,-0.9-r*0.25,0],radius=0.075,color=C[accent]))
    foot(scene,'fill → load → multiply → store  |  Operand shape, layout, and scope must agree.')


def timeline(scene, roles=False):
    heading(scene,'Concurrent producer and consumer' if roles else 'Sequential and double-buffered execution',
            'Same four K tiles in both schedules; equal-duration bars illustrate the dependency order.')
    x0=-3.6; unit=1.12
    def bar(start,y,txt,accent):
        box=RoundedRectangle(width=unit-0.06,height=0.46,corner_radius=0.05,
                             stroke_color=C[accent],fill_color=C[accent],fill_opacity=0.14)
        box.move_to([x0+(start+0.5)*unit,y,0]);scene.add(box,label(txt,*box.get_center()[:2],size=18))
    for title,y0,pipeline in [('Sequential: load, then compute',1.9,False),('Overlap: load next while computing current',-0.7,True)]:
        scene.add(label(title,y=y0,size=22,color=C['blue'] if pipeline else C['fg']))
        for r,name in enumerate(['Producer' if roles else 'Load','Consumer' if roles else 'Compute']):
            scene.add(label(name,-5.05,y0-0.65-r*0.7,18,width=2.0))
        for k in range(4):
            bar(k if pipeline else 2*k,y0-0.65,f'L{k}','blue')
            bar(k+1 if pipeline else 2*k+1,y0-1.35,f'C{k}','orange')
        scene.add(Arrow([x0,y0-1.82,0],[5.55,y0-1.82,0],buff=0,color=C['dim'],stroke_width=1.4),
                  label('time',6.0,y0-1.82,16))
    foot(scene,'Schematic timing. Events protect readiness and reuse; overlap does not imply a fixed speedup.')


def persistent(scene):
    heading(scene,'Persistent scheduling: fixed workers, repeated tiles','NUM_GROUPS = 4  |  total_tiles = 10  |  tile_id = tile_iter # block_id')
    scene.add(label('block_id',-5.3,1.6,19))
    for b in range(4):scene.add(label(str(b),-3.0+b*2.0,1.6,21,C['blue']))
    for iteration in range(3):
        y=0.8-iteration*0.9
        scene.add(label('iter '+str(iteration),-5.3,y,19))
        for b in range(4):
            tid=iteration*4+b;valid=tid<10;accent='blue' if valid else 'red'
            box=RoundedRectangle(width=1.72,height=0.65,corner_radius=0.08,
                                 stroke_color=C[accent],fill_color=C[accent],fill_opacity=0.1).move_to([-3+b*2,y,0])
            scene.add(box,label(str(tid) if valid else str(tid)+'  skip',-3+b*2,y,20,C[accent]))
    scene.add(label('if (tile_id < total_tiles) { … compute tile … }',y=-2.2,size=22,mono=True,width=12))
    foot(scene,'Each block processes its column. The guard skips padding iterations 10 and 11.')


def descriptor(scene):
    heading(scene,'Descriptor-based tensor movement','The descriptor describes the layout; the source expression selects a tile origin.')
    a=panel(scene,'Tensor descriptor',['Base address','Dimensions and strides','Tile shape and swizzle'], -4.3,w=3.6,h=3.3)
    b=panel(scene,'TMA transfer',['Source tile coordinates','Hardware address generation','Read global data'],0,w=3.6,h=3.3,accent='purple')
    c=panel(scene,'Shared tile',['[WARP_M, TILE_K]','Selected source slice','Ready after completion'],4.3,w=3.6,h=3.3,accent='orange')
    connect(scene,a,b);connect(scene,b,c)
    foot(scene,'tma.copy lhs.subspan(WARP_M, TILE_K).at(block_m, iv_k) => lhs_load_s;')


def expressiveness(scene):
    heading(scene,'Structured data movement','Croqtile makes tile geometry and layout explicit so the compiler can lower structured transfers.')
    a=panel(scene,'Per-element address code',['Flexible indexing','Per-lane address calculations','Coalescing and conflicts need care'], -3.2,h=3.8,accent='orange')
    b=panel(scene,'Croqtile tile operations',['dma.copy / tma.copy','Explicit tile shape and destination','Matching swizzle on copy and load'],3.2,h=3.8)
    connect(scene,a,b,'structure')
    foot(scene,'Performance still depends on tile sizes, alignment, synchronization, and the target hardware.')


def tma_vs_dma(scene):
    heading(scene,'DMA and TMA: two ways to move a tile','Both paths select a source tile and write a destination tile.')
    panel(scene,'dma.copy',['Threads cooperate on the transfer','Each lane computes its addresses','Collectively cover the tile'], -3.2,h=3.8,accent='orange')
    panel(scene,'tma.copy',['Descriptor-based tensor transfer','Hardware generates tile addresses','Dedicated transfer engine'],3.2,h=3.8)
    foot(scene,'Both need correct layout and synchronization. TMA reduces per-thread transfer choreography.')


def element_vs_block(scene):
    heading(scene,'Scalar indexing and explicit tile movement','Two views of the data path: read individual elements, or stage a tile before computing.')
    for x,title,accent in [(-3.3,'Scalar indexing','orange'),(3.3,'Tile movement','blue')]:
        scene.add(label(title,x,1.9,24,C[accent]),label('Global input',x,1.25,20))
        cells=grid(scene,1,8,x,0.65,0.55,lambda r,c:True,accent,True)[0]
        grid(scene,1,8,x,-1.25,0.55,lambda r,c:True,accent)
        if x<0:
            for cell in cells:
                cx=cell.get_center()[0]
                scene.add(Arrow([cx,0.3,0],[cx,-0.9,0],buff=0.05,color=C[accent],stroke_width=1.8))
            scene.add(label('Element reads',x,-1.9,21,C[accent]))
        else:
            scene.add(Arrow([x,0.3,0],[x,-0.9,0],buff=0.05,color=C[accent],stroke_width=3),
                      label('dma.copy',x+1.25,-0.3,20,C[accent]),label('Shared destination tile',x,-1.9,21,C[accent]))
    foot(scene,'Explicit movement separates selecting the source tile, staging its data, and computing on it.')


def chunkat(scene):
    heading(scene,'chunkat: select one equal-sized tile','lhs [64, 128]  |  parallel {tr, tc} by [4, 8]')
    scene.add(label('4 × 8 grid of tiles',-3.2,1.95,22))
    tiles=grid(scene,4,8,-3.2,0.4,0.58,lambda r,c:(r,c)==(1,3),values=lambda r,c:f'{r},{c}')
    b=panel(scene,'lhs.chunkat(1, 3)',['Tile shape: [16, 16]','Rows: 16–31','Columns: 48–63'],3.3,y=0.4,h=3.25)
    scene.add(label('64 / 4 = 16 rows per tile',-3.2,-1.45,21),label('128 / 8 = 16 columns per tile',-3.2,-2.0,21))
    foot(scene,'The parallel extents determine the equal partition; (tr, tc) chooses the tile.')


def compose(scene):
    heading(scene,'Compose coordinates: outer # inner','output [128]  |  parallel tile by 8  |  16 elements per tile')
    scene.add(label('Global tile grid',y=1.8,size=22))
    tiles=grid(scene,1,8,0,1.1,0.85,lambda r,c:c==2,'blue',lambda r,c:str(c))
    scene.add(label('tile = 2 → global positions 32–47',y=0.3,size=23,color=C['blue']))
    b=panel(scene,'Local offset i = 3',['tile # i = 2 × 16 + 3 = 35','output.at(tile # i) = output.at(35)'],0,y=-1.4,w=10.8,h=2.0)
    foot(scene,'Compose maps an in-tile coordinate to the corresponding coordinate in the full tensor.')


def extent(scene):
    heading(scene,'Extent: prefix #name','parallel tile by 8  |  vector length = 128')
    grid(scene,1,8,0,1.25,0.9,lambda r,c:True,'blue',lambda r,c:str(c))
    panel(scene,'Prefix: #tile',['Extent = 8','Number of tiles along this axis','128 / #tile = 16 elements per tile'], -3.2,y=-0.85,h=3.0)
    panel(scene,'Infix: tile # i',['Composed global coordinate','Tile index plus in-tile offset','For tile = 2, i = 3: position 35'],3.2,y=-0.85,h=3.0,accent='purple')
    foot(scene,'Prefix # asks how many positions an axis has; infix # combines coordinates.')


def span(scene):
    heading(scene,'span: full shape or one dimension','lhs [4, 8]  |  dimensions are numbered from zero')
    grid(scene,4,8,-3.0,0.2,0.6)
    scene.add(label('4 rows',-3,1.65,21,C['blue']),label('8 columns',-3,-1.35,21,C['orange']))
    panel(scene,'Shape queries',['lhs.span       → [4, 8]','lhs.span(0)    → 4','lhs.span(1)    → 8'],3.2,y=0.2,h=3.1)
    foot(scene,'s32 [lhs.span(0), rhs.span(1)] output;  |  Derive output dimensions from the operands.')


def virtual_parallelism(scene):
    heading(scene,'Parallel tasks and execution capacity','The same eight independent tasks can run under different schedules.')
    for count,y in [(1,1.25),(4,-0.05),(8,-1.7)]:
        scene.add(label(f'{count} at a time',-4.8,y,22,width=2.7))
        steps=8//count
        for k in range(8):
            col=k//count;row=k%count
            # For the wide schedule, two display rows fit four simultaneous tasks each.
            if count==8:col=k%4;row=k//4
            x=-2.6+col*0.73 if count!=8 else -1.2+col*0.82
            yy=y+(count-1)/2*0.38-row*0.38 if count!=8 else y+0.3-row*0.6
            box=RoundedRectangle(width=0.7,height=0.33 if count==4 else 0.47,corner_radius=0.04,
                                 stroke_color=C['blue'],fill_color=C['blue'],fill_opacity=0.1).move_to([x,yy,0])
            scene.add(box,label(f'T{k}',x,yy,16))
        scene.add(label(f'{steps} time step'+('s' if steps>1 else ''),4.9,y,20,C['purple'],2.7))
    foot(scene,'parallel declares independence; it does not promise that every task runs simultaneously.')


def logical_physical(scene):
    heading(scene,'Logical structure and physical mapping','parallel declares independent work; foreach preserves iteration order.')
    rows=[('parallel {px, py} by [8, 16]',': block','8 × 16 blocks','blue'),
          ('foreach {tile_k} in [16]','sequential','K-tile loop in each block','purple'),
          ('parallel {qx, qy} by [16, 16]',': thread','256 threads per block','blue'),
          ('foreach k in [16]','sequential','Accumulation in each thread','purple')]
    for i,(code,mapping,target,accent) in enumerate(rows):
        y=1.55-i*1.05
        scene.add(label(code,-3.55,y,21,width=6.3,mono=True),
                  label(mapping,0.55,y,19,C[accent],2.0),label(target,4.35,y,20,width=3.6))
        if i<3:scene.add(Arrow([-3.55,y-0.3,0],[-3.55,y-0.74,0],buff=0.03,color=C['dim'],stroke_width=1.5))
    foot(scene,'Space specifiers connect the logical axes to execution resources; sequential loops stay ordered.')


def scalar_matmul(scene):
    heading(scene,'Scalar matmul: output ownership','output [128, 256]  |  parallel {p, q} by [16, 64]')
    scene.add(label('First 4 × 8 tiles of a 16 × 64 grid',-3.2,1.95,21))
    tiles=grid(scene,4,8,-3.2,0.3,0.6,lambda r,c:(r,c)==(1,3),'blue',lambda r,c:f'{r},{c}')
    b=panel(scene,'Tile (p = 1, q = 3)',['Shape: [8, 4]','Rows: 8–15','Columns: 12–15','Each output sums over K = 256'],3.3,y=0.3,h=3.8)
    scene.add(label('… 16 row-tiles, 64 column-tiles',-3.2,-1.35,20,C['fg2']))
    foot(scene,'Global coordinates: p # m = p × 8 + m; q # n = q × 4 + n. This version uses scalar reads.')


def dma_matmul(scene):
    heading(scene,'DMA matmul: stage operands, then compute','8 × 16 blocks cover output [128, 256]; each block owns a [16, 16] output tile.')
    for x,title,code,accent in [(-3.2,'Left operand','lhs.chunkat(px, tile_k)','blue'),(3.2,'Right operand','rhs.chunkat(tile_k, py)','orange')]:
        panel(scene,title,[code,'dma.copy => shared','Copied tile: [16, 16]'],x,y=0.65,h=2.7,accent=accent)
        scene.add(Arrow([x,-0.8,0],[x,-1.45,0],buff=0.08,color=C[accent]))
    scene.add(label('16 × 16 threads: each computes one output element',y=-1.85,size=23))
    scene.add(label('Read lhs_load.data and rhs_load.data; accumulate over k.',y=-2.45,size=21))
    foot(scene,'foreach tile_k in [16] visits all K slices; px # qx and py # qy give global output coordinates.')


def register_layout(scene):
    heading(scene,'MMA operands are distributed fragments','Schematic lane-group ownership; the exact mapping depends on the instruction and data type.')
    colors=['blue','orange','green','purple']
    for r in range(6):
        for c in range(8):
            color=C[colors[(r//2+c//2)%4]]
            box=Square(side_length=0.48,stroke_color=color,fill_color=color,fill_opacity=0.15)
            box.move_to([-3.2+(c-3.5)*0.5,0.25+(2.5-r)*0.5,0]);scene.add(box)
    panel(scene,'The layout contract',['mma.load prepares operand fragments','mma.store writes the result tile','Row/column declarations must match'],3.2,y=0.25,h=3.6)
    foot(scene,'Colors illustrate fragmented ownership, not an exact lane-to-element map for a specific MMA instruction.')


def event_credit(scene):
    heading(scene,'One pipeline stage: the full and empty events','Bootstrap: the consumer triggers every empty[stage] before the producer starts.')
    a=panel(scene,'Producer',['wait empty[stage]','Copy into shared stage','trigger full after copy completion'], -3.2,y=0.3,h=3.25)
    b=panel(scene,'Consumer',['wait full[stage]','Compute; wait for completion','trigger empty[stage]'],3.2,y=0.3,h=3.25,accent='orange')
    connect(scene,a,b,'full','purple')
    scene.add(Arrow([3.2,-1.65,0],[-3.2,-1.65,0],buff=0,color=C['green'],stroke_width=2.5),
              label('empty: this slot can be reused',y=-2.15,size=21,color=C['green']))
    foot(scene,'stage = iv_k % STAGES  |  The consumer releases a stage only after all reads have finished.')


def swizzle(scene):
    heading(scene,'Swizzle spreads accesses across banks','Eight-bank schematic; arrows refer to different addresses, so a shared bank requires serialization.')
    for y,title,targets,accent in [(1.4,'Without swizzle',[0,0,0,0],'orange'),(-1.3,'With matching XOR swizzle',[0,3,5,6],'blue')]:
        scene.add(label(title,-4.45,y+0.2,21,width=3.1,color=C[accent]))
        banks=grid(scene,1,8,1.2,y-0.6,0.65,values=lambda r,c:f'B{c}')[0]
        for i,t in enumerate(targets):
            x=-1+i*1.1
            scene.add(label(f'Lane {i}',x,y+0.45,18,C[accent]))
            scene.add(Arrow([x,y+0.22,0],banks[t].get_top(),buff=0.04,color=C[accent],stroke_width=2))
    foot(scene,'Copy and MMA load must use matching swiz<N>; these bank indices illustrate the effect, not the exact mapping.')


def view_from(scene):
    heading(scene,'Regular tiles and a window at an arbitrary offset','An 8 × 8 schematic makes the different origins visible.')
    for x,title,hl,accent,code in [(-3.2,'Aligned 4 × 4 tile',lambda r,c:r<4 and c<4,'blue','lhs.chunkat(0, 0)'),
                                 (3.2,'4 × 4 window from (2, 3)',lambda r,c:2<=r<6 and 3<=c<7,'orange','lhs.view(4, 4).from(2, 3)')]:
        scene.add(label(title,x,1.95,22,C[accent]))
        grid(scene,8,8,x,0,0.43,hl,accent)
        scene.add(label(code,x,-2.15,20,C[accent],5.8,mono=True))
    foot(scene,'The same rule selects matrix.view(16, 16).from(37, 50): rows 37–52, columns 50–65.')


def subspan_step(scene):
    heading(scene,'Tile extent and tile step are separate','Small schematic: a tile has 3 × 3 elements; its step controls the spacing between origins.')
    for x,step,title,accent in [(-3.2,3,'Packed: step = tile extent','blue'),(3.2,5,'Spaced: step > tile extent','orange')]:
        scene.add(label(title,x,1.95,22,C[accent]))
        grid(scene,8,8,x,0,0.43,lambda r,c,s=step:(r % s<3 and c % s<3),accent)
        if step == 3:
            for offset in (3, 6):
                pos = (offset - 4) * 0.43
                scene.add(Line([x+pos,-1.72,0],[x+pos,1.72,0],color=C[accent],stroke_width=3),
                          Line([x-1.72,-pos,0],[x+1.72,-pos,0],color=C[accent],stroke_width=3))
        scene.add(label(f'.subspan(3, 3).step({step}, {step})',x,-2.15,19,C[accent],5.9,mono=True))
    foot(scene,'In the [16, 16] / step(32, 32) example, tile (1, 0) begins at row 32, leaving a 16-row gap.')


def zfill(scene):
    heading(scene,'zfill: zero-fill a boundary tile','Source [6, 8]  |  .subspan(4, 4).at(1, 1) selects rows 4–7, columns 4–7.')
    scene.add(label('Source tensor',-3.25,1.95,22,C['blue']))
    grid(scene,8,8,-3.25,0.05,0.44,lambda r,c:r<6,'blue',lambda r,c:'' if r<6 else '—')
    scene.add(label('Rows 6–7 are outside the source',-3.25,-2.05,19,C['fg2'],5.8))
    scene.add(label('Copied [4, 4] tile',3.2,1.95,22,C['orange']))
    grid(scene,4,4,3.2,0.25,0.68,lambda r,c:r>=2,'orange',lambda r,c:'data' if r<2 else '0')
    scene.add(Arrow([-1.1,0.15,0],[1.65,0.15,0],buff=0.15,color=C['blue']),label('.zfill',0.2,0.65,21,C['blue']))
    foot(scene,'The destination tile keeps its full shape. Valid source elements are copied; out-of-range elements become zero.')


def span_as(scene):
    heading(scene,'span_as: reinterpret the shape without moving data','The element count and linear element order stay the same.')
    scene.add(label('Flat buffer [16]',-3.25,1.65,23,C['blue']))
    # Two display rows represent one continuous 1D strip, preserving order explicitly.
    grid(scene,2,8,-3.25,0.5,0.57,values=True)
    scene.add(label('Indices 0–15, shown on two lines',-3.25,-0.75,19,C['fg2'],5.8))
    scene.add(label('2D view [4, 4]',3.2,1.95,23,C['orange']))
    grid(scene,4,4,3.2,0.15,0.7,values=True)
    scene.add(Arrow([-0.65,0.3,0],[1.5,0.3,0],buff=0.12,color=C['purple']),label('.span_as([4, 4])',0.1,0.9,18,C['purple'],2.9))
    foot(scene,'4 × 4 = 16  |  This changes logical rank and shape; it does not allocate or copy a new buffer.')
