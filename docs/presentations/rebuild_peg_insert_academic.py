from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_THEME_COLOR
from pathlib import Path

# 版式参考：用户提供的《智元开放基金_清扬负责部分_机制图文版.pptx》；不直接复制其内容或图片。
OUT='docs/presentations/Isaac_Factory_PegInsert_academic_style_3slides.pptx'
prs=Presentation()
prs.slide_width=Inches(13.3333333333)
prs.slide_height=Inches(7.5)
# Visual system (font, colors, top/bottom rules, card headers and information density)
# is reconstructed from the user-supplied academic reference deck.
W,H=13.3333333333,7.5
BLUE='0B4FA1'; NAVY='082B72'; NAVY2='0A3C91'; TEXT='171717'; GRAY='586574'; GREEN='5A993D'; LIGHT_BLUE='EDF4FC'; PALE='F5F8FC'; BORDER='9CB7D9'; PALE_GREEN='EFF7EA'; RED='A43A2E'; WHITE='FFFFFF'; ORANGE='BD7C23'; GREEN_DARK='375D4F'

def rgb(s):return RGBColor.from_string(s)
def rect(s,x,y,w,h,fill=None,line=None,lw=0.7,rad=False):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rad else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
    else:sh.fill.background()
    if line:sh.line.color.rgb=rgb(line);sh.line.width=Pt(lw)
    else:sh.line.fill.background()
    if rad:
        try:sh.adjustments[0]=0.08
        except:pass
    return sh

def line(s,x1,y1,x2,y2,color=BLUE,width=1.0):
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    sh.line.color.rgb=rgb(color);sh.line.width=Pt(width)
    return sh

def text(s,x,y,w,h,v,sz=15,color=TEXT,bold=False,align='left',val='mid',margin=0.02,wrap=True):
    sh=s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf=sh.text_frame;tf.clear();tf.word_wrap=wrap
    tf.margin_left=tf.margin_right=Inches(margin)
    tf.margin_top=tf.margin_bottom=Inches(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE if val=='mid' else MSO_ANCHOR.TOP
    p=tf.paragraphs[0]
    p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[align]
    # explicit line breaks are supported in run
    r=p.add_run();r.text=v
    f=r.font;f.name='微软雅黑';f.size=Pt(sz);f.bold=bold;f.color.rgb=rgb(color)
    return sh

def block_title(s,x,y,w,title):
    text(s,x,y,w,0.42,title,18.5,BLUE,True)
    line(s,x,y+0.45,x+w,y+0.45,'C2D5EA',0.9)

def base(title,sub,num,foot):
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    # Blank slide background is already white; avoid a full-bleed shape theme shadow.
    text(slide,0.0,0.19,13.25,.61,title,29.5,BLUE,True,margin=.07)
    rect(slide,0,.812,W,.057,BLUE)
    text(slide,.42,1.025,12.5,.42,sub,16.7,TEXT,False)
    rect(slide,0,6.985,W,.075,GREEN_DARK)
    text(slide,.42,7.095,11.93,.25,foot,10.8,BLUE,False,margin=0)
    text(slide,12.72,7.11,.23,.19,str(num),10.3,BLUE,False,align='right')
    return slide

def panel(s,x,w,title,color=NAVY,y=1.78,height=3.28):
    rect(s,x,y,w,height,WHITE,BORDER,.95,True)
    rect(s,x,y,w,.54,color)
    text(s,x+.04,y+.04,w-.08,.44,title,17.3,WHITE,True,align='center')

def arrow(s,x,y):
    # modest reference-like green arrows between process panels
    line(s,x,y,x+.18,y,GREEN,6)
    sh=s.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x+.12), Inches(y-.11), Inches(.23), Inches(.22))
    sh.fill.solid();sh.fill.fore_color.rgb=rgb(GREEN);sh.line.fill.background()

# ---------------- 1 ----------------
s=base('1.1 Isaac Factory PegInsert：项目目标与控制架构',
       '上层 PPO 生成末端位姿增量，底层任务空间 PD 和动态一致零空间控制生成关节力矩',1,
       '注：控制结构为已审计的 Factory PegInsert 基线；孔、插针绘图为机制示意，非真实实验截图。')
xs=[.42,3.65,6.88,10.11]; w=2.81
for x,t in zip(xs,['① 观测与训练','② PPO策略（15 Hz）','③ 任务空间控制','④ Panda执行（PhysX）']):panel(s,x,w,t)
for x in [3.35,6.58,9.81]:arrow(s,x,3.22)
# panel 1
rect(s,.72,2.57,2.2,.50,LIGHT_BLUE,BORDER,.7,True);text(s,.78,2.60,2.08,.42,'Actor：19维观测',16.5,BLUE,True,align='center')
text(s,.65,3.17,2.37,.76,'相对孔位姿、末端速度\n及上一时刻动作缓存',14.1,TEXT,False,align='center')
rect(s,.75,4.08,2.14,.42,PALE_GREEN,'A5C8A3',.7,True)
text(s,.77,4.09,2.10,.39,'Critic：43维特权状态',14.1,'326B37',True,align='center')
text(s,.72,4.57,2.19,.28,'非对称 Actor-Critic',12.6,GRAY,False,align='center')
# panel 2
text(s,3.89,2.54,2.32,.36,'LSTM：2 × 1024',15.6,BLUE,True,align='center')
line(s,5.05,2.95,5.05,3.18,GREEN,2)
text(s,3.91,3.18,2.27,.37,'MLP：512 → 128 → 64',13.7,TEXT,False,align='center')
line(s,5.05,3.58,5.05,3.79,GREEN,2)
text(s,3.80,3.84,2.53,.35,'6D末端增量动作',15.0,BLUE,True,align='center')
for i,t in enumerate(['Δx','Δy','Δz','Δrx','Δry','Δrz']):
    x=3.87+(i%3)*.81;y=4.23+(i//3)*.33
    rect(s,x,y,.72,.27,'E5F0FB','B9D3EE',.6,True)
    text(s,x+.02,y,.69,.25,t,11.2,NAVY,False,align='center')
# panel 3
text(s,7.13,2.52,2.26,.36,'目标位姿 / 速度误差',14.2,GRAY,False,align='center')
rect(s,7.03,2.99,2.52,.53,PALE,BORDER,.5)
text(s,7.09,3.04,2.4,.41,'W = Kp Δx + Kd Δv',15.0,BLUE,True,align='center')
text(s,7.08,3.65,2.42,.36,'τtask = Jᵀ W',17,BLUE,True,align='center')
line(s,8.27,4.00,8.27,4.17,GREEN,1.9)
rect(s,7.06,4.19,2.48,.49,LIGHT_BLUE,BORDER,.5)
text(s,7.09,4.20,2.42,.46,'τ = τtask + τnull',15.1,NAVY,True,align='center')
text(s,7.05,4.77,2.47,.22,'姿态调节经零空间投影',11.8,GRAY,False,align='center')
# panel 4: schematic peg-in-hole, not a real screenshot
text(s,10.36,2.48,2.30,.30,'7个关节力矩目标',14.7,BLUE,True,align='center')
# peg and jaws, U shaped cavity at y3.4..4.42
rect(s,11.33,2.91,.55,.92,'A3B8CA','637D91',.65)
rect(s,11.22,3.02,.13,.22,'899CAD','637D91',.5)
rect(s,11.88,3.02,.13,.22,'899CAD','637D91',.5)
rect(s,10.84,3.75,.38,.65,'D7E3EF','738FA8',.7)
rect(s,12.01,3.75,.38,.65,'D7E3EF','738FA8',.7)
rect(s,11.22,4.13,.79,.28,'D7E3EF','738FA8',.7)
line(s,11.61,3.82,11.61,4.10,ORANGE,2)
text(s,10.43,4.48,2.19,.26,'Peg → Hole（剖面示意）',12.4,GRAY,False,align='center')
text(s,10.50,4.79,2.05,.21,'物理120 Hz；策略15 Hz',11.0,BLUE,True,align='center')
# flow and 2 summary band
block_title(s,.46,5.25,12.42,'状态—决策—控制—执行：闭环反馈')
text(s,.61,5.85,11.98,.37,'状态更新    →    PPO输出动作    →    EMA与目标构建    →    PD / Jacobian / 零空间    →    物理执行与反馈',17.0,BLUE,True,align='center')
rect(s,.53,6.36,12.30,.38,'EAF2FC')
text(s,.63,6.39,12.04,.30,'64个并行环境  ×  128步Rollout  ×  200 Epoch  =  1,638,400 Transitions（训练Seed 0）',14.1,NAVY,True,align='center')
s.notes_slide.notes_text_frame.text=(
'这页说明 Isaac Factory PegInsert 项目的真实控制结构。项目采用 Panda 7DoF 机械臂和 Isaac Lab/PhysX 环境，使用 PPO 来学习末端的连续相对动作，而不是让 PPO 直接产生关节力矩。'
'Actor使用19维观测、Critic使用43维特权状态，是非对称Actor-Critic。Actor包含2层、每层1024单元的LSTM及后续MLP，生成6维相对位姿动作；动作经过clip、EMA及缩放后得到目标位姿。'
'底层按任务空间PD计算命令Wrench，再使用J转置及动态一致零空间姿态控制生成7维关节力矩。仿真物理频率120Hz，策略15Hz，目标和PD在每个物理子步中重新计算。'
'在主要基线中，使用64个并行环境、128步rollout、200个epoch，完成1,638,400条transition。这里画的插针剖面是机制示意图，不是实际渲染截图。'
'讲解时强调：这是RL与传统控制结合的架构，而非纯RL力矩控制；Actor观测来自仿真机制，并非所有量已在真实机器人上验证可用。')

# ---------------- 2 ----------------
s=base('1.2 PPO 训练与独立几何评估',
       '训练 Seed 0 的冻结策略，在3个独立评估 Seed（1001/1002/1003）上完成288个 Episode',2,
       '注：Success 为原 Factory 几何判据，非接触力安全认证；50–150与200轮的评估分母不同。')
block_title(s,.45,1.62,5.99,'不同Checkpoint的独立评估')
block_title(s,6.84,1.62,6.03,'Epoch 200：三个评估种子')
# axes and bars left
plotx,ploty,plotw,ploth=.95,2.39,5.19,2.65
line(s,plotx,ploty+ploth,plotx+plotw,ploty+ploth,'61768E',.9)
line(s,plotx,ploty,plotx,ploty+ploth,'61768E',.9)
for vv in [0,50,75,100]:
    yy=ploty+ploth-vv/100*2.30
    text(s,.47,yy-.10,.43,.23,f'{vv}',10.5,GRAY,False,align='right')
    line(s,plotx,yy,plotx+plotw,yy,'DFE7F0',.6)
items=[('50',69/96,'69/96'),('100',81/96,'81/96'),('150',92/96,'92/96'),('200',280/288,'280/288')]
for i,(ep,v,ratio) in enumerate(items):
    x=1.31+i*1.21; height=2.30*v; y=ploty+ploth-height
    rect(s,x,y,.68,height,BLUE if i<3 else GREEN)
    text(s,x-.16,y-.34,1.00,.26,f'{v*100:.1f}%',12.8,BLUE if i<3 else GREEN,True,align='center')
    text(s,x-.13,5.10,.94,.24,ep,12.4,TEXT,False,align='center')
    text(s,x-.20,5.37,1.05,.23,ratio,10.9,GRAY,False,align='center')
text(s,.91,5.67,5.05,.28,'横轴：Epoch；柱上：对应独立评估几何成功率',12.5,GRAY)
# right seed bars
for i,(label,a,b) in enumerate([('Seed 1001',95,96),('Seed 1002',96,96),('Seed 1003',89,96)]):
    yy=2.36+i*.78
    text(s,7.00,yy,1.70,.28,label,14.0,TEXT,True)
    rect(s,8.75,yy,.0+3.00,.27,'EEF2F6')
    rect(s,8.75,yy,3.00*a/b,.27,BLUE if i<2 else '5485B5')
    text(s,11.86,yy-.02,.95,.31,f'{a}/{b}',14.4,BLUE,True,align='right')
text(s,7.03,4.90,5.53,.35,'汇总：280 / 288 = 97.22%',20.0,BLUE,True)
text(s,7.00,5.30,5.77,.52,'Ever-success 与 Terminal-success 均为 280/288；\n每个 Episode 运行到149个策略步超时结束。',12.9,GRAY)
# bottom table style
block_title(s,.46,5.90,12.42,'成功判据与解释范围')
for x,wid,h1,h2 in [( .52,3.70,'几何成功判据','XY < 2.5 mm；有符号 ΔZ < 1 mm'),(4.32,3.71,'实验统计范围','训练Seed=0；评估Seed=3个'),(8.13,4.63,'不能据此推断','安全接触力、有效装配保持或Sim2Real成功')]:
    rect(s,x,6.35,wid,.35,'DDE9F6',BORDER,.4)
    text(s,x+.05,6.37,wid-.1,.28,h1,13.7,BLUE,True,align='center')
    # body slightly compressed to footer
    rect(s,x,6.70,wid,.25,'F6FAFD',BORDER,.35)
    text(s,x+.03,6.71,wid-.06,.22,h2,10.6,GRAY,False,align='center')
s.notes_slide.notes_text_frame.text=(
'本页展示已经完成的PPO训练和独立评估，不是训练日志中的即时成功率。主要基线使用Seed 0、64个环境训练200个Epoch，总计1,638,400条transition。'
'冻结Epoch200的最终checkpoint，在Seed1001/1002/1003中分别评估96回合，对应95/96、96/96、89/96，总计280/288，几何成功率97.22%。'
'在Epoch50/100/150的独立评估分别为69/96、81/96、92/96，Epoch200改为三个评估Seed各96回合，即288回合，不能直接当作等样本显著性检验。'
'当前成功谓词要求Peg资产基准的XY距离小于2.5mm、有符号Z差小于1mm；不包含力、力矩、穿透、接触稳定时长等判据，因此必须称作几何成功率。'
'训练随机种子只有1个，不能将3个评估Seed误说成3个独立训练模型。训练和部署状态均未验证全量真实观测可测。')

# ---------------- 3 ----------------
s=base('1.3 Sim2Sim 迁移：失稳诊断与控制变量实验',
       '冻结PPO权重，由 Isaac Sim / PhysX 迁移到 MuJoCo，区分物理惯量与控制器质量矩阵的影响',3,
       '来源：G4-S2A-R1/R2与G4 Sim2Sim Closeout；局部实验改进≠完整任务零样本迁移成功。')
block_title(s,.45,1.62,5.99,'首次闭环异常：按物理子步定位')
block_title(s,6.84,1.62,6.03,'Plant / Controller 2×2 实验')
# left sequence
rect(s,.57,2.20,5.68,.45,LIGHT_BLUE,BORDER,.5,True)
text(s,.67,2.22,5.47,.4,'PhysX训练的Policy  →  MuJoCo独立运行',16.3,BLUE,True,align='center')
steps=[('Step 2','J7速度反转','+0.505 → −5.150','rad/s'),('Step 3','出现关节力矩饱和','夹爪滑移扩大',''),('Step 4','右侧正接触力丢失','夹持不再稳定','')]
for i,(a,b,c,d) in enumerate(steps):
    x=.54+i*1.95
    rect(s,x,2.91,1.82,1.18,'F6FAFD',BORDER,.8,True)
    rect(s,x,2.91,1.82,.33,NAVY)
    text(s,x+.04,2.92,1.74,.27,a,13.2,WHITE,True,align='center')
    text(s,x+.10,3.32,1.62,.31,b,12.9,BLUE,True,align='center')
    text(s,x+.07,3.65,1.68,.33,c,11.3,TEXT,False,align='center')
    if d:text(s,x+.05,3.95,1.72,.13,d,10.2,GRAY,False,align='center')
    if i<2:arrow(s,x+1.82,3.52)
rect(s,.59,4.36,5.57,.87,'F3F8FC','A7C3DD',.8)
text(s,.73,4.41,5.31,.28,'Factory中各刚体转动惯量另加 0.01 I₃ kg·m²',15.0,BLUE,True)
text(s,.74,4.74,5.22,.34,'MuJoCo适配11个映射body；局部改善不代表完全等价。',11.8,GRAY)
# right 2x2 table with tiny bar in 5th col
cols=[(6.97,.51,'组'),(7.49,1.43,'物理Plant'),(8.94,1.35,'控制器 M'),(10.31,.92,'max |q̇|'),(11.26,1.45,'速度对照')]
for x,wid,txt in cols:
    rect(s,x,2.21,wid,.39,NAVY)
    text(s,x+.02,2.24,wid-.04,.30,txt,12.0,WHITE,True,align='center')
rows=[('A','原始 P0','原始 C0',9.3577),('B','适配 P1','原始 C0',.4213),('C','原始 P0','适配 C1',9.4098),('D','适配 P1','适配 C1',.4875)]
for i,(n,p,m,v) in enumerate(rows):
    y=2.60+i*.62
    rect(s,6.97,y,5.74,.61,'F1F5F9' if i%2==0 else WHITE,BORDER,.4)
    text(s,7.01,y+.08,.38,.38,n,15.1,BLUE,True,align='center')
    text(s,7.57,y+.08,1.28,.38,p,12.7,TEXT,False,align='center')
    text(s,8.98,y+.08,1.30,.38,m,12.7,TEXT,False,align='center')
    text(s,10.37,y+.08,.82,.38,f'{v:.2f}',13.8,BLUE if v<1 else RED,True,align='center')
    rect(s,11.37,y+.21,1.17,.17,'E7EFF7')
    rect(s,11.37,y+.21,1.17*v/10,.17,GREEN if v<1 else '6A89AA')
text(s,7.02,5.17,5.69,.25,'单位：rad/s；同一初态前5个物理步（局部对照）',11.8,GRAY)
# bottom conclusions row
block_title(s,.45,5.49,12.42,'结果解释与当前边界')
res=[(.55,3.55,'已验证的局部结论','仅调物理惯量 A→B：\n9.36 → 0.42 rad/s',BLUE),
     (4.19,3.88,'尚未解决的问题','后续长窗口仍有夹爪失稳、\n异常穿透与接触切换',TEXT),
     (8.17,4.52,'任务级结论','冻结策略可运行、动力学局部改善；\n完整零样本插孔迁移未获证明',RED)]
for x,wid,t,b,accent in res:
    rect(s,x,6.02,wid,.75,'F8FAFC',BORDER,.6)
    text(s,x+.08,6.05,wid-.16,.26,t,13.8,accent,True)
    text(s,x+.08,6.32,wid-.16,.41,b,11.65,GRAY)
s.notes_slide.notes_text_frame.text=(
'Sim2Sim 的目的不是证明已经能够 Sim2Real，而是冻结PPO权重并更换目标物理引擎，独立检查策略接口、底层控制和动力学/接触行为。初次MuJoCo闭环运行，在第2个物理子步发现J7角速度从正0.5047突然变成负5.1497rad/s，第三步出现力矩饱和和滑移，第四步右侧夹爪正接触力消失。异常发生时PPO的第一个动作还没执行完，因此不能直接归因于策略频繁改变动作。'
'通过源码审查发现Factory在PhysX中对刚体3x3转动惯量施加加0.01 I处理，MuJoCo Menagerie Panda资产没有自动继承该处理，于是针对11个映射刚体进行适配。'
'设计了Plant惯量原始/适配和控制器质量矩阵原始/适配的2x2实验：A原始/原始，B适配Plant/原始Controller，C原始Plant/适配Controller，D双方适配。前5步最大关节速度A9.3577、B0.4213、C9.4098、D0.4875 rad/s。实验支持物理惯量是初期振荡缓解的重要因素，而非证明唯一根因。'
'后续仍然发生夹爪接触失稳、接触集合变化与异常穿透。完整任务级MuJoCo零样本迁移没有被证明，研究阶段是PARTIAL。向面试官强调工程中控制变量、时序证据、模型差异与失败边界。')

# Avoid default Office effectRef=2 drop shadows (not present in reference deck).
for _sl in prs.slides:
 for _sh in _sl.shapes:
  for _node in _sh._element.iter():
   if _node.tag.endswith('effectRef'):
    _node.set('idx', '0')
prs.save(OUT)
print('saved',OUT,'size',Path(OUT).stat().st_size,'slides',len(prs.slides))
for i,slide in enumerate(prs.slides):
 print(i+1,len(slide.shapes),'notes',len(slide.notes_slide.notes_text_frame.text))
