#!/usr/bin/env python3
"""Numerically exact 3-D Piper representation illustrations, PNG/GIF only.

Run from the repository:
  python script/plot_piper_representation.py --font /path/to/NotoSansCJKsc-Regular.otf
Requires numpy, matplotlib, Pillow, and the repository's pyAgxArm dependency.
All poses come from Piper modified-DH FK on explicitly specified example joints.
These are synthetic trajectories, not recorded episodes or trained-policy results.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opendm.data import se3
from opendm.kinematics import piper
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

BLUE, ORANGE = '#2467BD', '#CC7222'
TEAL, PURPLE, PINK = '#07866E', '#7856B5', '#BF397B'
AXIS = ['#C64040', '#2C8A51', '#326BC0']
INK, MUTED = '#21364D', '#597086'
STEPS = 21


def pose(p=(0, 0, 0), r=(0, 0, 0)):
    return se3.make_transform(np.asarray(p, float), se3.rotvec_to_mat(np.asarray(r, float)))


def chain(q):
    """Actual MDH chain vertices; no manually positioned illustrative links."""
    a, b = piper._link_transforms(np.asarray(q)[None])
    acc = np.eye(4)
    points = [acc[:3, 3].copy()]
    for ai, bi in zip(a[0], b[0]):
        acc = acc @ ai
        points.append(acc[:3, 3].copy())
        acc = acc @ bi
        points.append(acc[:3, 3].copy())
    np.testing.assert_allclose(acc, piper.fk(q), atol=1e-12)
    return np.asarray(points)


def points_in(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def build_data():
    q0 = np.array([[-.6, 1.2, -.8, .2, .6, .1], [.6, 1.2, -.8, -.2, .6, -.1]])
    dq = np.array([[.16, .14, -.18, .18, -.12, .2], [-.14, .12, -.16, -.15, .1, -.16]])
    q = q0[:, None] + np.linspace(0, 1, STEPS)[None, :, None] * dq[:, None]
    limits = piper.PIPER_JOINT_LIMITS
    assert np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])
    base = np.stack([np.eye(4), piper.T_RIGHT_BASE_TO_LEFT_BASE])
    local = piper.fk(q)
    sdk_pose = np.array([fk_from_mdh(get_mdh('piper'), x.tolist()) for x in q.reshape(-1, 6)]).reshape(2, STEPS, 6)
    sdk_T = se3.make_transform(sdk_pose[..., :3], se3.rpy_to_mat(sdk_pose[..., 3:]))
    world = base[:, None] @ local
    body = np.linalg.inv(local[:, 0])[:, None] @ local
    pair = np.linalg.inv(world[0, 0]) @ world[1, 0]
    # C is a proper rotation, taking W's gravity vector (0,0,-1) to G's +z.
    C = pose(r=(np.pi, 0, 0))
    settings = [(0., .00, .00), (65., .10, -.08), (-55., -.12, .06)]
    H = np.stack([pose((a, b, 0), (0, 0, np.deg2rad(yaw))) @ C for yaw, a, b in settings])
    gravity = H[:, None, None] @ world[None]
    # A deliberately selected, feasible virtual base for a demonstration, not an estimate.
    virtual = np.stack([base[i] @ pose((.01, .01, 0), (0, 0, .1)) for i in range(2)])
    targets = np.linalg.inv(virtual)[:, None] @ world
    recovered = np.empty_like(q)
    residuals = []
    for arm in range(2):
        seed = q[arm, 0].copy()  # illustrative IK; does not test joint-free initialization
        for k in range(STEPS):
            seed, pe, re = piper.ik(targets[arm, k], seed, max_iters=300,
                                    pos_tol_m=1e-9, rot_tol_rad=1e-9)
            recovered[arm, k] = seed
            residuals.append((pe, re))
    reconstructed = virtual[:, None] @ piper.fk(recovered)
    rotations = np.concatenate([world.reshape(-1, 4, 4), H, virtual])[:, :3, :3]
    checks = {
        'fk_vs_sdk_matrix_max': float(np.max(np.abs(local-sdk_T))),
        'rot6d_roundtrip_max': float(np.max(np.abs(se3.rot6d_to_mat(se3.mat_to_rot6d(body[..., :3, :3]))-body[..., :3, :3]))),
        'rotation_orthogonality_max': float(np.max(np.abs(rotations.swapaxes(-1, -2) @ rotations - np.eye(3)))),
        'rotation_det_error_max': float(np.max(np.abs(np.linalg.det(rotations) - 1))),
        'local_world_roundtrip_max': float(np.max(np.abs(np.linalg.inv(base)[:, None] @ world - local))),
        'body_local_vs_world_max': float(np.max(np.abs(np.linalg.inv(world[:, 0])[:, None] @ world - body))),
        'body_after_episode_transform_max': float(np.max(np.abs(np.linalg.inv(gravity[:, :, 0])[:, :, None] @ gravity - body[None]))),
        'pair_after_episode_transform_max': float(np.max(np.abs(np.linalg.inv(gravity[:, 0, 0]) @ gravity[:, 1, 0] - pair))),
        'gravity_alignment_max': float(np.max(np.abs(H[:, :3, :3] @ np.array([0., 0., -1.]) - [0, 0, 1]))),
        'episode_roundtrip_max': float(np.max(np.abs(np.linalg.inv(H)[:, None, None] @ gravity - world[None]))),
        'ik_position_max_m': float(np.max(np.linalg.norm(reconstructed[..., :3, 3] - world[..., :3, 3], axis=-1))),
        'ik_rotation_matrix_max': float(np.max(np.abs(reconstructed[..., :3, :3] - world[..., :3, :3]))),
    }
    for name, value in checks.items():
        assert value < (1e-7 if name.startswith('ik_') else 1e-12), (name, value)
    assert np.max(np.abs(gravity[1] - gravity[0])) > .1
    return dict(q=q, base=base, local=local, world=world, body=body, pair=pair,
                H=H, gravity=gravity, virtual=virtual, targets=targets,
                recovered=recovered, reconstructed=reconstructed,
                settings=settings, checks=checks)


def frame(ax, T, name='', length=.06, alpha=1., labels=False):
    """Columns of R are basis vectors in the axes' reference coordinate system."""
    origin = T[:3, 3]
    for i, color in enumerate(AXIS):
        direction = T[:3, i] * length
        ax.quiver(*origin, *direction, color=color, alpha=alpha,
                  linewidth=1.7, arrow_length_ratio=.2, normalize=False)
        if labels:
            end = origin + direction * 1.22
            ax.text(*end, 'xyz'[i], color=color, fontsize=9)
    if name:
        ax.text(*(origin + np.array([.007, .007, .009])), name,
                color=PURPLE, fontsize=10, bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=.5))


def arrow(ax, a, b, color=TEAL):
    ax.quiver(*a, *(b-a), color=color, linewidth=2., arrow_length_ratio=.13)


def trajectory(ax, Ts, color, name, frames=True, length=.055, endpoint=-1):
    p = Ts[:, :3, 3]
    ax.plot(*p.T, color=color, linewidth=2.4)
    ax.scatter(*p[0], color=color, s=34, depthshade=False)
    ax.scatter(*p[endpoint], edgecolors=color, facecolors='white', s=48, depthshade=False)
    if frames:
        frame(ax, Ts[0], f'{name}(t)', length)
        frame(ax, Ts[endpoint], '', length, alpha=.6)


def bounds(points, pad=.08):
    points = np.concatenate([np.asarray(p).reshape(-1, 3) for p in points])
    lo, hi = points.min(0), points.max(0)
    center = (lo+hi)/2
    radius = max(float(np.max(hi-lo))/2 + pad, .10)
    return center-radius, center+radius


def setup(ax, title, lim, reference):
    ax.set_title(title, fontsize=15, pad=15, color=INK)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(lim[0][0], lim[1][0]); ax.set_ylim(lim[0][1], lim[1][1]); ax.set_zlim(lim[0][2], lim[1][2])
    ax.view_init(elev=26, azim=42)
    for axis, label in zip([ax.xaxis, ax.yaxis, ax.zaxis], ['x', 'y', 'z']):
        axis.set_major_locator(plt.MaxNLocator(4))
        axis.set_pane_color((.96,.97,.99,1))
        axis.label.set_color(MUTED)
    ax.set_xlabel(f'x [{reference}] / m', labelpad=8)
    ax.set_ylabel(f'y [{reference}] / m', labelpad=8)
    ax.set_zlabel(f'z [{reference}] / m', labelpad=8)
    ax.tick_params(labelsize=9, pad=1)
    ax.grid(True, alpha=.3)


TITLES = {
    's0': '真实关节：关节空间差分',
    's1': '各自 base 系：位置差与轴角向量差',
    's2': '当前末端系：严格 SE(3) 相对动作',
    's3a': '双臂 state：从各自 base 系统一到 W',
    's3': '增加观测：右末端相对当前左末端的位姿',
    's4': '重力对齐：每 episode 固定的水平坐标变换',
    's5': '虚拟 base：真实 IK 计算与 FK 回验',
}


def page(key, n=2):
    fig=plt.figure(figsize=(18, 11), facecolor='white')
    label='S3a' if key=='s3a' else key.upper()
    fig.text(.035,.95,f'{label}  |  {TITLES[key]}', fontsize=23, color=INK)
    fig.text(.035,.91,'蓝：左臂 L    橙：右臂 R    坐标轴：红 x / 绿 y / 蓝 z    实心：t    空心：t+k',fontsize=12,color=MUTED)
    fig.text(.035,.88,'Piper MDH 正运动学生成的合成示例；不是采集数据。三维正交投影，各轴等米制比例；坐标轴方向均由旋转矩阵列向量计算。',fontsize=11,color=MUTED)
    gs=fig.add_gridspec(1,n,left=.035,right=.955,bottom=.32,top=.81,wspace=.22)
    return fig, gs


def footer(fig, state, action, formula, note):
    fig.text(.04,.255,state,fontsize=15,color=BLUE)
    fig.text(.04,.205,action,fontsize=15,color=TEAL)
    fig.text(.04,.15,formula,fontsize=15,color=INK)
    fig.text(.04,.09,note,fontsize=12,color=PURPLE)
    fig.text(.04,.035,'全部组均有 state；双臂左→右拼接；夹爪预测未来绝对宽度 g(t+k)，不做差分。末端为 flange，TCP offset = 0。',fontsize=11,color=MUTED)


def vec(a):
    return '['+', '.join(f'{x:+.4f}' for x in np.asarray(a))+']'


def robot(ax, q, T, color, alpha=.5, style='-'):
    vertices = points_in(T,chain(q))
    ax.plot(*vertices.T, color=color, alpha=alpha, linestyle=style, linewidth=2.2)


def own(ax,D,i,title,k=20):
    Ts=D['local'][i]
    lim=bounds([Ts[:,:3,3], [0,0,0], chain(D['q'][i,0])],.065)
    setup(ax,title,lim,'B_L' if i==0 else 'B_R')
    frame(ax,np.eye(4),'B_L' if i==0 else 'B_R',.08,labels=True)
    robot(ax,D['q'][i,0],np.eye(4),[BLUE,ORANGE][i],.23)
    trajectory(ax,Ts[:k+1],[BLUE,ORANGE][i],['L','R'][i])
    return lim


def world_points(D):
    return [D['world'][..., :3, 3], D['base'][:, :3, 3]] + [
        points_in(D['base'][i], chain(D['q'][i,k])) for i in range(2) for k in [0,-1]
    ]


def world_scene(ax,D,title,lim=None,frames=True,reference_frame=True):
    if lim is None:
        lim=bounds(world_points(D),.1)
    setup(ax,title,lim,'W')
    if reference_frame:
        frame(ax,np.eye(4),'W = B_L',.1,labels=True)
    for i,col in enumerate([BLUE,ORANGE]):
        robot(ax,D['q'][i,0],D['base'][i],col,.25)
        trajectory(ax,D['world'][i],col,['L','R'][i],frames,.06)
    return lim


def render(D,key,episode=0,k=20):
    fig,gs=page(key,3 if key=='s3a' else 2)
    if key=='s0':
        ax=fig.add_subplot(gs[0],projection='3d')
        world_scene(ax,D,'几何回验：真实 MDH 关节链（W 仅用于绘图）',frames=False)
        for i,col in enumerate([BLUE,ORANGE]): robot(ax,D['q'][i,-1],D['base'][i],col,.6,'--')
        ax=fig.add_subplot(gs[1]);ids=np.arange(1,7)
        for i,col,name in [(0,BLUE,'L'),(1,ORANGE,'R')]:
            ax.plot(ids,D['q'][i,-1]-D['q'][i,0],'-o',color=col,label=name)
        ax.axhline(0,color=MUTED,lw=.8);ax.set_xticks(ids)
        ax.set_xlabel('Joint index');ax.set_ylabel('Joint delta / rad');ax.legend()
        ax.set_title('同一对 t / t+k 的 6 个关节差分',fontsize=15,color=INK);ax.grid(alpha=.2)
        footer(fig,'state：14D · 真实关节角 q(t) + 夹爪 g(t)',
               'action：14D · 关节空间；没有 Cartesian base/body 参考系',
               r'$\Delta q=q_{t+k}-q_t$',
               '实线关节链：t；虚线关节链：t+k。W 是画图参考系，不是网络额外输入。')
    elif key=='s1':
        for i,name in enumerate(['L','R']):
            ax=fig.add_subplot(gs[i],projection='3d');own(ax,D,i,f'{name}：自身 base 系中的绝对位姿')
            arrow(ax,D['local'][i,0,:3,3],D['local'][i,-1,:3,3])
        dp=D['local'][0,-1,:3,3]-D['local'][0,0,:3,3]
        dr=se3.mat_to_rotvec(D['local'][0,-1,:3,:3])-se3.mat_to_rotvec(D['local'][0,0,:3,:3])
        footer(fig,'state：14D · 每臂 [p(t), r(t), g(t)]，参考系分别为 B_L / B_R',
               'action：14D · base 系位置差 + 轴角向量差；绿色箭头为位置差',
               r'$\Delta p^B=p^B_{t+k}-p^B_t,\quad \Delta r^B=r^B_{t+k}-r^B_t$',
               f'L 数值：Δp = {vec(dp)} m；Δr = {vec(dr)} rad。轴角差不是 SE(3) 相对旋转。')
    elif key=='s2':
        ax=fig.add_subplot(gs[0],projection='3d'); own(ax,D,0,'state 不变：左臂仍表达在 B_L',k=k)
        ax=fig.add_subplot(gs[1],projection='3d')
        Ts=D['body'][0];lim=bounds([Ts[:,:3,3],[0,0,0]],.07)
        setup(ax,'action 换系：所有未来目标表达在 E_L(t)',lim,'E_L(t)')
        trajectory(ax,Ts[:k+1],BLUE,'L',frames=False)
        frame(ax,np.eye(4),'',.065,labels=True)
        frame(ax,Ts[k],'',.045,alpha=.65)
        ax.text2D(.02,.02,f'Origin: E_L(t) | target: E_L(t+{k})',transform=ax.transAxes,color=PURPLE,fontsize=10)
        arrow(ax,np.zeros(3),Ts[k,:3,3])
        footer(fig,'state：14D · 与 S1 相同，仍是各自 base 系的绝对末端位姿',
               'action：20D · 每臂 [Δp, rot6d(ΔR), g(t+k)]，参考系为该臂当前末端',
               r'$\Delta T=T_t^{-1}T_{t+k},\quad \Delta p=R_t^{\mathsf{T}}(p_{t+k}-p_t),\quad \Delta R=R_t^{\mathsf{T}}R_{t+k}$',
               f'图中展开左臂（右臂同样计算）：k={k}，Δp = {vec(Ts[k,:3,3])} m；当前锚点 t 固定。')
    elif key=='s3a':
        for i,name in enumerate(['L','R']): own(fig.add_subplot(gs[i],projection='3d'),D,i,f'输入 {name}：自身 base 系')
        ax=fig.add_subplot(gs[2],projection='3d');world_scene(ax,D,'输出：共享系 W = B_L',frames=False)
        frame(ax,D['base'][1],'B_R',.08)
        footer(fig,'state：14D · 双臂绝对位姿统一到 W；不添加双臂 AUX 特征',
               'action：20D · 与 S2 完全相同，仍在各臂当前末端系',
               r'$T_L^W=T_L^{B_L},\quad T_R^W={}^{W}T_{B_R}\,T_R^{B_R},\quad {}^{W}p_{B_R}=[0,-0.60,0]^{\mathsf{T}}\ \mathrm{m}$',
               '三个面板各自标明参考系，独立取景但各轴等比例；右臂只做真实的 −0.60 m 坐标平移。')
    elif key=='s3':
        ax=fig.add_subplot(gs[0],projection='3d');world_scene(ax,D,'S3a state：双臂在共享系 W',frames=False)
        frame(ax,D['world'][0,0],'E_L(t)',.07)
        frame(ax,D['world'][1,0],'E_R(t)',.07)
        arrow(ax,D['world'][0,0,:3,3],D['world'][1,0,:3,3],PINK)
        ax=fig.add_subplot(gs[1],projection='3d');pair=D['pair']
        setup(ax,'新增 AUX：右末端表达在当前左末端系',bounds([[0,0,0],pair[:3,3]],.1),'E_L(t)')
        frame(ax,np.eye(4),'E_L(t) = I',.075,labels=True);frame(ax,pair,'E_R(t)',.075)
        arrow(ax,np.zeros(3),pair[:3,3],PINK)
        footer(fig,'state：23D · S3a 的 14D + 双末端相对位姿的 9D AUX',
               'action：20D · 与 S3a 相同；粉色相对位姿仅为当前观测，不是未来动作',
               r'$T_{LR}=(T_L^W(t))^{-1}T_R^W(t),\quad \mathrm{AUX}=[p_{LR},\mathrm{rot6d}(R_{LR})]$',
               f'粉色箭头的精确坐标 p_LR = {vec(pair[:3,3])} m；三色轴表示 R_LR，不是手动指定朝向。')
    elif key=='s4':
        H=D['H'][episode];Ts=D['gravity'][episode]
        all_frames=np.linalg.inv(D['H'])
        lim=bounds(world_points(D)+[all_frames[:,:3,3]],.13)
        ax=fig.add_subplot(gs[0],projection='3d');world_scene(ax,D,'物理场景 W 不动；只更换参考系 G_e',lim,False,False)
        frame(ax,np.linalg.inv(H),f'G_{episode+1}',.13,labels=True)
        p=np.array([-.12,-.35,.25]);arrow(ax,p,p+[0,0,-.16],PURPLE)
        ax.text(*(p+[0,0,-.19]),'gravity',color=PURPLE,fontsize=10)
        ax=fig.add_subplot(gs[1],projection='3d')
        # Identical axis limits across episodes; no autoscale/camera animation.
        lim=bounds([D['gravity'][...,:3,3],[0,0,0]],.13)
        setup(ax,f'Episode {episode+1}：同一轨迹在 G_e 中的数值',lim,'G_e')
        frame(ax,np.eye(4),'G_e',.12,labels=True)
        for i,col in enumerate([BLUE,ORANGE]):trajectory(ax,Ts[i],col,['L','R'][i],frames=True,length=.055)
        yaw,a,b=D['settings'][episode]
        footer(fig,'state：14D · 双臂共享 G_e；yaw 与水平原点按 episode 改变，episode 内固定',
               'action：20D · body-frame 相对动作不变；主 S4 的高度零点保持不变',
               r'$T^{G_e}=H_eT^W,\quad {}^WT_{G_e}=H_e^{-1},\quad H_e=[R_z(\psi_e)R_x(\pi),\ (a_e,b_e,0)^{\mathsf{T}}]$',
               f'ψ={yaw:+.0f}°，a={a:+.2f} m，b={b:+.2f} m；L 当前坐标 = {vec(Ts[0,0,:3,3])} m；G_e 的 +z 沿重力。')
    elif key=='s5':
        ax=fig.add_subplot(gs[0],projection='3d');world_scene(ax,D,'W：原轨迹与虚拟机械臂 FK 重建',frames=False)
        for i,col in enumerate([BLUE,ORANGE]):
            robot(ax,D['recovered'][i,0],D['virtual'][i],col,.95)
            frame(ax,D['virtual'][i],['Bhat_L','Bhat_R'][i],.08)
            ax.scatter(*D['reconstructed'][i,-1,:3,3],marker='x',s=70,color=col)
        ax=fig.add_subplot(gs[1],projection='3d');targets=D['targets'][0]
        setup(ax,'左臂：先转到虚拟 base，再实际求解 IK',bounds([targets[:,:3,3],chain(D['recovered'][0,0]),[0,0,0]],.07),'Bhat_L')
        frame(ax,np.eye(4),'Bhat_L',.08,labels=True)
        trajectory(ax,targets,BLUE,'target',length=.05)
        robot(ax,D['recovered'][0,0],np.eye(4),TEAL,.8)
        p=piper.fk(D['recovered'][0])[:,:3,3]
        ax.scatter(*p[::4].T,marker='x',color=TEAL,s=36)
        err=D['checks']['ik_position_max_m']*1000
        footer(fig,'state：14D · 真实 IK 输出的虚拟关节 qhat(t) + 夹爪 g(t)',
               'action：14D · 虚拟关节空间差分；绿色叉号为 FK(IK(target)) 的回验位置',
               r'$T^{\hat B}=({}^{W}T_{\hat B})^{-1}T^W\ \longrightarrow\ \hat q=\mathrm{IK}(T^{\hat B}),\quad \Delta\hat q=\hat q_{t+k}-\hat q_t$',
               f'本图固定可行候选演示 IK，未运行 base 搜索；用真实 q(t) 初始化 IK。双臂 42 个目标的最大位置残差 {err:.2e} mm。')
    return fig


def serialize(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,dict):return {k:serialize(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [serialize(x) for x in obj]
    return obj


def as_image(fig):
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=125,facecolor='white');buf.seek(0)
    im=Image.open(buf).convert('RGB');plt.close(fig)
    return im


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font',required=True,type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'docs/zh/assets/piper_representation')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    font_manager.fontManager.addfont(str(args.font))
    name=font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({'font.family':['DejaVu Sans',name], 'axes.unicode_minus':False,
                         'font.size':11, 'mathtext.fontset':'dejavusans'})
    D=build_data()
    for key in TITLES:
        fig=render(D,key);fig.savefig(args.out/f'{key}.png',dpi=160,facecolor='white');plt.close(fig)
    frames=[as_image(render(D,'s4',episode=i)) for i in range(3)]
    frames[0].save(args.out/'s4_episodes.gif',save_all=True,append_images=frames[1:],duration=2500,loop=0,disposal=2)
    frames=[as_image(render(D,'s2',k=k)) for k in [1,4,8,12,16,20]]
    frames[0].save(args.out/'s2_body.gif',save_all=True,append_images=frames[1:],duration=[700]*5+[1800],loop=0,disposal=2)
    report={
        'description':'Synthetic FK example with Piper SDK MDH. All translations in metres; joints in radians. Not recorded data.',
        'convention':'T^F maps flange coordinates to F; columns of R are local axes in F. rot6d stores first two rows.',
        's5_limitation':'Fixed illustrative virtual base, not estimated. IK seeded with source joint state, then warm-started.',
        'data':D,
        'rot6d_body_final':se3.mat_to_rot6d(D['body'][:,-1,:3,:3]),
        'rot6d_pair':se3.mat_to_rot6d(D['pair'][:3,:3]),
    }
    (args.out/'numerical_validation.json').write_text(json.dumps(serialize(report),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(D['checks'],indent=2))
    print(f'Wrote 7 PNGs, 2 GIFs, numerical inputs and validation to {args.out}')


if __name__=='__main__':main()
