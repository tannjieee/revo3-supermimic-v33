"""Eight synchronized views of actual saved policy-play capture data; CPU only."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refusing to overwrite a previous video'
    source = json.loads((a.dataset/'manifest.json').read_text())
    assert source['completed'] and source['mode'] == 'policy_rollout_rgbd'
    paths = sorted(a.dataset.glob('[0-9][0-9][0-9][0-9][0-9]/pair.npz'))
    assert len(paths) == source['samples']
    mask = np.zeros((360,640), bool)
    low, high = float('inf'), -float('inf')
    for path in paths:
        with np.load(path) as d:
            assert d['play_raw_rgb'].shape == (360,640,3)
            mask |= d['rgb_visible']
            z = d['depth_m']
            z = z[np.isfinite(z) & (z > 0) & (z < 3)]
            low, high = min(low,float(z.min())), max(high,float(z.max()))
    yy, xx = np.nonzero(mask)
    side = int(max(xx.max()-xx.min(),yy.max()-yy.min())+32)
    cx, cy = int((xx.max()+xx.min())//2), int((yy.max()+yy.min())//2)
    box = (cx-side//2,cy-side//2,cx-side//2+side,cy-side//2+side)
    roi_size = 300
    fonts = {s:ImageFont.truetype(str(a.font),s) for s in (17,19,21,24,30)}
    fps = float(source['capture_fps'])
    transitions = [json.loads(line) for line in (a.dataset/'policy_transitions.jsonl').read_text().splitlines()]
    by_step = {row['step']:row for row in transitions}
    cumulative = np.cumsum([0]+[int(row['reached']) for row in transitions])
    definitions = [
        ('01  原始 policy play', 'play_raw_rgb · 原生网格渲染 · 无 SH 增强', '完整相机画面：手、黄色骰子、支架、AprilTag', '由 32100 轮 Actor 实际驱动；不是零动作'),
        ('02  干净可见 RGB', 'rgb_clean · 高斯渲染 · 未做 SH 扰动', '应用真实场景可见掩码，背景和遮挡处为黑', '640×360 原始保存；此处固定裁剪放大'),
        ('03  增强后的 RGB 输入', 'rgb · ViserDex 原生 13 步 SH 链', '这是为预测器采集的 RGB 输入', '只改 SH 颜色；位置、尺度、物理状态不变'),
        ('04  RGB 可见掩码', 'rgb_visible · 场景 first-hit 语义分割', '白色：相机能看见的骰子；黑色：其余区域', '手指、支架、背景均不进入骰子输入'),
        ('05  原始场景深度', 'depth_m · 848×480 · 光轴 Z 深度', '包含手、骰子和支架的第一可见表面', '这是虚拟共位深度；颜色仅用于显示'),
        ('06  骰子可见深度', 'depth_visible × depth_m', '只保留可见骰子深度，遮挡处不补全', '按相机内参映射到 RGB 网格，仅用于显示'),
        ('07  点云输入：同相机视角', 'points_camera_m + point_valid', '1024 个可见采样点；颜色表示光轴深度', '用 K_rgb 投影，与 RGB 使用同一裁剪比例'),
        ('08  GT 位姿与 9 个关键点', 'T_camera_object + keypoints_camera_m', '中心 + 8 个语义包围盒角点：仿真真值', '包含不可见角点；只作标签，不是预测结果'),
    ]
    noise_cfg = source['capture'].get('point_noise')
    if noise_cfg:
        definitions[6] = ('07  加噪点云：同相机视角', 'points_camera_m · 沿相机光线扰动深度',
            f"深度噪声 σ={noise_cfg['depth_std_m']*1000:g} mm，截断 ±{noise_cfg['depth_clip_m']*1000:g} mm",
            '干净点另存；RGB 对应与 GT 标签保持不变')
    a.output.mkdir(parents=True)
    kwargs = dict(fps=fps,codec='libx264',quality=8,macro_block_size=16,
                  ffmpeg_params=['-pix_fmt','yuv420p','-threads','2','-movflags','+faststart'])
    writer = imageio.get_writer(str(a.output/'capture_dashboard.mp4'), **kwargs)
    raw_writer = imageio.get_writer(str(a.output/'original_policy_play.mp4'), **kwargs)
    snapshot_ids = set(np.linspace(0,len(paths)-1,4,dtype=int).tolist())
    snapshots = []
    max_uv_error = 0.
    frame_rows = []

    def color_depth(z, valid):
        normalized = np.clip((np.where(valid,z,low)-low)/(high-low),0,1)
        colored = (colormaps['viridis'](normalized)[...,:3]*255).astype('uint8')
        colored[~valid] = 0
        return colored

    def crop(rgb):
        return Image.fromarray(rgb).crop(box).resize((roi_size,roi_size),Image.Resampling.NEAREST)

    def project(points,k):
        h = points @ k.T
        assert (h[:,2] > 0).all()
        return h[:,:2]/h[:,2:]

    def display_uv(uv):
        return (uv-np.asarray(box[:2])+.5)*(roi_size/side)-.5

    try:
        for i,path in enumerate(paths):
            with np.load(path) as d:
                assert int(d['frame_id']) == i
                assert not d['rgb'][~d['rgb_visible']].any()
                points = d['points_camera_m'][d['point_valid']]
                uv = project(points,d['K_rgb'])
                error = float(np.abs(uv-d['point_uv_rgb'][d['point_valid']]).max())
                assert error < 1e-4
                max_uv_error = max(max_uv_error,error)
                xy = display_uv(uv)
                colors = color_depth(points[:,2],np.ones(len(points),bool))
                cloud = Image.new('RGB',(roi_size,roi_size),'black')
                cd = ImageDraw.Draw(cloud)
                for j in np.argsort(-points[:,2]):
                    x,y = xy[j]
                    cd.ellipse((x-1,y-1,x+1,y+1), fill=tuple(colors[j]))
                raw_depth = d['depth_m']
                scene_valid = np.isfinite(raw_depth) & (raw_depth>0) & (raw_depth<3)
                scene_depth = color_depth(raw_depth,scene_valid)
                # Display-only nearest-neighbor mapping of co-located optical rays.
                gy,gx = np.mgrid[:360,:640]
                kr,kd = d['K_rgb'],d['K_depth']
                dx = np.rint((gx-kr[0,2])/kr[0,0]*kd[0,0]+kd[0,2]).astype(int)
                dy = np.rint((gy-kr[1,2])/kr[1,1]*kd[1,1]+kd[1,2]).astype(int)
                in_bounds = (dx>=0)&(dx<raw_depth.shape[1])&(dy>=0)&(dy<raw_depth.shape[0])
                dx = dx.clip(0,raw_depth.shape[1]-1); dy = dy.clip(0,raw_depth.shape[0]-1)
                target_depth = color_depth(raw_depth[dy,dx],in_bounds & d['depth_visible'][dy,dx])
                label = crop(d['rgb_clean'])
                ld = ImageDraw.Draw(label)
                kp = display_uv(project(d['keypoints_camera_m'],kr))
                corners = d['keypoints_object_m']
                for u in range(1,9):
                    for v in range(u+1,9):
                        if np.count_nonzero(np.abs(corners[u]-corners[v])>1e-7)==1:
                            ld.line((tuple(kp[u]),tuple(kp[v])), fill='#ffb657',width=1)
                tco = d['T_camera_object']
                axes = np.vstack([np.zeros(3),np.eye(3)*.025]) @ tco[:3,:3].T+tco[:3,3]
                ap = display_uv(project(axes,kr))
                for j,col in enumerate(('#ff4e5c','#4dec81','#5aa7ff'),1):
                    ld.line((tuple(ap[0]),tuple(ap[j])),fill=col,width=3)
                    ld.text(tuple(ap[j]),'XYZ'[j-1],font=fonts[17],fill=col)
                for j,(x,y) in enumerate(kp):
                    ld.ellipse((x-3,y-3,x+3,y+3),fill='#ff64d8' if j==0 else 'white')
                panels = [Image.fromarray(d['play_raw_rgb']).resize((448,252)),
                          crop(d['rgb_clean']),crop(d['rgb']),crop(np.repeat((d['rgb_visible']*255).astype('uint8')[...,None],3,axis=2)),
                          Image.fromarray(scene_depth).resize((448,254),Image.Resampling.NEAREST),
                          crop(target_depth),cloud,label]
                canvas = Image.new('RGB',(1920,1120),'#0a1220')
                draw = ImageDraw.Draw(canvas)
                draw.text((20,14),'黄色骰子数据采集：原始 Play 与 8 个同步窗口',font=fonts[30],fill='white')
                draw.text((20,51),'新标定 2026-11-01  |  原支架已加载  |  Policy 32100  |  控制 20 Hz / 采集 10 Hz  |  正常速度',font=fonts[19],fill='#aebed2')
                for j,(title,field,note1,note2) in enumerate(definitions):
                    x=20+(j%4)*475; y=86+(j//4)*454
                    draw.rounded_rectangle((x,y,x+460,y+440),radius=10,fill='#172335',outline='#33455e',width=1)
                    draw.text((x+12,y+10),title,font=fonts[24],fill='white')
                    draw.text((x+12,y+43),field,font=fonts[17],fill='#9db5d2')
                    panel=panels[j]
                    px=x+(460-panel.width)//2; py=y+68+(300-panel.height)//2
                    canvas.paste(panel,(px,py))
                    if j in (4,5,6):
                        draw.text((x+12,y+374),f'深度颜色范围 {low*1000:.0f}—{high*1000:.0f} mm',font=fonts[17],fill='#b5cbe3')
                    draw.text((x+12,y+399),note1,font=fonts[17],fill='#d7e3f3')
                    draw.text((x+12,y+421),note2,font=fonts[17],fill='#aebed2')
                step=round(float(d['sim_time_s'])*source['policy']['control_hz'])
                status=by_step.get(step)
                timing=f"帧 {i:03d}/{len(paths)-1}  |  仿真时间 {float(d['sim_time_s']):05.2f}s  |  序列 {int(d['sequence_id'])+1}  |  有效点 {len(points)}"
                draw.text((20,1005),timing,font=fonts[21],fill='white')
                if status:
                    status_text=f"Hold 成功 {int(cumulative[step])} 次  |  最近控制步误差 {status['orientation_error_deg']:.1f}°  |  角速度 {status['angular_speed_rad_s']:.2f} rad/s"
                else:
                    status_text='初始状态：即将执行 Actor 动作；前 0.4 秒保留原策略预热'
                draw.text((20,1037),status_text,font=fonts[19],fill='#a7dbbb')
                t=tco[:3,3]*1000
                draw.text((1000,1005),f'GT 中心 / 相机坐标 [mm]：X {t[0]:.1f}   Y {t[1]:.1f}   Z {t[2]:.1f}',font=fonts[19],fill='#ffd79d')
                draw.text((1000,1037),'位姿和关键点来自仿真真值；本视频没有运行预测器或训练',font=fonts[19],fill='#ffd79d')
                draw.text((20,1082),'每帧保存：RGB / 干净 RGB / 原始 Play / 深度 / 掩码 / 点云 / K / T / 9 点标签 / 关节 / 下一步动作 / 时间戳',font=fonts[19],fill='#c4d3e5')
                writer.append_data(np.asarray(canvas))
                raw_canvas=Image.new('RGB',(640,416),'#0a1220')
                raw_canvas.paste(Image.fromarray(d['play_raw_rgb']),(0,56))
                rd=ImageDraw.Draw(raw_canvas)
                rd.text((12,5),'原始 policy play：无 SH 增强 / 支架已加载',font=fonts[21],fill='white')
                rd.text((12,32),f'Policy 32100  |  t={float(d["sim_time_s"]):.2f}s  |  帧 {i:03d}',font=fonts[17],fill='#b5cbe3')
                raw_writer.append_data(np.asarray(raw_canvas))
                if i in snapshot_ids:
                    canvas.save(a.output/f'frame_{i:03d}.png')
                    snapshots.append(canvas.resize((960,560)))
                frame_rows.append(dict(frame=i,sim_time_s=float(d['sim_time_s']),valid_points=len(points),projection_error_px=error))
            if i%50==0: print(f'Encoded {i+1}/{len(paths)}',flush=True)
    finally:
        writer.close(); raw_writer.close()
    for name,shape in [('capture_dashboard.mp4',(1120,1920,3)),('original_policy_play.mp4',(416,640,3))]:
        reader=imageio.get_reader(str(a.output/name)); count=0
        for frame in reader:
            assert frame.shape==shape
            count+=1
        reader.close()
        assert count==len(paths)
    contact=Image.new('RGB',(1920,1120))
    for i,frame in enumerate(snapshots): contact.paste(frame,((i%2)*960,(i//2)*560))
    contact.save(a.output/'contact_sheet.png')
    result=dict(completed=True,source_dataset=str(a.dataset.resolve()),frames=len(paths),fps=fps,
        duration_s=len(paths)/fps,all_videos_fully_decoded=True,max_projection_error_px=max_uv_error,
        roi_box_rgb=box,depth_color_range_m=[low,high],calibration_sha256=source['capture']['calibration_sha256'],
        fixture=source['capture']['fixture'],policy=source['policy'],panels=definitions,
        source_data_modified=False,gt_labels_are_predictions=False,point_noise=noise_cfg,frame_records=frame_rows)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k in ['completed','frames','fps','duration_s','all_videos_fully_decoded','max_projection_error_px']}),flush=True)


if __name__=='__main__':
    main()
