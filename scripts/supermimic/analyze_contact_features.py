"""Independent paired comparison of fixed-budget WM feature-ablation results.
No fitting, checkpoint selection, simulator invocation or production changes.
"""
from pathlib import Path
import argparse, json,sys,hashlib
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source/viserdex'))
from experiment_contact_features import dataset,physical_delta,GROUPS,SIZES,sha,save_json


def cluster_ci(values,cluster,seed=8123,repeats=2000):
    if len(values)==0:return None
    unique,inv=cluster.unique(return_inverse=True)
    if len(unique)<2:return None
    sums=torch.zeros(len(unique),dtype=torch.float64).scatter_add_(0,inv,values.double())
    counts=torch.bincount(inv,minlength=len(unique)).double()
    generator=torch.Generator().manual_seed(seed);ids=torch.randint(len(unique),(repeats,len(unique)),generator=generator)
    estimates=sums[ids].sum(-1)/counts[ids].sum(-1)
    return torch.quantile(estimates,torch.tensor([.025,.975],dtype=torch.float64)).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    result=json.loads((a.fits/'result.json').read_text());assert result['status']=='complete'
    plan=result['plan'];test=dataset([a.experiment/f'branches_seed{s}' for s in plan['test_seeds']],'cpu')
    clusters=test['cluster'];response=test['response'];report=dict(scope='Fixed final200; two minibatch seeds of the same source WM, not two independent end-to-end trainings; CIs resample physical worlds and retain repeated starts',test_worlds=int(clusters.unique().numel()),test_state_world_starts=len(clusters),source_result_sha256=sha(a.fits/'result.json'),summary={},paired_comparisons={},physics_response_concentration={})
    models={'initial':torch.load(a.fits/'initial_test_predictions.pt',weights_only=False)}
    for key in result['fits']:models[key]=torch.load(a.fits/(key+'_test_predictions.pt'),weights_only=False)
    estimated={key:physical_delta(pred['minus'],pred['plus']) for key,pred in models.items()}
    for key in models:
        stats=result['initial_test'] if key=='initial' else result['fits'][key]['test']
        report['summary'][key]={}
        for stratum in ('all','pulse','hold5'):
            report['summary'][key][stratum]={}
            for h in ('1','8'):
                report['summary'][key][stratum][h]={g:stats[stratum][h][g] for g in ('q','qd','rotation')}
    for g,(name,r) in enumerate(zip(GROUPS,response[7].split(SIZES,-1))):
        mask=test['resolved'][7,:,g]
        if not mask.any():continue
        rr=r[mask];energies=rr.square().sum(-1);keep=energies.argmax();not_largest=torch.arange(len(rr))!=keep
        report['physics_response_concentration'][name]={'resolved':len(rr),'independent_worlds':int(clusters[mask].unique().numel()),'largest_squared_energy_fraction':float(energies.max()/energies.sum())}
        for arm in ('geometry','actuator','both'):
            cos_diffs=[];relative_errors=[]
            details=[]
            for seed in plan['optimizer_seeds']:
                base=estimated[f'baseline_seed{seed}'][7].split(SIZES,-1)[g][mask]
                alternative=estimated[f'{arm}_seed{seed}'][7].split(SIZES,-1)[g][mask]
                denom_r=rr.norm(dim=-1)
                bcos=(base*rr).sum(-1)/(base.norm(dim=-1)*denom_r).clamp_min(1e-20)
                acos=(alternative*rr).sum(-1)/(alternative.norm(dim=-1)*denom_r).clamp_min(1e-20)
                difference=acos-bcos;cos_diffs.append(difference)
                relative_errors.append((alternative-rr).norm(dim=-1)/denom_r-(base-rr).norm(dim=-1)/denom_r)
                details.append(dict(seed=seed,cosine_difference=float(difference.mean()),cosine_difference_cluster95=cluster_ci(difference,clusters[mask]),base_relative_vector_error=float((base-rr).norm()/rr.norm()),alternative_relative_vector_error=float((alternative-rr).norm()/rr.norm()),base_relative_vector_error_without_largest=float((base[not_largest]-rr[not_largest]).norm()/rr[not_largest].norm()) if not_largest.any() else None,alternative_relative_vector_error_without_largest=float((alternative[not_largest]-rr[not_largest]).norm()/rr[not_largest].norm()) if not_largest.any() else None))
            diff=torch.stack(cos_diffs).mean(0);rdiff=torch.stack(relative_errors).mean(0)
            report['paired_comparisons'][name+'/'+arm]=dict(per_seed=details,mean_across_two_sampling_seeds_cosine_difference=float(diff.mean()),world_cluster95=cluster_ci(diff,clusters[mask]),mean_per_sample_relative_error_difference=float(rdiff.mean()),relative_error_difference_world_cluster95=cluster_ci(rdiff,clusters[mask]))
    save_json(a.fits/'comparison.json',report)
    lines=['# 冻结 WM 特征实验：固定第200次更新的独立测试','',f"测试域环境：{report['test_worlds']}，重复起点：{report['test_state_world_starts']}。两个seed仅改变优化器采样顺序。",'', '| 模型 | q余弦 | qdot余弦 | 物体旋转余弦 | 旋转幅值比中位数 | 旋转相对向量误差 | 静止误报 |','|---|---:|---:|---:|---:|---:|---:|']
    for key,summary in report['summary'].items():
        s=summary['all']['8'];r=s['rotation'];quiet=r['quiet'];amp=r.get('amplitude_ratio',{});lines.append(f"| {key} | {s['q'].get('cosine_mean',float('nan')):.4f} | {s['qd'].get('cosine_mean',float('nan')):.4f} | {r.get('cosine_mean',float('nan')):.4f} | {amp.get('median',float('nan')):.3f} | {r.get('relative_vector_error',float('nan')):.3f} | {quiet['false_response_gt1e_4']}/{quiet['count']} |")
    rotation=report['physics_response_concentration'].get('rotation',{})
    lines+=['',f"8步旋转可辨响应：{rotation.get('resolved',0)}个起点 / {rotation.get('independent_worlds',0)}个物理域；方向余弦只统计此条件子集。",'方向比较必须同时看幅值、相对误差、无响应误报、脉冲/保持分组及独立域区间。零响应基线的相对误差为1。','本实验不包含完整接触动力学模型，也没有成对响应训练损失；不能据此否定或确认ContactNets或ContactSDF。']
    (a.fits/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
