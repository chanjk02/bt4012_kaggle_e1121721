"""V23: combine the two proven eight-category experts with the V12 core.

Full reproduction: python train_v23.py --data /path/to/data --out ./V23_run
Reuse the included cache: add --baseline-dir baseline_v22
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
import train_v12
import train_v21
import train_v22
from graph_postprocess import predecessor_correction

GRAPH_ALPHA=0.02
VERIFIED_V22_SHA256='02705e23b175c62a12eb1636d789682d9eb5b27836edfb109aba13bd752c815c'


def find_data():return train_v12.find_data()


def run(data_path,out_dir,baseline_dir=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    if baseline_dir is None:
        baseline_dir=out/'baseline_v22'
        train_v22.run(data_path,baseline_dir)
    baseline_dir=Path(baseline_dir)
    _,te,sample=train_v12.load_data(data_path)
    base_csv=baseline_dir/'submission_v22.csv'
    baseline=pd.read_csv(base_csv)
    if len(baseline)!=len(te) or not baseline['index'].is_unique or set(baseline['index'])!=set(te['index']):
        raise ValueError('Invalid baseline submission rows')
    pre22=np.load(baseline_dir/'new_pre_graph.npy')
    if pre22.shape!=(len(te),):raise ValueError('Invalid baseline rank vector')
    edges=train_v21.load_edges(data_path)
    reproduced,_=predecessor_correction(edges,te.txId.to_numpy(),pre22,GRAPH_ALPHA)
    baseline_error=float(np.max(np.abs(reproduced-baseline.set_index('index').loc[te['index'],'target'].to_numpy())))
    if baseline_error>1e-12:raise ValueError('Baseline cache does not match test row order')
    d=pd.read_csv(baseline_dir/'baseline_v21/baseline_predictions/submission_v12.csv')
    if len(d)!=len(te) or not d['index'].is_unique or set(d['index'])!=set(te['index']):
        raise ValueError('Invalid V12 cached prediction rows')
    p12=d.set_index('index').loc[te['index'],'target'].to_numpy()
    probability={}
    paths={1500:baseline_dir/'baseline_v21/models/catboost_categories8_recent14_1500.cbm',
           2000:baseline_dir/'models/catboost_categories8_recent14_2000.cbm'}
    target=train_v21.cat_frame(te)
    for count,path in paths.items():
        model=CatBoostClassifier();model.load_model(str(path))
        probability[count]=model.predict_proba(target,thread_count=4)[:,1]
    rank=lambda p:rankdata(p,method='average')/len(p)
    pre23=.65*rank(p12)+.35*(.5*rank(probability[1500])+.5*rank(probability[2000]))
    p23,graph=predecessor_correction(edges,te.txId.to_numpy(),pre23,GRAPH_ALPHA)
    sub=sample[['index']].merge(pd.DataFrame({'index':te['index'],'target':p23}),
        on='index',how='left',sort=False,validate='one_to_one')
    assert len(sub)==len(sample) and np.array_equal(sub['index'],sample['index'])
    assert sub.target.notna().all() and sub.target.between(0,1).all()
    train_v12.write_csv(sub,out/'submission_v23.csv')
    fallback=out/'submission_v22_fallback.csv'
    if base_csv.resolve()!=fallback.resolve():shutil.copyfile(base_csv,fallback)
    np.save(out/'new_pre_graph.npy',pre23)
    pd.DataFrame({'index':te['index'],'v12':p12,'cat8_1500':probability[1500],
                  'cat8_2000':probability[2000]}).to_csv(out/'component_predictions.csv',index=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    fallback_hash=sha(fallback)
    meta={
        'version':'V23','weights_before_graph':{'rank_V12':.65,'rank_CatBoost8_1500':.175,'rank_CatBoost8_2000':.175},
        'graph':graph,'baseline_reconstruction_max_error':baseline_error,
        'baseline_matches_verified_v22_bytes':fallback_hash==VERIFIED_V22_SHA256,
        'baseline_public_auc':.94885 if fallback_hash==VERIFIED_V22_SHA256 else None,
        'candidate_public_auc':None,'new_pseudo_labels_used':False,'domain_weights_used':False,
        'leaderboard_feedback_informed_selection':True,
        'future_labels_used':False,'external_data_used':False,
        'validation_summary':'Four of six reused historical windows improved. The two decreases were small; recent-period evidence is mixed. This candidate was chosen after rejecting several unreliable training experiments. Public and private gains are unknown.',
        'candidate_sha256':sha(out/'submission_v23.csv'),'fallback_sha256':fallback_hash,
    }
    (out/'run_metadata_v23.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
    return sub


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=None)
    parser.add_argument('--out',default='BT4012_V23_run')
    parser.add_argument('--baseline-dir',default=None)
    args=parser.parse_args()
    run(args.data or find_data(),args.out,args.baseline_dir)
