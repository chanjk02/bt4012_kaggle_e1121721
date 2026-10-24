"""Final V24: 60% V12 and 40% equally mixed categorical experts.

Full reproduction: python train_v24.py --data /path/to/data --out ./V24_run
Fast reproduction: add --baseline-dir baseline_v23
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
import train_v23
from graph_postprocess import predecessor_correction

CAT_WEIGHT=.40
GRAPH_ALPHA=.02
VERIFIED_V23_SHA256='e6850c2821d861ed293479abb960deba5ed3a4c220c38165a7338eb293eadf01'


def find_data():return train_v12.find_data()


def run(data_path,out_dir,baseline_dir=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    if baseline_dir is None:
        baseline_dir=out/'baseline_v23'
        train_v23.run(data_path,baseline_dir)
    baseline_dir=Path(baseline_dir)
    _,te,sample=train_v12.load_data(data_path)
    baseline_csv=baseline_dir/'submission_v23.csv'
    baseline=pd.read_csv(baseline_csv)
    if len(baseline)!=len(te) or not baseline['index'].is_unique or set(baseline['index'])!=set(te['index']):
        raise ValueError('Invalid baseline submission rows')
    baseline_pre=np.load(baseline_dir/'new_pre_graph.npy')
    if baseline_pre.shape!=(len(te),):raise ValueError('Invalid baseline rank vector')
    edges=train_v21.load_edges(data_path)
    baseline_reconstructed,_=predecessor_correction(edges,te.txId.to_numpy(),baseline_pre,GRAPH_ALPHA)
    baseline_error=float(np.max(np.abs(baseline_reconstructed-baseline.set_index('index').loc[te['index'],'target'].to_numpy())))
    if baseline_error>1e-12:raise ValueError('Baseline cache does not match test rows')
    cache=baseline_dir/'baseline_v22'
    d=pd.read_csv(cache/'baseline_v21/baseline_predictions/submission_v12.csv')
    if len(d)!=len(te) or not d['index'].is_unique or set(d['index'])!=set(te['index']):
        raise ValueError('Invalid cached V12 prediction rows')
    p12=d.set_index('index').loc[te['index'],'target'].to_numpy()
    target=train_v21.cat_frame(te)
    probs={}
    paths={1500:cache/'baseline_v21/models/catboost_categories8_recent14_1500.cbm',
           2000:cache/'models/catboost_categories8_recent14_2000.cbm'}
    for count,path in paths.items():
        model=CatBoostClassifier();model.load_model(str(path))
        probs[count]=model.predict_proba(target,thread_count=4)[:,1]
    rank=lambda p:rankdata(p,method='average')/len(p)
    categorical=.5*rank(probs[1500])+.5*rank(probs[2000])
    pre=(1-CAT_WEIGHT)*rank(p12)+CAT_WEIGHT*categorical
    pred,graph=predecessor_correction(edges,te.txId.to_numpy(),pre,GRAPH_ALPHA)
    sub=sample[['index']].merge(pd.DataFrame({'index':te['index'],'target':pred}),
        on='index',how='left',sort=False,validate='one_to_one')
    assert len(sub)==len(sample) and np.array_equal(sub['index'],sample['index'])
    assert sub.target.notna().all() and sub.target.between(0,1).all()
    train_v12.write_csv(sub,out/'submission_v24.csv')
    fallback=out/'submission_v23_fallback.csv'
    if baseline_csv.resolve()!=fallback.resolve():shutil.copyfile(baseline_csv,fallback)
    np.save(out/'new_pre_graph.npy',pre)
    pd.DataFrame({'index':te['index'],'v12':p12,'cat8_1500':probs[1500],
                  'cat8_2000':probs[2000]}).to_csv(out/'component_predictions.csv',index=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    fallback_hash=sha(fallback)
    meta={
        'version':'V24','final_requested_submission':True,
        'weights_before_graph':{'rank_V12':.60,'rank_CatBoost8_1500':.20,'rank_CatBoost8_2000':.20},
        'graph':graph,'baseline_reconstruction_max_error':baseline_error,
        'baseline_matches_verified_v23_bytes':fallback_hash==VERIFIED_V23_SHA256,
        'baseline_public_auc':.94888 if fallback_hash==VERIFIED_V23_SHA256 else None,
        'candidate_public_auc':None,'leaderboard_feedback_informed_selection':True,
        'new_pseudo_labels_used':False,'domain_weights_used':False,
        'future_labels_used':False,'external_data_used':False,
        'selection_note':'A moderate increase to 40% categorical weight was fixed before an additional cutoff-26 check, which improved pooled and mean per-step AUC. Recent historical windows favored more categorical weight; an older stress window worsened. Results are mixed and reused, not an untouched final test.',
        'candidate_sha256':sha(out/'submission_v24.csv'),'fallback_sha256':fallback_hash,
    }
    (out/'run_metadata_v24.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
    return sub


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=None)
    parser.add_argument('--out',default='BT4012_V24_run')
    parser.add_argument('--baseline-dir',default=None)
    args=parser.parse_args()
    run(args.data or find_data(),args.out,args.baseline_dir)
