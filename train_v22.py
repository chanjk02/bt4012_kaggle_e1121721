"""V22: combine V21 with a longer-trained recent categorical expert.

Full reproduction: python train_v22.py --data /path/to/data --out ./V22_run
Reuse the included baseline cache: add --baseline-dir baseline_v21
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
from graph_postprocess import predecessor_correction

ITERATIONS=2000
NEW_EXPERT_FRACTION=0.25
GRAPH_ALPHA=0.02
VERIFIED_V21_SHA256='df4b6c758af5b4c9d0181dd426380449965fca8e8fb85bcc08d7910c04302d8c'


def make_model():
    return CatBoostClassifier(iterations=ITERATIONS,depth=6,learning_rate=.035,
        loss_function='Logloss',eval_metric='AUC',l2_leaf_reg=6,
        random_seed=4012,verbose=False,thread_count=4,allow_writing_files=False)


def find_data():return train_v12.find_data()


def run(data_path,out_dir,baseline_dir=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    tr,te,sample=train_v12.load_data(data_path)
    if baseline_dir is None:
        baseline_dir=out/'baseline_v21'
        train_v21.run(data_path,baseline_dir)
    baseline_dir=Path(baseline_dir)
    rank=lambda p:rankdata(p,method='average')/len(p)
    base={}
    for name in ['v12','v19']:
        d=pd.read_csv(baseline_dir/f'baseline_predictions/submission_{name}.csv')
        if not d['index'].is_unique or len(d)!=len(te) or set(d['index'])!=set(te['index']):
            raise ValueError('Invalid baseline prediction indices')
        base[name]=d.set_index('index').loc[te['index'],'target'].to_numpy()
    model_path=baseline_dir/'models/catboost_categories8_recent14_1500.cbm'
    old_model=CatBoostClassifier();old_model.load_model(str(model_path))
    old_probability=old_model.predict_proba(train_v21.cat_frame(te),thread_count=4)[:,1]
    pre21=.5*base['v19']+.5*(.65*rank(base['v12'])+.35*rank(old_probability))
    edges=train_v21.load_edges(data_path)
    baseline_prediction,_=predecessor_correction(edges,te.txId.to_numpy(),pre21,GRAPH_ALPHA)
    baseline_csv=baseline_dir/'submission_v21.csv'
    published=pd.read_csv(baseline_csv)
    if not published['index'].is_unique or len(published)!=len(te) or set(published['index'])!=set(te['index']):
        raise ValueError('Invalid V21 baseline submission')
    baseline_error=float(np.max(np.abs(published.set_index('index').loc[te['index'],'target'].to_numpy()-baseline_prediction)))
    if baseline_error>1e-12:raise ValueError('Baseline models and baseline CSV are inconsistent')
    known=tr.label.notna().to_numpy()
    y=tr.loc[known,'label'].astype(int).to_numpy()
    steps=tr.loc[known,'time_step'].to_numpy()
    recent=steps>int(steps.max())-14
    assert np.isin(y,[0,1]).all() and tr.time_step.max()<te.time_step.min()
    if recent.sum()<1000:raise ValueError('Insufficient recent known labels')
    model=make_model()
    model.fit(train_v21.cat_frame(tr.loc[known].loc[recent]),y[recent],cat_features=train_v21.CATEGORIES)
    probability=model.predict_proba(train_v21.cat_frame(te))[:,1]
    models=out/'models';models.mkdir(exist_ok=True)
    model.save_model(str(models/'catboost_categories8_recent14_2000.cbm'))
    expert_pre=.65*rank(base['v12'])+.35*rank(probability)
    pre22=(1-NEW_EXPERT_FRACTION)*pre21+NEW_EXPERT_FRACTION*expert_pre
    p22,graph=predecessor_correction(edges,te.txId.to_numpy(),pre22,GRAPH_ALPHA)
    sub=sample[['index']].merge(pd.DataFrame({'index':te['index'],'target':p22}),
        on='index',how='left',sort=False,validate='one_to_one')
    assert len(sub)==len(sample) and np.array_equal(sub['index'],sample['index'])
    assert sub.target.notna().all() and sub.target.between(0,1).all()
    train_v12.write_csv(sub,out/'submission_v22.csv')
    destination=out/'submission_v21_fallback.csv'
    if baseline_csv.resolve()!=destination.resolve():shutil.copyfile(baseline_csv,destination)
    pd.DataFrame({'index':te['index'],'new_expert_probability':probability}).to_csv(out/'new_expert_predictions.csv',index=False)
    # Exact float arrays support direct auditing without CSV round-trip rounding.
    np.save(out/'baseline_pre_graph.npy',pre21)
    np.save(out/'new_pre_graph.npy',pre22)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    fallback_hash=sha(destination)
    meta={
        'version':'V22','new_expert_iterations':ITERATIONS,
        'new_expert_fraction':NEW_EXPERT_FRACTION,
        'recipe':'blend V21 pre-graph predictions with 65% rank(V12) + 35% rank(new 2000-tree eight-category CatBoost), then apply the same 2% predecessor correction',
        'categorical_features':train_v21.CATEGORIES,'depth':6,'learning_rate':.035,
        'l2_leaf_reg':6,'seed':4012,'recent_horizon':14,
        'known_training_rows':int(known.sum()),'recent14_known_rows':int(recent.sum()),
        'unknown_labels_excluded':int((~known).sum()),'prediction_rows':len(te),
        'baseline_reconstruction_max_error':baseline_error,'graph':graph,
        'baseline_matches_verified_v21_bytes':fallback_hash==VERIFIED_V21_SHA256,
        'baseline_public_auc':.94880 if fallback_hash==VERIFIED_V21_SHA256 else None,
        'candidate_public_auc':None,'future_labels_used':False,
        'external_data_used':False,'pseudo_labels_used':False,
        'candidate_sha256':sha(out/'submission_v22.csv'),'fallback_sha256':fallback_hash,
        'validation_limitations':'Chronological windows overlap and include reused data. Additional windows assess a fixed recipe, not an untouched final test. Public and private gains are unknown.'
    }
    (out/'run_metadata_v22.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
    return sub


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=None)
    parser.add_argument('--out',default='BT4012_V22_run')
    parser.add_argument('--baseline-dir',default=None)
    args=parser.parse_args()
    run(args.data or find_data(),args.out,args.baseline_dir)
