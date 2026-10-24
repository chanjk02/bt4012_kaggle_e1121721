"""V21: add an eight-category CatBoost expert to V20's original ensemble.

Full reproduction: python train_v21.py --data /path/to/data --out ./V21_run
Reuse saved V19 predictions: add --baseline-dir baseline_predictions
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
import train_v12
import train_v19
from graph_postprocess import predecessor_correction

CATEGORIES=['feat_3','feat_112','feat_113','feat_148','feat_149','feat_15','feat_160','feat_7']
ITERATIONS=1500
NEW_EXPERT_FRACTION=0.5
GRAPH_ALPHA=0.02
VERIFIED_V20_SHA256='7102c1c7f5f55294bcd390e75dc1666833cd5b86c90b73d5197894f8e42e65e7'


def cat_frame(frame):
    x=frame[train_v12.FEATURES].copy()
    for col in CATEGORIES:x[col]=x[col].round(8).astype(str)
    return x


def build_model():
    return CatBoostClassifier(iterations=ITERATIONS,depth=6,learning_rate=.035,
        loss_function='Logloss',eval_metric='AUC',l2_leaf_reg=6,
        random_seed=4012,verbose=False,thread_count=4,allow_writing_files=False)


def load_edges(path):
    path=Path(path)
    if path.is_dir():
        matches=list(path.rglob('txs_edgelist.csv'))
        if len(matches)!=1:raise ValueError('Expected exactly one official txs_edgelist.csv')
        return pd.read_csv(matches[0])
    with zipfile.ZipFile(path) as archive:
        matches=[n for n in archive.namelist() if Path(n).name=='txs_edgelist.csv' and not n.startswith('__MACOSX')]
        if len(matches)!=1:raise ValueError('Expected exactly one official txs_edgelist.csv')
        with archive.open(matches[0]) as stream:return pd.read_csv(stream)


def find_data():
    return train_v12.find_data()


def run(data_path,out_dir,baseline_dir=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    tr,te,sample=train_v12.load_data(data_path)
    if baseline_dir is None:
        baseline_dir=out/'baseline_v19'
        train_v19.run(data_path,baseline_dir)
    baseline_dir=Path(baseline_dir)
    pbase={}
    cache=out/'baseline_predictions';cache.mkdir(exist_ok=True)
    for key in ['v12','v19']:
        path=baseline_dir/f'submission_{key}.csv'
        d=pd.read_csv(path)
        if list(d.columns)!=['index','target'] or not d['index'].is_unique:
            raise ValueError(f'Invalid baseline file: {path}')
        if len(d)!=len(te) or set(d['index'])!=set(te['index']):
            raise ValueError(f'Baseline indices do not match test rows: {path}')
        p=d.set_index('index').loc[te['index'],'target'].to_numpy()
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
            raise ValueError('Invalid baseline prediction values')
        pbase[key]=p
        destination=cache/path.name
        if path.resolve()!=destination.resolve():shutil.copyfile(path,destination)
    known=tr.label.notna().to_numpy()
    y=tr.loc[known,'label'].astype(int).to_numpy()
    steps=tr.loc[known,'time_step'].to_numpy()
    assert np.isin(y,[0,1]).all() and tr.time_step.max()<te.time_step.min()
    recent=steps>int(steps.max())-14
    if recent.sum()<1000:raise ValueError('Insufficient recent known labels')
    model=build_model()
    model.fit(cat_frame(tr.loc[known].loc[recent]),y[recent],cat_features=CATEGORIES)
    pc=model.predict_proba(cat_frame(te))[:,1]
    model_dir=out/'models';model_dir.mkdir(exist_ok=True)
    model.save_model(str(model_dir/'catboost_categories8_recent14_1500.cbm'))
    # Exact selected recipe: half of the old categorical contribution is retained.
    rank=lambda p:rankdata(p,method='average')/len(p)
    new_blend=.65*rank(pbase['v12'])+.35*rank(pc)
    pre_graph=(1-NEW_EXPERT_FRACTION)*pbase['v19']+NEW_EXPERT_FRACTION*new_blend
    edges=load_edges(data_path)
    p21,graph=predecessor_correction(edges,te.txId.to_numpy(),pre_graph,GRAPH_ALPHA)
    p20,_=predecessor_correction(edges,te.txId.to_numpy(),pbase['v19'],GRAPH_ALPHA)
    for name,p in [('submission_v21.csv',p21),('submission_v20_fallback.csv',p20)]:
        sub=sample[['index']].merge(pd.DataFrame({'index':te['index'],'target':p}),
            on='index',how='left',sort=False,validate='one_to_one')
        assert len(sub)==len(sample) and np.array_equal(sub['index'],sample['index'])
        assert sub.target.notna().all() and sub.target.between(0,1).all()
        train_v12.write_csv(sub,out/name)
    # Save raw new-model predictions to support independent model-reload checks.
    pd.DataFrame({'index':te['index'],'new_catboost_probability':pc}).to_csv(out/'new_expert_predictions.csv',index=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    fallback_hash=sha(out/'submission_v20_fallback.csv')
    meta={
        'version':'V21',
        'recipe':'half V19 + half (65% rank(V12) + 35% rank(eight-category 1500-tree CatBoost)), followed by the V20 2% predecessor correction',
        'equivalent_pre_graph_weights':{'rank_V12':.65,'rank_original_CatBoost':.175,'rank_new_CatBoost':.175},
        'categorical_features':CATEGORIES,'new_categorical_features':['feat_15','feat_160','feat_7'],
        'new_model_iterations':ITERATIONS,'new_model_depth':6,'new_model_learning_rate':.035,'new_model_l2_leaf_reg':6,'seed':4012,
        'known_training_rows':int(known.sum()),'recent14_known_rows':int(recent.sum()),
        'unknown_labels_excluded':int((~known).sum()),'prediction_rows':len(te),
        'graph':graph,'candidate_public_auc':None,
        'baseline_matches_verified_v20_bytes':fallback_hash==VERIFIED_V20_SHA256,
        'baseline_public_auc':.94859 if fallback_hash==VERIFIED_V20_SHA256 else None,
        'future_labels_used':False,'external_data_used':False,'pseudo_labels_used':False,
        'candidate_sha256':sha(out/'submission_v21.csv'),'fallback_sha256':fallback_hash,
        'validation_limitations':'Several overlapping chronological windows were reused during selection. Additional checks assess the fixed recipe, not an untouched final test. Public and private score gains remain unknown.',
    }
    (out/'run_metadata_v21.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
    return pd.read_csv(out/'submission_v21.csv')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=None)
    parser.add_argument('--out',default='BT4012_V21_run')
    parser.add_argument('--baseline-dir',default=None)
    args=parser.parse_args()
    run(args.data or find_data(),args.out,args.baseline_dir)
