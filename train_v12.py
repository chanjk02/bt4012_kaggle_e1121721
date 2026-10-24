"""V12: rank blend of V8 and recent-regime LightGBM experts."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

FEATURES=[f'feat_{i}' for i in range(1,166) if i not in [100,101,103,136,139]]
FOLDS=[(24,35),(26,35),(28,35),(30,35)]
MEMORY_HORIZONS=(7,14)
EXPERT_WEIGHT=.6
REGIME_START=24

def build_model():
 return lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=.035,n_estimators=352,num_leaves=15,min_child_samples=55,colsample_bytree=1.,subsample=1.,subsample_freq=0,reg_lambda=6.,max_bin=127,random_state=4012,n_jobs=4,verbosity=-1)

def write_csv(frame,path):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');frame.to_csv(tmp,index=False)
 check=pd.read_csv(tmp);assert len(check)==len(frame) and list(check.columns)==list(frame.columns)
 tmp.replace(path)

def load_data(path):
 path=Path(path);names=['train.csv','test.csv','sample_submission.csv'];frames=[]
 if path.is_dir():
  for n in names:
   f=list(path.rglob(n))
   if len(f)!=1:raise ValueError(f'Expected one {n}; found {len(f)}')
   frames.append(pd.read_csv(f[0]))
 else:
  with zipfile.ZipFile(path) as z:
   for n in names:
    f=[x for x in z.namelist() if Path(x).name==n and not x.startswith('__MACOSX')]
    if len(f)!=1:raise ValueError(f'Expected one {n}')
    with z.open(f[0]) as stream:frames.append(pd.read_csv(stream))
 return tuple(frames)

def predict(source,y,target,source_steps,model_dir=None):
 source=source[FEATURES].astype('float32');target=target[FEATURES].astype('float32');steps=np.asarray(source_steps);y=np.asarray(y,dtype=int)
 pred={};v4=None
 for name,params,sw in [('v4',{},None),('class_weight_1p6',{},np.where(y==1,1.6,1.)),('goss',dict(data_sample_strategy='goss',n_estimators=500),None)]:
  m=build_model().set_params(**params).fit(source,y,sample_weight=sw);pred[name]=m.predict_proba(target)[:,1]
  if name=='v4':v4=m
  if model_dir is not None:m.booster_.save_model(str(model_dir/f'{name}.txt'))
 gain=v4.booster_.feature_importance(importance_type='gain');order=np.argsort(-gain,kind='stable');selected=[FEATURES[i] for i in order[:80]]
 top=build_model().fit(source[selected],y);pred['top80']=top.predict_proba(target[selected])[:,1]
 if model_dir is not None:
  top.booster_.save_model(str(model_dir/'top80.txt'));(model_dir/'selected_features.json').write_text(json.dumps(selected,indent=2))
 p5=.75*pred['v4']+.25*pred['class_weight_1p6'];p8=.75*p5+.25*(.5*pred['goss']+.5*pred['top80'])
 max_step=int(steps.max());active=max_step>=REGIME_START
 if active:
  recent={}
  for h in MEMORY_HORIZONS:
   use=steps>max_step-h;x=source.loc[use];yr=y[use]
   if int(use.sum())<1000:raise ValueError(f'Insufficient known labels in recent {h}-step window')
   r0=build_model().fit(x,yr);r1=build_model().fit(x,yr,sample_weight=np.where(yr==1,1.6,1.))
   recent[h]=.75*r0.predict_proba(target)[:,1]+.25*r1.predict_proba(target)[:,1]
   if model_dir is not None:
    r0.booster_.save_model(str(model_dir/f'recent{h}.txt'));r1.booster_.save_model(str(model_dir/f'recent{h}_class_weight_1p6.txt'))
  rank=lambda p: rankdata(p,method='average')/len(p)
  p10=.4*rank(p8)+.3*rank(recent[7])+.3*rank(recent[14])
 else:p10=p8.copy()
 diag=dict(regime_experts_active=active,known_through_step=max_step,recent7_rows=int((steps>max_step-7).sum()),recent14_rows=int((steps>max_step-14).sum()),expert_weight=EXPERT_WEIGHT,memory_horizons=list(MEMORY_HORIZONS))
 return p8,p10,diag

def run(data_path,out_dir,validate=False):
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);tr,te,sa=load_data(data_path);known=tr.label.notna().values;y=tr.label.values;s=tr.time_step.values
 assert tr.label.dropna().isin([0,1]).all() and s.max()<te.time_step.min()
 assert tr.txId.is_unique and te.txId.is_unique and not set(tr.txId)&set(te.txId)
 X=tr[FEATURES].astype('float32');Xt=te[FEATURES].astype('float32')
 assert np.isfinite(X.values).all() and np.isfinite(Xt.values).all()
 if validate:
  rows=[]
  for cut,end in FOLDS:
   if cut<REGIME_START:continue
   ti=np.flatnonzero(known&(s<=cut));vi=np.flatnonzero(known&(s>cut)&(s<=end))
   p8,p10,diag=predict(X.iloc[ti],y[ti].astype(int),X.iloc[vi],s[ti]);a8=roc_auc_score(y[vi],p8);a10=roc_auc_score(y[vi],p10)
   rows.append(dict(train_end=cut,valid_start=cut+1,valid_end=end,rows=len(vi),regime_experts_active=diag['regime_experts_active'],v8_auc=a8,v12_auc=a10,delta=a10-a8))
   write_csv(pd.DataFrame(dict(row=vi,step=s[vi],y=y[vi],v8=p8,v12=p10)),out/f'validation_predictions_cut{cut}.csv')
   write_csv(pd.DataFrame(rows),out/'validation_v12.csv');print('VALIDATION',rows[-1],flush=True)
 models=out/'models';models.mkdir(exist_ok=True);ti=np.flatnonzero(known);p8,p10,diag=predict(X.iloc[ti],y[ti].astype(int),Xt,s[ti],models)
 for name,p in [('submission_v12',p10),('v8_reproduced_for_audit',p8)]:
  sub=sa[['index']].merge(pd.DataFrame({'index':te['index'],'target':p}),on='index',how='left',sort=False,validate='one_to_one')
  assert len(sub)==len(te) and sub.target.notna().all() and sub.target.between(0,1).all() and np.array_equal(sub['index'],sa['index'])
  write_csv(sub,out/f'{name}.csv')
 meta=dict(recipe='when known training reaches step 24, rank-blend 40% V8 + 30% recent-7 model + 30% recent-14 model; otherwise V8',known_labels=int(known.sum()),unknown_labels_excluded=int((~known).sum()),prediction_rows=len(te),features=FEATURES,regime_start=REGIME_START,final_training_diagnostics=diag,future_labels_used=False,unknown_labels_pseudo_labeled=False,leaderboard_feedback_informed_model_selection=True,public_score_used_for_parameter_fitting=False,validation_windows_reused_and_overlapping=True,lightgbm_version=lgb.__version__,numpy_version=np.__version__,pandas_version=pd.__version__,submission_sha256=hashlib.sha256((out/'submission_v12.csv').read_bytes()).hexdigest(),v8_sha256=hashlib.sha256((out/'v8_reproduced_for_audit.csv').read_bytes()).hexdigest())
 (out/'run_metadata_v12.json').write_text(json.dumps(meta,indent=2));print('FINAL',meta,flush=True)
 return pd.read_csv(out/'submission_v12.csv')

def find_data():
 for root in [Path('/kaggle/input'),Path.cwd()/'data',Path.cwd()]:
  if root.exists():
   for p in sorted(root.rglob('train.csv')):
    if (p.parent/'test.csv').exists() and (p.parent/'sample_submission.csv').exists():return p.parent
 raise FileNotFoundError('Attach the official competition dataset or pass --data')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--out',default='BT4012_V12_run');ap.add_argument('--validate',action='store_true');a=ap.parse_args();run(a.data or find_data(),a.out,a.validate)
