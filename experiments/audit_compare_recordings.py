import csv,json,pathlib,numpy as np,hashlib
import argparse
ap=argparse.ArgumentParser();ap.add_argument('data_dir');ap.add_argument('output');ap.add_argument('--source-main-sha',default='25d4aaad29113cdae4e7cb5ad45fa48b0c7c4a98');args=ap.parse_args()
root=pathlib.Path(args.data_dir);results=[];rng=np.random.default_rng(20261115)
def pct(x):return np.percentile(x,[50,95,100]).tolist() if len(x) else []
def metrics(y,p):
 return {'rmse_cm':(100*np.sqrt(np.mean((p-y)**2,axis=0))).tolist(),'zero_motion_rmse_cm':(100*np.sqrt(np.mean(y*y,axis=0))).tolist(),'pearson':[float(np.corrcoef(p[:,k],y[:,k])[0,1]) if np.std(p[:,k])*np.std(y[:,k])>0 else None for k in range(3)],'predicted_to_true_rms':np.sqrt(np.sum(p*p,axis=0)/np.maximum(1e-15,np.sum(y*y,axis=0))).tolist()}
for jf in sorted(root.glob('*_cmp_*.json')):
 if jf.name.endswith('_audit.json'):continue
 base=str(jf)[:-5];j=json.load(open(jf));v=list(csv.DictReader(open(base+'_vision.csv')));h=list(csv.DictReader(open(base+'_holo.csv')));frames=list(csv.DictReader(open(base+'_frames.csv')));cam=json.load(open(base+'_camera_audit.json'))
 vt=np.array([float(x['t_cap']) for x in v]);lat=np.array([float(x['t_res'])-float(x['t_cap']) for x in v]);ht=np.array([float(x['t_ms']) for x in h]);hp=np.array([[float(x['shown_'+k]) for k in 'xyz'] for x in h]);ct=np.array([float(x['capture_time']) for x in frames]);present=sum(x['present']=='1' for x in v)
 q={'session':j['session'],'wav_seconds':j['wav']['seconds'],'vision_rows':len(v),'present':present,'present_pct':100*present/len(v),'vision_span_s':(vt[-1]-vt[0])/1000,'vision_fps':(len(v)-1)*1000/(vt[-1]-vt[0]),'vision_dt_ms_p50_p95_max':pct(np.diff(vt)),'processing_latency_ms_p50_p95_max':pct(lat),'holo_rows':len(h),'holo_dt_ms_p50_p95_max':pct(np.diff(ht)),'camera_callback_fps':(len(ct)-1)*1000/(ct[-1]-ct[0]) if len(ct)>1 else None,'camera_distance_cm_p50_p95_max':pct([100*x['distance'] for x in cam]),'outside45_present_pct':100*sum(x['distance']>.45 for x in cam)/len(cam) if cam else None,'scale':j['vision']['cam_model']['scale'],'holo':j['holo'],'wav_timing':j['wav'],'near_window_method':'fixed final stored camera scale; nearest logged pre-soft-limit display position within30ms; target250ms ±50ms; endpoints camera distance<=45cm; no minimum movement filter','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob(jf.stem+'*') if p.suffix in ['.csv','.json'] and not p.name.endswith('_audit.json')}}
 jump_indices=np.flatnonzero(np.linalg.norm(np.diff(hp,axis=0),axis=1)>.05)+1;jumps=ht[jump_indices];q['shown_jump_gt5cm']=[{'t_ms':float(ht[k]),'relative_s':float((ht[k]-ht[0])/1000),'step_cm':float(100*np.linalg.norm(hp[k]-hp[k-1]))} for k in jump_indices]
 paired=[]
 for c in cam:
  i=np.searchsorted(ht,c['t']);cand=[k for k in [i-1,i] if 0<=k<len(ht)];i=min(cand,key=lambda k:abs(ht[k]-c['t']))
  if abs(ht[i]-c['t'])<=30:paired.append({**c,'p':hp[i]})
 ws=[];allws=[]
 for i,a in enumerate(paired):
  end=np.searchsorted([x['t'] for x in paired],a['t']+250);cand=[k for k in [end-1,end] if i<k<len(paired)]
  if not cand:continue
  k=min(cand,key=lambda k:abs(paired[k]['t']-a['t']-250));b=paired[k]
  if not 200<=b['t']-a['t']<=300:continue
  z={'t':a['t'],'y':np.array(b['c'])-a['c'],'p':b['p']-a['p'],'crosses_jump':bool(np.any((jumps>a['t'])&(jumps<=b['t'])))};allws.append(z)
  if a['distance']<=.45 and b['distance']<=.45:ws.append(z)
 no_jump=[w for w in ws if not w['crosses_jump']];q['sensitivity_excluding_jump_windows']={'windows':len(no_jump),'excluded':len(ws)-len(no_jump),'threshold_cm':5,'note':'diagnostic only, not adjudicated recentering and not removed from primary results',**(metrics(np.array([w['y'] for w in no_jump]),np.array([w['p'] for w in no_jump])) if len(no_jump)>1 else {})}
 q['paired_present_frames']=len(paired);q['all_range_windows']=len(allws);q['in_range_windows']=len(ws)
 for label,windows in [('all_range',allws),('in_range',ws)]:
  if len(windows)<2:continue
  y=np.array([w['y'] for w in windows]);p=np.array([w['p'] for w in windows]);q[label]=metrics(y,p)
  if label=='in_range':
   blocks=np.floor((np.array([w['t'] for w in windows])-windows[0]['t'])/5000).astype(int);ids=np.unique(blocks);q['nonempty_5s_blocks']=len(ids)
   if len(ids)>1:
    ag=np.array([[sum(blocks==b),*np.sum((p[blocks==b]-y[blocks==b])**2,axis=0),*np.sum(y[blocks==b]**2,axis=0)] for b in ids]);dr=ag[rng.integers(len(ids),size=(2000,len(ids)))].sum(axis=1);rr=100*np.sqrt(dr[:,1:4]/dr[:,0,None]);zz=100*np.sqrt(dr[:,4:7]/dr[:,0,None]);q['within_session_rmse95_cm']=np.percentile(rr,[2.5,97.5],axis=0).T.tolist();q['within_session_rmse_minus_zero95_cm']=np.percentile(rr-zz,[2.5,97.5],axis=0).T.tolist()
 results.append(q)
for q in results:
 h=list(csv.DictReader(open(root/(q['session']+'_holo.csv'))));q['levelDb_p10_p50_p90']=np.percentile([float(r['levelDb']) for r in h if r['levelDb']],[10,50,90]).tolist();q['volume_ui']=json.load(open(root/(q['session']+'.json')))['volume_ui']
raw=[]
for f in sorted(root.glob('*_holo_*.json')):
 j=json.load(open(f));log=j['log'];raw.append({'session':j['session'],'seconds':j['seconds'],'build':j.get('build'),'yReal':j.get('yReal'),'yGain':j.get('yGain'),'tau':j.get('tau'),'log_rows':len(log),'log_widths':sorted(set(map(len,log))),'has_performance_clock':all(len(r)>=22 for r in log),'levelDb_median':float(np.median([r[8] for r in log if len(r)>8 and r[8] is not None])),'category':'sound-only recording; no hand truth, no prompted gesture accuracy'})
out={'source_main_sha':args.source_main_sha,'axes':'x left-right, y height, z depth','scope':'offline recorded display coordinates before soft limit; fixed final camera scale; not exact live panel; no WAV replay, no lag tuning, no fitting; overlapping windows and exploratory single-session block intervals','sessions':results,'holo_sessions':raw,'no_new_categorical_scores':True}
pathlib.Path(args.output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
for q in results:print(q['session'],q['wav_seconds'],'vis',round(q['present_pct'],1),'fps',round(q['vision_fps'],1),'rangeOut',round(q['outside45_present_pct'],1),'windows',q['in_range_windows'],'err',q.get('in_range',{}).get('rmse_cm'),'corr',q.get('in_range',{}).get('pearson'),'baseline',q.get('in_range',{}).get('zero_motion_rmse_cm'))
