import csv,json,hashlib,pathlib,numpy as np
import argparse
ap=argparse.ArgumentParser();ap.add_argument('base');ap.add_argument('full_csv');args=ap.parse_args();p=pathlib.Path(args.base)
read=lambda f:list(csv.DictReader(open(f,encoding='utf-8')))
full=read(args.full_csv);live=read(str(p)+'_vision.csv');cb=read(str(p)+'_frames.csv')
ft=np.array([float(x['t_cap']) for x in full]);pts=np.array([float(x['pts_ms']) for x in full]);ct=np.array([float(x['capture_time']) for x in cb]);lt=np.array([float(x['t_cap']) for x in live]);idx=np.searchsorted(ft,lt);idx=np.clip(idx,1,len(ft)-1);idx=np.where(abs(ft[idx-1]-lt)<abs(ft[idx]-lt),idx-1,idx)
good=[];mismatch=0
for k,j in enumerate(idx):
 if abs(ft[j]-lt[k])>3 or live[k]['present']!='1' or full[j]['present']!='1':continue
 good.append(np.hypot((float(full[j]['palm_x'])-float(live[k]['palm_x']))*1280,(float(full[j]['palm_y'])-float(live[k]['palm_y']))*720));mismatch+=full[j]['hand']!=live[k]['hand']
matched=np.array([x['matched']=='1' for x in full]);offset=float(np.median(ft[matched]-pts[matched]));rawres=ft[matched]-(pts[matched]+offset)
fi=np.array([int(x['frame_idx']) for x in full]);present=sum(x['present']=='1' for x in full);result={'session':json.load(open(str(p)+'.json',encoding='utf-8'))['session'],'derived_not_new_session':True,'full_sha256':hashlib.sha256(pathlib.Path(args.full_csv).read_bytes()).hexdigest(),'full_rows':len(full),'full_present':present,'full_present_pct':100*present/len(full),'live_rows':len(live),'live_present':sum(x['present']=='1' for x in live),'span_s':(ft[-1]-ft[0])/1000,'mean_interval_ms':float(np.mean(np.diff(ft))),'dt_p50_p95_max_ms':np.percentile(np.diff(ft),[50,95,100]).tolist(),'nonpositive_dt':int(sum(np.diff(ft)<=0)),'frame_index_gaps':int(sum(np.diff(fi)!=1)),'matched_callback_rows':int(sum(matched)),'nearest_live_residual_ms_max':float(np.max(abs(ft[idx]-lt))),'both_present_pairs':len(good),'palm_difference_px_p50_p95':np.percentile(good,[50,95]).tolist(),'matched_hand_mismatch':mismatch,'capture_minus_pts_offset_ms':offset,'raw_pts_alignment_residual_ms_std':float(np.std(rawres)),'raw_pts_alignment_abs_residual_p95_max_ms':np.percentile(abs(rawres),[95,100]).tolist(),'blank_derived_columns':{k:sum(x[k]=='' for x in full) for k in ['t_res','v_right','v_up','v_screen','auto']},'warning':'Alignment uses callbacks and shared model-derived palm, not independent optical onset or audio hardware delay; raw WEBM not decoded in this audit.'}
result['live_within3ms']=int(sum(abs(ft[idx]-lt)<=3));result['live_before_full_first']=int(sum(lt<ft[0]));overlap=(lt>=ft[0])&(lt<=ft[-1]);result['overlap_nearest_residual_max_ms']=float(np.max(abs(ft[idx]-lt)[overlap])) if overlap.any() else None
print(json.dumps(result,indent=2))
