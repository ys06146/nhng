"""Audit a local snapshot. Uses the repository's replay_burst.py without editing ML/UI.
Run: python experiments/audit_current.py --data ml/data --ml ml --out experiments
SciPy is optional (Poisson CI and exact McNemar p value); Wilson intervals need no packages.
"""
import argparse, csv, json, math, statistics, sys
from pathlib import Path
from collections import Counter

def wilson(h,n):
 if not n:return (None,None)
 z=1.95996398454; p=h/n; d=1+z*z/n
 return ((p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d,(p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d)

def outcomes(fr,events):
 ans=[]
 for x in fr:
  if x['cue'] not in ('push','pull'):continue
  got=[(t,k) for t,k in events if x['t']<=t<=x['t']+1500]
  ans.append({'t':x['t'],'kind':x['cue'],'result':'miss' if not got else 'hit' if got[0][1]==x['cue'] else 'wrong','latency_ms':got[0][0]-x['t'] if got else None})
 return ans

def elapsed_rates(fr,events):
 # Only endpoint idle (not between-trial resets) and other. No fixed-FPS duration assumptions.
 non=[x['t'] for x in fr if x['label'] not in ('idle','skip')]
 if not non:return {}
 first,last=min(non),max(non)
 def block(x):
  if x['label']=='other':return 'other'
  if x['label']=='idle' and (x['t']<first or x['t']>last+2000):return 'idle'
  return None
 rates={}
 for lab in ['idle','other']:
  seconds=sum((b['t']-a['t'])/1000 for a,b in zip(fr,fr[1:]) if block(a)==lab and block(b)==lab and 0<b['t']-a['t']<=250)
  times={x['t'] for x in fr if block(x)==lab}
  count=sum(t in times for t,k in events) # all gesture kinds, including wave
  lo=hi=None
  try:
   from scipy.stats import chi2
   if seconds:
    lo=(chi2.ppf(.025,2*count)/2 if count else 0)*60/seconds
    hi=chi2.ppf(.975,2*(count+1))/2*60/seconds
  except ImportError:pass
  rates[lab]={'seconds':seconds,'events':count,'per_min':count*60/seconds if seconds else None,'poisson95':[lo,hi]}
 return rates

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--ml',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 sys.path.insert(0,str(Path(a.ml).resolve()));import replay_burst as rb
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);sessions=[];paired=[];details=[]
 for path in sorted(Path(a.data).glob('*.csv')):
  raw=list(csv.DictReader(path.open(encoding='utf-8-sig')))
  if not raw:continue
  meta_path=path.with_suffix('.json');m=json.loads(meta_path.read_text()) if meta_path.exists() else {}
  fr=rb.load(str(path));evt=[(x['t'],x['g']) for x in fr if x['g'] and not x['g'].endswith('_sim')]
  re=rb.burst_detect(fr);ro=outcomes(fr,re);live=outcomes(fr,evt)
  bad=0
  required=['t_ms','dUp_db','dDown_db','v_mps','snr_db','noise_db']+[s+str(i) for s in ['u','d'] for i in range(1,7)]
  for r in raw:
   for k in required:
    try:bad+=not math.isfinite(float(r[k]))
    except (ValueError,TypeError,KeyError):bad+=1
  status='complete' if m and m.get('aborted') is False else 'aborted' if m.get('aborted') else 'metadata_missing'
  counts=Counter(t['result'] for t in m.get('trials',[]) if t.get('kind') in ('push','pull'))
  n=sum(counts.values());ci=wilson(counts['hit'],n)
  main=status=='complete' and not m.get('swipeOnly') and n>0
  r={'file':path.name,'session':m.get('session',path.stem),'status':status,'main_live':main,'dist':m.get('dist'),'who':m.get('who'),'device':m.get('device'),'stereo':m.get('stereo'),'noise':m.get('noise'),'foreignTones':m.get('foreignTones'),'suspected_interference':'20261114_164412' in path.name,'rows':len(raw),'span_s':(fr[-1]['t']-fr[0]['t'])/1000,'numeric_bad_cells':bad,'simulated_events':sum(x['g'].endswith('_sim') for x in fr),'time_reversals':sum(b['t']<a['t'] for a,b in zip(fr,fr[1:])),'gaps_over_250ms':sum(b['t']-a['t']>250 for a,b in zip(fr,fr[1:])),'live_n':n,'live_hit':counts['hit'],'live_wrong':counts['wrong'],'live_miss':counts['miss'],'live_wilson95':ci,'byGesture':m.get('live',{}).get('byGesture',{}),'replay_n':len(ro),'replay_hit':sum(x['result']=='hit' for x in ro),'replay_wrong':sum(x['result']=='wrong' for x in ro),'replay_miss':sum(x['result']=='miss' for x in ro),'live_event_rates':elapsed_rates(fr,evt),'replay_event_rates':elapsed_rates(fr,re)}
  if main:
   for l,rr in zip(live,ro):paired.append((l['result']=='hit',rr['result']=='hit'))
  if m.get('device')=='n0c4' and not m.get('swipeOnly') and main:
   old=outcomes(fr,rb.burst_detect(fr,veto_snr=None));nv=outcomes(fr,rb.burst_detect(fr,veto_snr=-1e9)); r['old_veto_replay_hits']=sum(x['result']=='hit' for x in old);r['no_veto_replay_hits']=sum(x['result']=='hit' for x in nv)
   for cue in [x for x in fr if x['cue'] in ('push','pull')]:
    part=[x for x in fr if cue['t']<=x['t']<=cue['t']+1500]
    details.append({'session':m['session'],'cue':cue['cue'],'t':cue['t'],'max_abs_v':max(abs(x['v']) for x in part),'peak_db':max(max(x['up'],x['dn']) for x in part)})
  sessions.append(r)
 (out/'audit.json').write_text(json.dumps({'sessions':sessions,'paired_same_1500ms':paired,'n0c4_cue_features':details},indent=2,ensure_ascii=False))
 cols=['session','status','main_live','dist','who','device','stereo','noise','rows','span_s','numeric_bad_cells','simulated_events','time_reversals','gaps_over_250ms','live_n','live_hit','live_wrong','live_miss','replay_n','replay_hit','replay_wrong','replay_miss']
 with (out/'session_stats.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,cols,extrasaction='ignore');w.writeheader();w.writerows(sessions)
 lines=['# Bat Ears 실험 결과 — 강소연','', '분석 기준: main 595220058ff788c447b70c5449d2cf6b167c94ec, Drive nhng/ml/data 원본 사본. CSV/JSON은 같은 세션으로 한 번만 계산. WAV·fake 데이터는 제외.','', '## 실시간 밀기·당기기 (완료 JSON, swipeOnly 제외)','', '| 세션 | 기기 | 거리cm | stereo | 맞힘/시도 | 95% Wilson CI | 잘못된 제스처 | 미인식 |','|---|---|---|---|---|---|---|---|']
 for r in sessions:
  if not r['main_live']:continue
  n=r['live_n'];l,h=r['live_wilson95'];lines.append(f"| {r['session'][9:15]} | {r['device']} | {r['dist']} | {r['stereo']} | {r['live_hit']}/{n} ({100*r['live_hit']/n:.1f}%) | {100*l:.1f}–{100*h:.1f}% | {r['live_wrong']} | {r['live_miss']} |")
 for label,rs in [('전체 완료 세션',[r for r in sessions if r['main_live']]),('간섭 의심 세션 제외(민감도 분석)',[r for r in sessions if r['main_live'] and not r['suspected_interference']])]:
  n=sum(r['live_n'] for r in rs);hit=sum(r['live_hit'] for r in rs);wrong=sum(r['live_wrong'] for r in rs);miss=sum(r['live_miss'] for r in rs);lo,hi=wilson(hit,n)
  lines+=['',f'{label}: {hit}/{n} = {100*hit/n:.1f}%, Wilson 95% CI {100*lo:.1f}–{100*hi:.1f}%; 잘못된 제스처 {wrong}/{n}, 미인식 {miss}/{n}.']
  for kind in ['idle','other']:
   seconds=sum(r['live_event_rates'].get(kind,{}).get('seconds',0) for r in rs);ct=sum(r['live_event_rates'].get(kind,{}).get('events',0) for r in rs)
   lines.append(f'- {kind}: 기록된 이벤트 {ct}, 관찰 {seconds:.2f}초, {ct*60/seconds:.2f}/분.' if seconds else f'- {kind}: 측정 없음')
 b=sum(l and not r for l,r in paired);c=sum(r and not l for l,r in paired)
 lines+=['','## 같은 CSV 안내·동일 1.5초 창의 실시간 기록 vs 최신 replay','',f'라이브만 맞힘={b}, replay만 맞힘={c}. JSON 라이브 점수와 replay의 평가 창은 직접 섞지 않는다.']
 try:
  from scipy.stats import binomtest
  lines.append(f'정확 McNemar 검정 p={binomtest(min(b,c),b+c,.5).pvalue:.6g} (불일치 {b+c}개; 시행 내 의존·기기 군집을 반영하지 않은 탐색적 값).')
 except ImportError:pass
 lines+=['','## 주의','', '- live 표는 저장된 JSON 판정. wrong은 반대 방향뿐 아니라 다른 종류 제스처일 수 있음. 단순 오류 전체는 wrong+miss.','- Wilson CI는 시행이 독립인 이항 근사. 한 세션 내 반복이므로 참가자 모집단 신뢰구간으로 해석하지 않는다.','- 낮은 정확도 자체로 세션을 삭제하지 않는다. 간섭 의심 세션은 포함/제외를 모두 제시한다. report.py의 사전 표시를 인용했으며 간섭의 인과관계는 입증되지 않았다.','- 모든 메타데이터 소음 라벨이 quiet. 음악 대 조용함 비교 불가. 자동 라벨은 보정된 소음계 dBA가 아님.','- 사람 anon은 식별 불가. 기기 ID는 사람 ID가 아니므로 사람별 통계 대신 기기별로 보고한다.','- 35/50/65cm 계획과 달리 35/55cm 데이터가 존재. 실제 측정 거리 확인 전 파일을 재라벨하지 않는다.','- 0건의 오작동도 위험 0을 뜻하지 않음. 15초에서 0건이면 Poisson 95% 상한 약 14.8회/분.','- 지연은 cue→첫 감지(사람 반응 포함)로만 계산 가능. 실제 손 움직임 시작→감지 지연은 그래프/동기 영상으로 별도 측정.']
 lines += ['', '자세한 데이터 품질·ML 결과·통계 검정·개선 계획: [methods.md](methods.md).']
 (out/'results.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines))

if __name__=='__main__':main()
