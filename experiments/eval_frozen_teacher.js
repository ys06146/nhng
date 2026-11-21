// Frozen-map displacement diagnostic. No fitting; reports whether the session is already in training.
// Models beyond the historical 5/7-session tracker require explicit BAT_EVAL_GATE and BAT_EVAL_TAU_S.
const fs=require('fs'),path=require('path'),ROOT=process.env.BAT_EVAL_REPO||path.resolve(__dirname,'..'),P=require(ROOT+'/tools/bravo/phase.js'),V=require(ROOT+'/tools/bravo/vision3d.js'),B=require(ROOT+'/web/bat3d.js');
const base=process.argv[2],J=JSON.parse(fs.readFileSync(base+'.json')),cal=B.DEFAULT_CALIB.galaxy;
const defaultModel=(B.create().info().model||'learned');
if(defaultModel!=='learned'&&process.env.BAT_EVAL_LINEAR_ONLY!=='1') throw new Error('This diagnostic evaluates only the fixed linear map, not the deployed non-linear/sequence model. Set BAT_EVAL_LINEAR_ONLY=1 only for an explicitly labeled linear-only diagnostic.');
const historical=/default \([57] sittings/.test(cal.session);
if (!historical && (!process.env.BAT_EVAL_GATE || !process.env.BAT_EVAL_TAU_S)) throw new Error('Explicit BAT_EVAL_GATE and BAT_EVAL_TAU_S required: match the source engine tracker, not historical defaults.');
const tracker={gate:Number(process.env.BAT_EVAL_GATE ?? 1.5),tauS:Number(process.env.BAT_EVAL_TAU_S ?? 0.25)};
if (!(tracker.gate>0&&tracker.tauS>0)) throw new Error('Positive gate/tauS required');
const inTraining=String(cal.note||'').includes(J.session.split('_').slice(0,2).join('_'));
const table=fs.readFileSync(base+'_vision.csv','utf8').trim().split(/\r?\n/).map(x=>x.split(','));const h=table.shift(),ix=Object.fromEntries(h.map((x,i)=>[x,i]));
const fr=table.filter(r=>r[ix.present]==='1').map(r=>({t:Number(r[ix.t_cap]),lm:Array.from({length:21},(_,i)=>['x','y','z'].map(k=>Number(r[ix[`lm${i}_${k}`]]))),wlm:Array.from({length:21},(_,i)=>['x','y','z'].map(k=>Number(r[ix[`wlm${i}_${k}`]])))}));
const lean=(J.screen_angle_deg??95)-90,ang=lean*Math.PI/180,C=[0,.19*Math.cos(ang),.10+.19*Math.sin(ang)];
const tr=fr.map(f=>{const c=V.handCam(f,{W:J.camera.width,H:J.camera.height,hfov:78,mirror:false,method:'pnp'});return c?{t:f.t,H:V.toLaptop(c,{tilt:lean,C}),distance:Math.hypot(...c)}:null}).filter(Boolean);
const wav=P.readWav(base+'.wav'),F=J.stereoFreqs,t0=J.wav.t0;const st=F.flatMap(f=>[0,1].map(c=>P.pathSteps(P.baseband(wav.ch[c],wav.rate,f),f,tracker)));
const rate=st[0].rate,cum=st.map(s=>{const q=new Float64Array(s.step.length+1);for(let i=0;i<s.step.length;i++)q[i+1]=q[i]+s.step[i];return q;});
const at=(c,t)=>cum[c][Math.max(0,Math.min(cum[c].length-1,Math.round((t-t0)/1000*rate)))];const ws=[];let rejected={gap:0,range:0,motion:0};
for(let i=0,j=0;i<tr.length;i++){
 while(j<tr.length&&tr[j].t<tr[i].t+250)j++;if(j>=tr.length)break;
 const a=tr[i],b=tr[j];if(b.t-a.t>330){rejected.gap++;continue;}
 if(a.distance>.45||b.distance>.45){rejected.range++;continue;}
 const y=b.H.map((v,k)=>v-a.H[k]);if(Math.hypot(...y)<.01||Math.hypot(...y)>.4){rejected.motion++;continue;}
 const x=cum.map((_,c)=>at(c,b.t)-at(c,a.t));const raw=cal.learned.laptop.W.map(w=>x.reduce((s,v,k)=>s+w[k]*v,0));ws.push({t:a.t,y,p:raw});
}
function stats(mode){const k=cal.amp.modes[mode].k,n=ws.length,e=[0,0,0],y2=[0,0,0],p2=[0,0,0],py=[0,0,0],ae=[0,0,0];for(const w of ws)for(let i=0;i<3;i++){const p=w.p[i]*k[i],y=w.y[i];e[i]+=(p-y)**2;ae[i]+=Math.abs(p-y);y2[i]+=y*y;p2[i]+=p*p;py[i]+=p*y;}return {rmse_cm:e.map(v=>100*Math.sqrt(v/n)),mae_cm:ae.map(v=>100*v/n),motion_cm:y2.map(v=>100*Math.sqrt(v/n)),uncentered_cosine:py.map((v,i)=>v/Math.sqrt(p2[i]*y2[i])),size:p2.map((v,i)=>Math.sqrt(v/y2[i]))};}
if(process.env.BAT_EVAL_WINDOWS)fs.writeFileSync(process.env.BAT_EVAL_WINDOWS,JSON.stringify(ws));
console.log(JSON.stringify({session:J.session,calibration:cal.session,evaluated_model:"learned-linear-only",deployed_default_model:defaultModel,training_session:inTraining,tracker,win_ms:250,camera:{hfov:78,tilt:lean,C},window_count:ws.length,rejected,range_limit_m:.45,wav:{rate:wav.rate,channels:wav.C,frames:wav.frames},stats:Object.fromEntries(['mse','half','matched'].map(m=>[m,stats(m)])),warning:(inTraining?'TRAINING SESSION, not holdout. ':'')+'One session, overlapping windows; fixed map feature-level displacement audit; no Kalman/display spring/filter; webcam pseudo-truth; no phone paths used.'},null,2));
