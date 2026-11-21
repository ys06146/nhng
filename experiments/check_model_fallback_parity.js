// Deterministic synthetic feed regression, not an accuracy or real-WAV test.
// Usage: node experiments/check_model_fallback_parity.js /path/to/old/bat3d.js /path/to/new/bat3d.js
const path=require('path'), O=require(path.resolve(process.argv[2])), N=require(path.resolve(process.argv[3]));
let seed=20261115;
const rnd=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296-.5};
const messages=[];
for(let b=0;b<600;b++){
 const steps=Array.from({length:4},(_,p)=>Float32Array.from({length:32},(_,i)=>{
  const t=b*32+i;return t%4000<2600?(0.00008*Math.sin(t*.013+p)+0.00001*rnd()):0;
 }));
 messages.push({steps,q:[2,3,2.5,3],act:[.8,.7,.9,.8],amp:[.01,.02,.006,.01],identical:0,mono:false,off:[0,0]});
}
const run=(B,model)=>{
 const frames=[],e=B.create({model,autoReset:false,onFrame:f=>frames.push(f.raw)});
 messages.forEach(m=>e.feed(m));return {frames,info:e.info()};
};
const parity={};
for(const model of ['learned','gated']){
 const a=run(O,model),b=run(N,model);
 if(a.frames.length!==b.frames.length)throw Error(model+' frame count changed');
 let max=0;
 for(let i=0;i<a.frames.length;i++)for(let k=0;k<3;k++)max=Math.max(max,Math.abs(a.frames[i][k]-b.frames[i][k]));
 parity[model]={frames:a.frames.length,max_raw_diff_m:max};
 if(max!==0)throw Error(model+' raw output changed');
}
const seq=run(N,'seqg');
if(!seq.frames.every(v=>v.every(Number.isFinite)))throw Error('Nonfinite seqg output');
console.log(JSON.stringify({scope:'synthetic feed regression only, not real-data accuracy',seed:20261115,parity,seqg:{frames:seq.frames.length,finite:true,default_model:N.create().info().model}},null,2));
