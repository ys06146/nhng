const fs=require('fs'),path=require('path'),V=require((process.env.BAT_COMPARE_REPO||path.resolve(__dirname,'..'))+'/web/vision3d.browser.js');
const dir=process.argv[2];
for(const name of fs.readdirSync(dir).filter(f=>/_cmp_.*\.json$/.test(f)&&!f.endsWith('_camera_audit.json'))){
 const base=path.join(dir,name.slice(0,-5)),j=JSON.parse(fs.readFileSync(base+'.json')),c=j.vision.cam_model;
 const cam=V.camFromCalib({cam:{hfov:c.hfov,tilt:c.tilt,mirror:false},lagMs:c.lagMs},{W:j.camera.width,H:j.camera.height});cam.scale=c.scale;
 const frames=V.parseVisionCSV(fs.readFileSync(base+'_vision.csv','utf8'));
 const rows=frames.map(fr=>{const p=V.landmarksToLaptop(fr,cam),cc=V.handCam(fr,cam);if(!p||!cc)return null;return {t:fr.t,c:[0,1,2].map(k=>[0,5,9,13,17].reduce((s,i)=>s+p[i][k],0)/5),distance:Math.hypot(...cc)};}).filter(Boolean);
 fs.writeFileSync(base+'_camera_audit.json',JSON.stringify(rows));
}
