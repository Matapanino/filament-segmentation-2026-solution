"""Run two actual GPU fits concurrently, bounded by a shared arm wall-time cap."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    assert Path("/kaggle/input").is_dir()
    started=time.monotonic();wave=json.loads(Path("wave.json").read_text())
    devices=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip().splitlines()
    assert len(devices)>=2,devices
    subprocess.run([sys.executable,"-u","prepare_train.py"],check=True,timeout=1200)
    jobs=wave["jobs"];assert len(jobs)==2
    children=[]
    for index,job in enumerate(jobs):
        config=Path(job["id"]+".json");config.write_text(json.dumps(job))
        stream=Path(job["id"]+".log").open("w",buffering=1)
        env={**os.environ,"CUDA_VISIBLE_DEVICES":str(index),"OMP_NUM_THREADS":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8"}
        child=subprocess.Popen([sys.executable,"-u","train_cuda.py",str(config)],stdout=stream,stderr=subprocess.STDOUT,env=env)
        children.append((job,child,stream))
    failure=None
    while any(p.poll() is None for _,p,_ in children):
        if time.monotonic()-started>wave["runtime_cap_seconds"]-300:
            failure="Declared arm runtime cap reached";break
        bad=[(j["id"],p.returncode) for j,p,_ in children if p.poll() not in (None,0)]
        if bad:failure="Worker failed: "+repr(bad);break
        gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu","--format=csv"],text=True).strip()
        print(json.dumps(dict(elapsed_seconds=time.monotonic()-started,gpu=gpu,processes={j["id"]:p.poll() for j,p,_ in children})),flush=True)
        time.sleep(30)
    if failure:
        for _,p,_ in children:
            if p.poll() is None:p.terminate()
    for _,p,stream in children:
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        stream.close()
    results={}
    for job,process,_ in children:
        path=Path("artifacts")/job["id"]/"result.json"
        if process.returncode==0 and path.exists():results[job["id"]]=json.loads(path.read_text())
    result=dict(status="paired_arm_complete" if len(results)==2 and not failure else "error",arm=wave["arm"],
                jobs=results,elapsed_seconds=time.monotonic()-started,error=failure,devices=devices)
    Path("result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result["status"]=="paired_arm_complete",failure


if __name__=="__main__":main()
