#!/usr/bin/env python
"""Metadata-only audit for NEMAR/OpenNeuro DS004194.

The script downloads only public metadata, event TSVs and channel TSVs. It
never requests the large iEEG payloads (.eeg/.dat/.mat).
"""
from __future__ import annotations
import argparse, csv, json, hashlib, subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
API="https://data.eegdash.org/api/eegdash/records"
NEMAR_ROOT="https://data.nemar.org/on004194/v1.0.0"

def get_json(url):
    with urlopen(Request(url,headers={"User-Agent":"neurocurate-ecog-audit/1.0"}),timeout=45) as r:
        return json.loads(r.read())

def get_text(url):
    import requests
    r=requests.get(url,headers={"User-Agent":"neurocurate-ecog-audit/1.0"},timeout=15)
    r.raise_for_status(); return r.text

def fetch_meta(r):
    import io
    sub=r.get("subject") or r.get("entities_mne",{}).get("subject"); task=r.get("task") or r.get("entities_mne",{}).get("task"); ses=r.get("session"); run=r.get("run")
    rel=r.get("bids_relpath") or r.get("bidspath",""); base=rel.rsplit("_ieeg.vhdr",1)[0]
    ev_url=f"{NEMAR_ROOT}/{base}_events.tsv"; ch_url=f"{NEMAR_ROOT}/{base}_channels.tsv"
    evrow={"subject":sub,"session":ses,"task":task,"run":run,"event_url":ev_url}
    chrow={"subject":sub,"session":ses,"task":task,"run":run,"channel_url":ch_url}
    try:
        ev=pd.read_csv(io.StringIO(get_text(ev_url)),sep="\t"); evrow.update(n_events=len(ev),n_trial_names=len(set(ev.get("trial_name",pd.Series(dtype=str)).dropna().astype(str))),trial_names=sorted(set(ev.get("trial_name",pd.Series(dtype=str)).dropna().astype(str))),trial_types=sorted(set(ev.get("trial_type",pd.Series(dtype=str)).dropna().astype(str))))
    except Exception as e: evrow.update(n_events=None,error=str(e))
    try:
        ch=pd.read_csv(io.StringIO(get_text(ch_url)),sep="\t"); bad=ch[ch.get("status",pd.Series(index=ch.index,dtype=str)).astype(str).str.lower().eq("bad")]; chrow.update(n_channels=len(ch),n_bad=len(bad),bad_channels=bad.get("name",pd.Series(dtype=str)).astype(str).tolist(),types=ch.get("type",pd.Series(dtype=str)).value_counts().to_dict())
    except Exception as e: chrow.update(error=str(e))
    return evrow,chrow

def git_commit():
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",default="results/ecog_iasl/phase0_audit"); ap.add_argument("--test-only",action="store_true"); args=ap.parse_args()
    out=Path(args.output_dir); out=out if out.is_absolute() else ROOT/out; out.mkdir(parents=True,exist_ok=True)
    # EEGDash query is metadata-only; records point to NEMAR and do not load data.
    import requests
    records=[]
    for skip in range(0,1000,1000):
        resp=requests.get(API,params={"filter":json.dumps({"dataset":"on004194"}),"limit":1000,"skip":skip},timeout=60); resp.raise_for_status(); page=resp.json().get("data",[]); records.extend(page)
        if len(page)<1000: break
    if not records: raise RuntimeError("EEGDash returned no DS004194 records")
    # Keep only canonical continuous VHDR records; data files are never fetched.
    vhdr=[r for r in records if r.get("extension")==".vhdr"]
    tasks=defaultdict(set); subjects=defaultdict(set); task_counts=defaultdict(int)
    events=[]; channels=[]
    def register(r):
        sub=r.get("subject") or r.get("entities_mne",{}).get("subject"); task=r.get("task") or r.get("entities_mne",{}).get("task"); ses=r.get("session"); run=r.get("run")
        tasks[task].add(sub); subjects[sub].add(task); task_counts[(sub,task)]+=1
    for r in vhdr: register(r)
    with ThreadPoolExecutor(max_workers=24) as pool:
        fs=[pool.submit(fetch_meta,r) for r in vhdr]
        for f in as_completed(fs):
            ev,ch=f.result(); events.append(ev); channels.append(ch)
    # Common trial-name intersections by task, using event metadata only.
    by_task=defaultdict(lambda:defaultdict(set))
    for e in events:
        if e.get("n_events") is None: continue
        by_task[e["task"]][e["subject"]].update(e.get("trial_names",[]))
    intersections={}
    for task,submap in by_task.items():
        if len(submap)>=1:
            inter=set.intersection(*submap.values()) if submap else set()
            intersections[task]={"subjects":sorted(submap),"n_subjects":len(submap),"common_trial_names":sorted(inter),"n_common_trial_names":len(inter)}
    manifest={"git_commit":git_commit(),"dataset":"DS004194 / on004194 Visual ECoG","source":"NEMAR v1.0.0; OpenNeuro ds004194 v3.0.0","license":"CC0","metadata_url":"https://data.nemar.org/on004194/metadata.json","dataset_api":"https://api.nemar.org/datasets/on004194","eegdash_api":API,"record_count_metadata":len(records),"vhdr_recordings":len(vhdr),"subjects":sorted(subjects),"tasks":{k:sorted(v) for k,v in tasks.items()},"download_policy":"metadata, events.tsv and channels.tsv only; no .eeg/.dat/.mat payload","raw_payload_downloaded":False,"dataset_size_bytes":62149898476,"dataset_size_human":"57.9 GB","source_records_sha256":hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with (out/"subject_task_matrix.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["subject","task","recording_count"])
        for s in sorted(subjects):
            for t in sorted(subjects[s]): w.writerow([s,t,task_counts[(s,t)]])
    with (out/"trial_counts.csv").open("w",newline="") as f:
        fields=["subject","session","task","run","n_events","n_trial_names","trial_types","error"]; w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(events)
    with (out/"channel_quality.csv").open("w",newline="") as f:
        fields=["subject","session","task","run","n_channels","n_bad","bad_channels","types","error"];w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(channels)
    (out/"stimulus_intersection.json").write_text(json.dumps(intersections,indent=2)+"\n")
    common=[(k,v) for k,v in intersections.items() if v["n_subjects"]>=4 and v["n_common_trial_names"]>0]
    status="E0_PASS" if common else "ECOG_NO_GO_DATA"
    lines=["# DS004194 ECoG metadata audit","",f"Status: **{status}**", "",f"- NEMAR version: v1.0.0; OpenNeuro source: ds004194 v3.0.0",f"- License: CC0",f"- Metadata records: {len(records)}; canonical VHDR recordings: {len(vhdr)}",f"- Dataset payload size: 57.9 GB (not downloaded)","- Downloaded during audit: only metadata, events.tsv and channels.tsv; no iEEG payload.","", "## Subject/task coverage", ""]
    for t,ss in sorted(tasks.items()): lines.append(f"- `{t}`: {len(ss)} subjects ({', '.join(sorted(ss))})")
    lines += ["", "## Common conditions", ""]
    for t,v in sorted(intersections.items()): lines.append(f"- `{t}`: {v['n_subjects']} subjects, {v['n_common_trial_names']} common trial names")
    lines += ["", "## E0 decision", "", "The audit does not treat all 14 participants as interchangeable. A common task is eligible only when at least four subjects share the task and event condition. Electrode indices are not directly comparable; channels.tsv provides local channel quality but no universal visual ROI label. Visual electrode masks require task-specific derivative coordinates or fold-local response selection.","", "No SAE, intervention or LOSO training was run before this audit."]
    (out/"audit_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":status,"records":len(records),"vhdr":len(vhdr),"tasks":{k:len(v) for k,v in tasks.items()},"output":str(out)},indent=2))

if __name__=="__main__": main()
