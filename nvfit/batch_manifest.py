"""Atomic batch records and row commits; reports are views of committed work."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
import math
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone


def fingerprint(path):
    p=Path(path);before=p.stat();digest=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    after=p.stat()
    if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):raise ValueError(f'Source changed while recording: {p}')
    return {'sha256':digest.hexdigest(),'size':after.st_size,'mtime_ns':str(after.st_mtime_ns)}


def row_id(relative_path,detector):
    return json.dumps([str(relative_path).replace('\\','/'),detector],separators=(',',':'))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(json_value(value),indent=2,allow_nan=False),encoding='utf-8')
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def json_value(value):
    if hasattr(value,'tolist'):return json_value(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_value(v) for v in value]
    if isinstance(value,Path):return str(value)
    return value


def atomic_csv(path,fields,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    fields=list(dict.fromkeys([*fields,*(key for row in rows for key in row)]))
    try:
        with temporary.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(directory):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True)
    with (p/'.batch.lock').open('a+b') as stream:
        stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise ValueError('Another batch is writing this output directory.') from exc
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def load_manifest(path, *, validate_sources=True):
    record=json.loads(Path(path).read_text(encoding='utf-8'))
    if record.get('format')!='smartfitter-batch' or record.get('version')!=1:
        raise ValueError('Not a supported SmartFitter batch manifest.')
    if validate_sources:
        for source in record['sources']:
            current=fingerprint(source['path'])
            if current['sha256']!=source['revision']['sha256']:
                raise ValueError(f'Source changed since this run: {source["path"]}. Start a new batch to use changed data.')
    return record


class RunRecord:
    def __init__(self,output_dir,options,sources,rows,requested,*,parent=None,auxiliary=None,ui_options=None):
        self.output=Path(output_dir).resolve();self.id=uuid.uuid4().hex
        self.folder=self.output/'runs'/self.id;self.folder.mkdir(parents=True)
        self.path=self.folder/'manifest.json'
        old={r['row_id']:r for r in (parent or {}).get('rows',[])}
        self.data={'format':'smartfitter-batch','version':1,'run_id':self.id,'parent_run':(parent or {}).get('run_id'),
            'manifest_path':str(self.path),'started_utc':datetime.now(timezone.utc).isoformat(),'state':'running',
            'options':options,'ui_options':ui_options or (parent or {}).get('ui_options') or {},'auxiliary':auxiliary or {},
            'sources':sources,'requested_rows':list(requested),'completed_count':0,'rows':[],
            'software':{'python':sys.version.split()[0],'libraries':{name:version(name) for name in ('numpy','scipy','matplotlib','h5py')},'code_sha256':{name:fingerprint(Path(__file__).with_name(name))['sha256'] for name in ['pro_batch.py','fit_engine.py','fit_workflows.py','prepared_fit.py','ramsey.py']}}}
        for row in rows:
            previous=deepcopy(old.get(row['row_id'],{}))
            self.data['rows'].append({**previous,**row,'state':'pending' if row['row_id'] in requested else 'preserved' if previous.get('summary') else 'not_requested'})
        self.commit()

    def commit(self):
        atomic_json(self.path,self.data)
        atomic_json(self.output/'batch_manifest.json',self.data)

    def complete(self,key,result,summary,validation,*,origin=False):
        row=next(r for r in self.data['rows'] if r['row_id']==key)
        stem=row['output_stem'];model=result.get('selected_model','none')
        generated=['_result.json']
        if summary['status']!='FAIL' or model!='none':
            generated.append('_fit.png')
            if origin:generated.extend(['_origin_bundle.csv','_origin_bundle_meta.json'])
        outputs=[]
        for suffix in generated:
            path=self.output/(stem+suffix)
            if path.exists():
                target=self.folder/path.name;shutil.copy2(path,target)
                outputs.append({'path':str(target),'revision':fingerprint(target)})
        summary['figure_path']=str(self.folder/(stem+'_fit.png')) if '_fit.png' in generated else ''
        row.update(state='completed',summary=deepcopy(summary),validation=deepcopy(validation),outputs=outputs)
        self.data['completed_count']+=1
        self.commit()

    def finish(self,state,error=None):
        for row in self.data['rows']:
            if row['state']=='pending' and row.get('summary'):row['state']='preserved'
        self.data.update(state=state,error=error,finished_utc=datetime.now(timezone.utc).isoformat())
        self.commit()
