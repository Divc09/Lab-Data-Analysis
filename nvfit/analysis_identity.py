"""Versioned analysis descriptors shared by calculation, session and export records."""
from __future__ import annotations
import json
import math
import struct
from pathlib import Path


def canonical(value):
    if value is None: return 'null'
    if isinstance(value, bool): return 'true' if value else 'false'
    if isinstance(value, (int,float)):
        return 'null' if not math.isfinite(value) else 'n:' + struct.pack('>d',float(value or 0)).hex()
    if isinstance(value, str): return json.dumps(value,ensure_ascii=False,separators=(',',':'))
    if isinstance(value, (list,tuple)): return '[' + ','.join(canonical(v) for v in value) + ']'
    return '{' + ','.join(canonical(k)+':'+canonical(value[k]) for k in sorted(value)) + '}'


def source_path(path):
    value = str(path).replace('\\','/')
    return value.lower() if len(value)>1 and value[1]==':' else value


def analysis_descriptor(path, detector, mode, processing=None):
    return {'source':source_path(path),'detector':detector,'observable':mode,'processing':processing or {}}


def scientific_revision(value):
    if isinstance(value,dict): return {k:scientific_revision(v) for k,v in value.items() if k!='mtime_ns'}
    if isinstance(value,list): return [scientific_revision(v) for v in value]
    return value


def result_envelope(descriptor, revision, settings=None, dependencies=None, kind='fit'):
    identity = 'analysis:' + canonical(descriptor)
    context = {'analysis':descriptor,'revision':scientific_revision(revision),'dependencies':scientific_revision(dependencies or {}),'settings':settings or {},'kind':kind}
    return {'version':1,'id':identity,'kind':kind,'descriptor':descriptor,'source_revision':revision,
            'dependencies':dependencies or {},'settings':settings or {},'signature':canonical(context)}
