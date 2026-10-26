"""Offline evidence negative controls, using independent byte/count fixtures."""
import gzip
import hashlib
import json
from pathlib import Path
import pytest
from scripts import collect_g4s2a as c


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    raw=tmp_path/'raw';raw.mkdir()
    def put(name,value):
        p=raw/name;p.parent.mkdir(exist_ok=True,parents=True)
        p.write_text(json.dumps(value)+'\n');return p
    put('preregistered_protocol.json',{'source_files_sha256':{}})
    put('summary.json',{'status':'G4_S2A_PARTIAL','episodes':[{'seed':1001,'physics_steps_completed':1,
        'policy_steps_completed':1,'policy_inferences':1,'technical_pass':False}]})
    put('seed_1001/physics.jsonl',{'state':{'qpos':[.125,.25]}})
    put('seed_1001/policy.jsonl',{'step':1})
    put('seed_1001/inference.jsonl',{'event':'response'})
    for name in ('process','console','resource'):(tmp_path/name).write_text(name+' independent bytes\n')
    args=(raw,tmp_path/'output',tmp_path/'process',tmp_path/'console',tmp_path/'resource')
    remanifest(raw)
    return args


def remanifest(raw):
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    (raw/'artifact_manifest.json').write_text(json.dumps(manifest))


def test_gzip_is_lossless_and_existing_bundle_is_never_overwritten(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    original=(args[0]/'seed_1001/physics.jsonl').read_bytes()
    result=c.collect(*args)
    assert result['counts'][0]['physics_lines']==1
    assert gzip.decompress((args[1]/'seed_1001/physics.jsonl.gz').read_bytes())==original
    before=(args[1]/'collection.json').read_bytes()
    with pytest.raises(FileExistsError):c.collect(*args)
    assert (args[1]/'collection.json').read_bytes()==before


def test_tampered_source_fails_before_creating_bundle(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    (args[0]/'seed_1001/physics.jsonl').write_text('{"state":"tampered"}\n')
    with pytest.raises(ValueError,match='RAW_DIGEST'):c.collect(*args)
    assert not args[1].exists()


def test_authenticated_but_wrong_trace_count_is_rejected(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    (args[0]/'seed_1001/physics.jsonl').write_text('')
    remanifest(args[0])
    with pytest.raises(ValueError,match='TRACE_COUNT'):c.collect(*args)
    assert not args[1].exists()


def test_unregistered_seed_is_not_silently_collected(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    p=args[0]/'summary.json';summary=json.loads(p.read_text());summary['episodes'][0]['seed']=1002;p.write_text(json.dumps(summary))
    remanifest(args[0])
    with pytest.raises(ValueError,match='UNREGISTERED'):c.collect(*args)
    assert not args[1].exists()
