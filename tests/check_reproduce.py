"""Portable one-command asset resolution and summary checks; no inference required."""
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hohlraum.experiments.reproduce import prepare_assets, summarize
from hohlraum.experiments.run import policy_config

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); data=root/'bundle'; runtime=data/'runtime'
    (runtime/'model').mkdir(parents=True);(runtime/'model/config.json').write_text('{}')
    (runtime/'predictor').mkdir();(runtime/'predictor/model.pkl').write_bytes(b'fixture')
    profile=dict(server_args=dict(model_path='model'),sim_config=dict(predictor=dict(database_path='predictor/model.pkl')))
    (runtime/'hisim-profile.json').write_text(json.dumps(profile))
    (runtime/'checksums.json').write_text(json.dumps(dict(files=[dict(path='predictor/model.pkl',bytes=7,sha256=hashlib.sha256(b'fixture').hexdigest())])))
    for ds in ['playing','agentx']:
        p=data/'processed'/ds/'s123';p.mkdir(parents=True)
        (p/('calibration_rows.json' if ds=='playing' else 'selected_rows.json')).write_text('[]')
        if ds=='agentx':(p/'dependencies.json').write_text('{}')
        p=runtime/'priors'/ds;p.mkdir(parents=True)
        for name in [*[str(i)+'.txt' for i in range(6 if ds=='playing' else 4)],'branch-prior.txt','length_curves.txt','training.json']:(p/name).write_text('fixture')
    assets=prepare_assets(data,root/'resolved',['playing','agentx'],[123])
    for ds,path in assets.items():
        a=json.loads(path.read_text());assert a['priors']==str(runtime/'priors'/ds)
        assert Path(a['datasets'][ds]['123']['rows']).is_absolute()
        p=json.loads(Path(a['hisim_profile']).read_text())
        assert p['server_args']['model_path']==str(runtime/'model')
        assert 0<p['server_args']['nccl_port']<65536
    print('PASS: data/runtime resolution, separate dataset priors, no assets.local.json needed')
    (runtime/'predictor/model.pkl').write_bytes(b'corrupt')
    try:prepare_assets(data,root/'bad',['playing'],[123])
    except ValueError:pass
    else:raise AssertionError('corrupt runtime accepted')
    output=root/'results';p=output/'playing-s123-medium'/'budget640';p.mkdir(parents=True)
    (p/'result.json').write_text(json.dumps(dict(throughput=100,goodput=80,ttft_p90_s=5,write_bytes=640)))
    summarize(output)
    rows=list(csv.DictReader((output/'summary.csv').open()))
    assert len(rows)==1 and rows[0]['throughput']=='100' and rows[0]['goodput']=='80'
    print('PASS: runtime checksum failure and result CSV/JSON')
    protocol=json.loads((Path(__file__).resolve().parents[1]/'hohlraum/experiments/protocol.json').read_text())
    stage=policy_config('staging',protocol,root,'/tmp/test-library',30)
    admission=policy_config('admission-dynamic',protocol,root,'/tmp/test-library')
    assert stage['stage_hbm_leaves'] and stage['stage_hbm_dram_leaves']
    assert not admission.get('stage_hbm_leaves')
    print('PASS: WT read extension enabled for staging only')
