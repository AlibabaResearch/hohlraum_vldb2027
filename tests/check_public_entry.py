"""Verify native default routing and independence from simulator assets."""
import json, os, sys, tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hohlraum.experiments import reproduce as entry

def run():
    with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2,3','SGLANG_USE_CPU_ENGINE':'1','SGLANG_SIMULATOR_CONFIG_PATH':'stale'},clear=False):
        env=entry.runtime_environment()
        assert env['CUDA_VISIBLE_DEVICES']=='2,3'
        assert 'SGLANG_USE_CPU_ENGINE' not in env and 'SGLANG_SIMULATOR_CONFIG_PATH' not in env
        assert entry.runtime_environment('hisim')['CUDA_VISIBLE_DEVICES']==''
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);data=root/'data';model=root/'model';model.mkdir()
        (model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'fixture only, never loaded')
        for dataset,count in [('playing',6),('agentx',4)]:
            p=data/'processed'/dataset/'s123';p.mkdir(parents=True)
            (p/('calibration_rows.json' if dataset=='playing' else 'selected_rows.json')).write_text('[]')
            if dataset=='agentx':(p/'dependencies.json').write_text('{}')
            p=data/'runtime/priors'/dataset;p.mkdir(parents=True)
            for name in [*[str(i)+'.txt' for i in range(count)],'branch-prior.txt','length_curves.txt','training.json']:(p/name).write_text('{}')
        commands=[]
        # No predictor, simulator profile or checksum inventory exists here.
        args=['run.sh','--data',str(data),'--model',str(model),'--ssd',str(root/'ssd'),'--seeds','123','--loads','medium','--concurrency','35','--check-only','--output',str(root/'output')]
        with patch.object(sys,'argv',args),patch.object(entry.subprocess,'run',side_effect=lambda cmd,**kw:commands.append(cmd)):
            entry.main()
        assert len(commands)==2
        for cmd in commands:
            assert cmd[cmd.index('--backend')+1]=='sglang'
            assert cmd[cmd.index('--concurrency')+1]=='35'
            assets=json.loads(Path(cmd[cmd.index('--assets')+1]).read_text())
            assert 'hisim_profile' not in assets
            profile=json.loads(Path(assets['native_server_args']).read_text())
            assert profile['device']=='cuda' and profile['load_format']!='dummy' and profile['tp_size']==2
        result=root/'output/case/staging/client/result.json';result.parent.mkdir(parents=True)
        result.write_text(json.dumps(dict(total_input_tokens_s=10,slo10_input_tokens_s=9,ttft_p90_s=2)))
        entry.summarize(root/'output')
        rows=json.loads((root/'output/summary.json').read_text())
        assert rows[0]['case']=='case' and rows[0]['policy']=='staging' and rows[0]['throughput']==10 and rows[0]['goodput']==9
    print('PASS native default, CUDA visibility, no simulator assets, concurrency routing and native summaries')
if __name__=='__main__':run()
