"""One allocated-GPU capacity probe on a specified full graph, not paper scores."""
import faulthandler
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from benchmark_v2.data import load_year, split_origins
from benchmark_v2.special import daily_clusters
from .manifests import make
from .runner import MODELS, normalized_adj, initialize, tensor_batch, snapshot, backward_logical_batch
from .influence import select_influence


def run(root, spec, data_root, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m = make(root, data_root, spec['method'], spec['seed'], out/'unused', district=spec['district'])
    year = int(spec['year'])
    data = load_year(Path(data_root)/spec['district']/str(year)/'data.npz')
    with np.load(Path(data_root)/spec['district']/str(year)/'graph.npz', allow_pickle=False) as z:
        assert np.array_equal(z['sensor_ids'], data['sensor_ids'])
        raw = torch.as_tensor(z['x'], dtype=torch.float32, device='cuda')
    adj = normalized_adj(raw.cpu().numpy(), 'cuda')
    cfg = dict(m['model_config'], year=year, begin_year=year, path=str(out),
               base_node_size=len(data['sensor_ids']), benchmark_strict=True, batch_size=m['batch_size'])
    args = SimpleNamespace(**cfg, logger=logging.getLogger('feasibility'))
    report = {'status': 'running', 'started': time.time(), 'source_sha256': spec['source_sha256'],
              'gpu': torch.cuda.get_device_name(), 'spec': spec, 'sensors': len(raw),
              'graph_density': float(torch.count_nonzero(raw))/raw.numel(), 'stages': {}}
    def event(stage, **values):
        report.update(stage=stage, **values)
        (out/'receipt.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'stage': stage, **values}), flush=True)
    # Keep an actionable stack if an author graph routine exceeds the allocation.
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    try:
        event('initialize')
        tic = time.time()
        model = MODELS[spec['method']](args).cuda()
        initialize(model, data, adj, raw, cfg, None, spec['seed'])
        torch.cuda.synchronize()
        report['stages']['initialization_seconds'] = time.time()-tic
        origins = split_origins(data, 'train')
        x, y, mask, _ = tensor_batch(data, origins[:1], 'cuda')
        model.eval()
        event('inference')
        tic = time.time()
        with torch.no_grad():
            output = model(x, adj)
            pred = output[0] if isinstance(output, tuple) else output
            if not torch.isfinite(pred).all():
                raise ValueError('Nonfinite full-graph predictions')
        torch.cuda.synchronize()
        report['stages']['single_origin_seconds'] = time.time()-tic
        micro = int(cfg['microbatch'])
        x, y, mask, _ = tensor_batch(data, origins[:micro], 'cuda')
        labels = None
        if spec['method'] == 'STKEC':
            labels, k = daily_clusters(data, cfg['cluster'])
            assert k == cfg['cluster']
        model.train()
        event('training_chunk', microbatch=micro)
        tic = time.time()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
        optimizer.zero_grad(set_to_none=True)
        backward_logical_batch(model, [(x, y, mask)], adj, cfg, labels)
        optimizer.step()
        checkpoint = snapshot(model)
        torch.cuda.synchronize()
        report['stages']['training_chunk_seconds'] = time.time()-tic
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        if spec['method'] == 'STKEC':
            event('influence')
            tic = time.time()
            changes = {'retained_current': np.arange(len(raw)), 'entered_current': np.array([], dtype=int)}
            _, influence = select_influence(model, data, adj, changes, cfg, np.random.default_rng(spec['seed']), tensor_batch)
            report['stages']['influence_seconds'] = time.time()-tic
            report['influence_solve'] = influence['solve']
        del checkpoint
        event('complete', status='passed')
    except BaseException as error:
        event(report['stage'], status='failed', error=repr(error))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        report.update(seconds=time.time()-report['started'], peak_gpu_gb=torch.cuda.max_memory_allocated()/1024**3)
        (out/'receipt.json').write_text(json.dumps(report, indent=2))
    return report
