"""Six baseline fidelity repairs, reusing the frozen observation/metric protocol."""
import argparse
import copy
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from benchmark_v2.data import load_year, split_origins, window_batch
from benchmark_v2.identity import transition
from benchmark_v2.metrics import MetricAccumulator
from benchmark_v2.online import LabelDelayQueue
from benchmark_v2.calibration import FRPlusModule
from benchmark_v2.special import daily_clusters, cluster_loss, ewc_penalty
from benchmark_v2.runner import STKEC_Model
from .recurrent import DCRNN, TGCN
from .astgcn import ASTGCN
from .pecpm import PECPM, select_pecpm
from .strap import STRAP
from .influence import select_influence

MODELS = {'DCRNN': DCRNN, 'TGCN': TGCN, 'ASTGCN': ASTGCN,
          'PECPM': PECPM, 'STRAP': STRAP, 'STKEC': STKEC_Model}
CONTINUAL = {'PECPM', 'STRAP', 'STKEC'}


def snapshot(model):
    return {'model_state_dict': copy.deepcopy(model.state_dict())}


def restore(model, state):
    model.load_state_dict(state['model_state_dict'], strict=True)


def tensor_batch(data, origins, device, nodes=None):
    b = window_batch(data, origins)
    if nodes is None:
        nodes = np.arange(len(data['sensor_ids']))
    def convert(name):
        return torch.from_numpy(b[name][:, :, nodes].transpose(0, 2, 1).copy()).to(device)
    x, y, mask, available = [convert(k) for k in ('x', 'y', 'y_mask', 'x_available')]
    index = np.asarray(origins)[:, None]+np.arange(-x.shape[-1]+1, 1)
    stamps = data['timestamps'][index]
    # Stored PeMS grid is a synthetic UTC encoding of station local wall time.
    # These deterministic calendar features use input timestamps only.
    slot = (stamps % 86400)/86400
    weekday = ((stamps//86400+3) % 7)/7
    times = np.stack((slot, weekday), -1).astype('float32')
    times = np.broadcast_to(times[:, None, :, :], (*x.shape, 2)).copy()
    return SimpleNamespace(x=x.reshape(-1, x.shape[-1]),
                           x_available=available.reshape(-1, x.shape[-1]),
                           time_features=torch.from_numpy(times.reshape(-1, 24)).to(device)), y, mask, b

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def normalized_adj(raw, device):
    return torch.as_tensor(raw / (raw.sum(1, keepdims=True) + 1e-6),
                           dtype=torch.float32, device=device)

def node_context(raw_adj, centers, hops):
    chosen = set(map(int, centers))
    connected = (raw_adj != 0) | (raw_adj.T != 0)
    for _ in range(hops):
        if chosen:
            chosen |= set(np.flatnonzero(connected[list(chosen)].any(0)).tolist())
    nodes = np.array(sorted(chosen), dtype=np.int64)
    return nodes, np.isin(nodes, centers)

def masked_loss(pred, target, mask, kind='mse'):
    if not mask.any(): return None
    if not torch.isfinite(pred[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite prediction/target at an observed label')
    if kind == 'mse': return F.mse_loss(pred[mask], target[mask])
    if kind == 'huber': return F.smooth_l1_loss(pred[mask], target[mask])
    raise ValueError('Unsupported training loss: ' + kind)

def evaluate(model, data, adj, split, cohorts, *, cap=None, ttc=None, loss_nodes=None):
    """One origin at a time: no cross-origin attention/pattern lookahead.

    All cohorts share the same prediction, including TTC corrections. Validation
    and test leave the pre-evaluation model/pattern state unchanged.
    """
    saved = snapshot(model)
    was_training = model.training
    model.eval()
    H = data['metadata']['output_length']
    accum = {name: {str(h): MetricAccumulator() for h in range(1, H+1)} for name in cohorts}
    origins = split_origins(data, split)
    if cap is not None: origins = origins[:cap]
    calibration = optimizer = None
    queue = LabelDelayQueue()
    updates = []
    prediction_hash = hashlib.sha256()
    prediction_trace = []
    if ttc is not None:
        calibration = FRPlusModule(len(data['sensor_ids']), H//2+1, ttc['groups']).to(adj.device)
        optimizer = torch.optim.Adam(calibration.parameters(), lr=ttc['lr'])
    try:
        for origin in origins:
            x, y, mask, b = tensor_batch(data, [origin], adj.device)
            now = int(b['origin_time'][0])
            if calibration is not None:
                for base_old, target_old, mask_old, target_end in queue.advance(now):
                    calibration.train()
                    corr = calibration(base_old.unsqueeze(1)).squeeze(1)
                    loss = masked_loss(corr, target_old, mask_old)
                    if loss is not None:
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                        updates.append({'origin': now, 'label_end': target_end})
            with torch.no_grad():
                output = model(x, adj)
                pred = (output[0] if isinstance(output,tuple) else output).reshape(y.shape)
                base = pred.detach()
                if calibration is not None:
                    calibration.eval()
                    pred = calibration(base.unsqueeze(1)).squeeze(1)
                p, target, observed = pred.cpu().numpy(), y.cpu().numpy(), mask.cpu().numpy()
            prediction_hash.update(p.tobytes())
            prediction_trace.append(hashlib.sha256(p.tobytes()).hexdigest())
            if loss_nodes is not None: observed &= loss_nodes[None, :, None]
            for name, indices in cohorts.items():
                for h in range(1, H+1):
                    accum[name][str(h)].update(p[:, indices, :h], target[:, indices, :h], observed[:, indices, :h])
            if calibration is not None:
                end = int(b['target_times'][0, -1])
                queue.add_after_prediction(end, (base, y.detach(), mask.detach(), end))
    finally:
        restore(model, saved)
        model.train(was_training)
    result = {}
    for name, metrics in accum.items():
        prefixes = {h: a.result() for h, a in metrics.items()}
        # Preserve and label historical aggregation; expose pooled score separately.
        legacy = {k: float(np.mean([r[k] for r in prefixes.values()]))
                  if all(r[k] is not None for r in prefixes.values()) else None
                  for k in ('MAE', 'RMSE', 'MAPE')}
        result[name] = {'prefixes': {h: prefixes[h] for h in ('3', '6', '12') if h in prefixes},
                        'overall': prefixes[str(H)], 'legacy_prefix_mean': legacy,
                        'sensor_count': len(cohorts[name])}
    return {'cohorts': result, 'origins': len(origins), 'prediction_sha256': prediction_hash.hexdigest(),
            'prediction_trace': prediction_trace, 'calibration_updates': len(updates), 'calibration_timing': updates,
            'timing_valid': all(r['label_end'] <= r['origin'] for r in updates)}


def fisher_state(model, data, adj, batch_size, limit=None):
    model.eval()
    means = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    origins = split_origins(data, 'train')
    batches = 0
    for j in range(0, len(origins), batch_size):
        if limit is not None and batches >= limit:
            break
        x, y, mask, _ = tensor_batch(data, origins[j:j+batch_size], adj.device)
        output = model(x, adj)
        pred = (output[0] if isinstance(output, tuple) else output).reshape(y.shape)
        loss = masked_loss(pred, y, mask)
        if loss is None:
            continue
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach().square()
        batches += 1
    model.zero_grad(set_to_none=True)
    return means, fisher, batches


def initialize(model, data, adj, raw, cfg, previous_path, seed):
    if hasattr(model, 'prepare_graph'):
        model.prepare_graph(raw)
    if isinstance(model, DCRNN):
        model.output_mean.fill_(float(data['metadata']['scaler_mean']))
        model.output_std.fill_(float(data['metadata']['scaler_std']))
    if isinstance(model, STRAP):
        model.set_year(cfg['year'])
        origins = split_origins(data, 'train')[:int(cfg.get('pattern_build_samples', 128))]
        x, _, _, _ = tensor_batch(data, origins, adj.device)
        model.initialize_patterns(x, adj)
    if isinstance(model, PECPM):
        origins = split_origins(data, 'train')[::12]
        vectors = []
        for j in range(0, len(origins), 32):
            x, _, _, _ = tensor_batch(data, origins[j:j+32], adj.device)
            vectors.append(x.x[x.x_available.all(-1)].detach().cpu().numpy())
        model.fit_patterns(np.concatenate(vectors), seed)


def run(m):
    if m['method'] not in MODELS or m['purpose'] not in ('smoke', 'benchmark'):
        raise ValueError('Explicit v3 method/purpose required')
    years = [x['year'] for x in m['years']]
    if years != list(range(years[0], years[-1]+1)):
        raise ValueError('Year chain must be consecutive')
    if m['purpose'] == 'benchmark':
        if any(m.get(k) is not None for k in ('max_batches', 'max_eval_origins')):
            raise ValueError('No caps in formal experiments')
        if not m.get('protocol_frozen') or not m.get('validation_gate_sha256'):
            raise ValueError('Formal runs require a frozen protocol and passed validation gate')
    out = Path(m['output'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(m, indent=2))
    torch.set_num_threads(int(m.get('cpu_threads', 2)))
    device = torch.device(m['device'])
    seed = int(m['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    protocol = {'schema': 'tsas-fidelity-v3', 'method': m['method'], 'purpose': m['purpose'],
                'model_config': m['model_config'], 'epochs': m['epochs'], 'batch_size': m['batch_size'],
                'early_stopping_patience': m['early_stopping_patience'],
                'max_batches': m.get('max_batches'), 'max_eval_origins': m.get('max_eval_origins'),
                'source_sha256': m['source_sha256'], 'seed': seed,
                'evaluation': 'v2 masks, cumulative horizons, full graph, single chronological origin',
                'optimizer': 'AdamW, wd=0.01, one step per logical batch; masked-count accumulation',
                'specialist_adaptations': 'PECPM regenerated training-input cosine labels; STKEC training-query damped GGN influence'}
    ph = digest(protocol)
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    previous_path = None
    previous_ids, historical_ids, results = [], set(), []
    previous_data = None
    cluster_count = None
    rng = np.random.default_rng(seed)
    for entry in m['years']:
        tic = time.time()
        year = entry['year']
        for key in ('data', 'graph'):
            if sha(entry[key]) != entry[key+'_sha256']:
                raise ValueError('Frozen input changed: '+entry[key])
        data = load_year(entry['data'])
        if data['metadata']['provenance']['district'] != m['district']:
            raise ValueError('District mismatch')
        with np.load(entry['graph'], allow_pickle=False) as z:
            raw_adj, graph_ids = z['x'], z['sensor_ids']
            graph_provenance = json.loads(str(z['metadata']))
        ids = data['sensor_ids']
        if not np.array_equal(graph_ids, ids):
            raise ValueError('Graph/data catalog mismatch')
        if graph_provenance['order_status'] != 'verified_ids':
            raise ValueError('Only verified graph order is accepted')
        changes = transition(previous_ids, ids, historical_ids)
        empty = np.array([], dtype=int)
        cohorts = {'all': np.arange(len(ids)),
                   'new_to_previous': changes['entered_current'] if previous_path else empty,
                   'first_ever': changes['first_seen_current'] if previous_path else empty,
                   'returning': changes['returning_current'], 'retained': changes['retained_current']}
        cfg = dict(m['model_config'], year=year, begin_year=years[0], path=str(out),
                   benchmark_strict=True, base_node_size=len(ids), batch_size=m['batch_size'])
        cluster_labels = None
        if m['method'] == 'STKEC':
            if cluster_count is None:
                cluster_labels, cluster_count = daily_clusters(data, cfg['cluster'])
            cfg['cluster'] = cluster_count
        args = SimpleNamespace(**cfg, logger=logging.getLogger('tsas-v3'))
        model = MODELS[m['method']](args).to(device)
        parent = None
        if previous_path and m['method'] in CONTINUAL:
            saved = torch.load(previous_path, map_location=device, weights_only=False)
            expect = {'year': year-1, 'method': m['method'], 'seed': seed, 'protocol_hash': ph,
                      'data_hash': m['years'][len(results)-1]['data_sha256'],
                      'graph_hash': m['years'][len(results)-1]['graph_sha256']}
            if any(saved['metadata'].get(k) != v for k, v in expect.items()):
                raise ValueError('Cross-year checkpoint provenance mismatch')
            restore(model, saved)
            parent = {'path': str(previous_path), 'sha256': sha(previous_path), 'year': year-1}
        adj = normalized_adj(raw_adj, device)
        raw = torch.as_tensor(raw_adj, dtype=torch.float32, device=device)
        initialize(model, data, adj, raw, cfg, previous_path, seed)
        centers = np.arange(len(ids))
        selection = None
        if previous_path and m['method'] == 'PECPM':
            centers, selection = select_pecpm(previous_data, data, changes, rng)
        if previous_path and m['method'] == 'STKEC':
            centers, selection = select_influence(model, data, adj, changes, cfg, rng, tensor_batch)
        skip = not len(centers)
        nodes, center_mask = node_context(raw_adj, centers, cfg.get('num_hops', 2)) if not skip else (centers, np.array([], dtype=bool))
        if m['method'] not in ('STKEC', 'PECPM'):
            nodes, center_mask = np.arange(len(ids)), np.ones(len(ids), dtype=bool)
        train_data = dict(data)
        for key in ('x', 'y', 'observed', 'available'):
            train_data[key] = data[key][:, nodes]
        train_data['sensor_ids'] = ids[nodes]
        train_adj = normalized_adj(raw_adj[np.ix_(nodes, nodes)], device)
        means = fisher = None
        fisher_batches = 0
        if previous_path and cfg.get('ewc') and not skip:
            means, fisher, fisher_batches = fisher_state(model, data, adj, m['batch_size'], m.get('max_batches'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
        steps, stale, auxiliary = 0, 0, 0
        best, best_state, history = float('inf'), None, []
        microbatch = int(cfg.get('microbatch', m['batch_size']))
        print(json.dumps({'event': 'year_start', 'year': year, 'method': m['method'], 'parent': parent}), flush=True)
        if not skip:
            for epoch in range(m['epochs']):
                model.train()
                origins = rng.permutation(split_origins(data, 'train'))
                if m.get('max_batches') is not None:
                    origins = origins[:m['max_batches']*m['batch_size']]
                loss_sum, label_count = 0., 0
                for j in range(0, len(origins), m['batch_size']):
                    logical = origins[j:j+m['batch_size']]
                    batches = [tensor_batch(train_data, logical[k:k+microbatch], device)[:3]
                               for k in range(0, len(logical), microbatch)]
                    for _, _, mask in batches:
                        mask &= torch.as_tensor(center_mask, device=device)[None, :, None]
                    total = sum(int(mask.sum()) for _, _, mask in batches)
                    if not total:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    for x, y, mask in batches:
                        count = int(mask.sum())
                        if not count:
                            continue
                        output = model(x, train_adj, labels=y, label_mask=mask) if isinstance(model, DCRNN) else model(x, train_adj)
                        pred = (output[0] if isinstance(output, tuple) else output).reshape(y.shape)
                        loss = masked_loss(pred, y, mask)
                        if isinstance(model, PECPM):
                            loss = loss+float(cfg['pattern_weight'])*model.pattern_loss(x, output[1], mask)
                            auxiliary += 1
                        if cluster_labels is not None:
                            loss = loss+.1*cluster_loss(output[1], cluster_labels[nodes], mask)
                            auxiliary += 1
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite training loss')
                        (loss*(count/total)).backward()
                        loss_sum += float(loss.detach())*count
                        label_count += count
                    if means is not None:
                        penalty = ewc_penalty(model, means, fisher, cfg['ewc_lambda'], 'ewc')
                        penalty.backward()
                    if cfg.get('gradient_clip'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    if isinstance(model, DCRNN):
                        model.batches_seen += 1
                    steps += 1
                val = evaluate(model, data, adj, 'val', {'all': np.arange(len(ids))}, cap=m.get('max_eval_origins'))
                score = val['cohorts']['all']['overall']['MAE']
                history.append({'epoch': epoch, 'train_loss': loss_sum/label_count if label_count else None, 'validation_MAE': score})
                if score is not None and score < best:
                    best, best_state, stale = score, snapshot(model), 0
                else:
                    stale += 1
                print(json.dumps({'event': 'epoch_end', 'year': year, 'epoch': epoch+1, 'validation_MAE': score}), flush=True)
                if stale >= m['early_stopping_patience']:
                    break
            if best_state is None or not steps:
                raise ValueError('No valid training/validation observations')
            restore(model, best_state)
        state = snapshot(model)
        state['metadata'] = {'district': m['district'], 'year': year, 'method': m['method'], 'seed': seed,
                             'protocol_hash': ph, 'data_hash': entry['data_sha256'], 'graph_hash': entry['graph_sha256'],
                             'parent': parent, 'purpose': m['purpose'], 'source_sha256': m['source_sha256']}
        checkpoint = out/f'{year}.pkl'
        with checkpoint.open('xb') as f:
            torch.save(state, f)
        reloaded = MODELS[m['method']](args).to(device)
        restore(reloaded, torch.load(checkpoint, map_location=device, weights_only=False))
        if hasattr(reloaded, 'prepare_graph'):
            reloaded.prepare_graph(raw)
        # Compare prediction after full reconstruction before touching test labels.
        x, _, _, _ = tensor_batch(data, split_origins(data, 'val')[:1], device)
        model.eval()
        reloaded.eval()
        with torch.no_grad():
            a, b = model(x, adj), reloaded(x, adj)
            a = a[0] if isinstance(a, tuple) else a
            b = b[0] if isinstance(b, tuple) else b
            if not torch.equal(a, b):
                raise ValueError('Checkpoint prediction did not restore exactly')
        metrics = evaluate(reloaded, data, adj, 'test', cohorts, cap=m.get('max_eval_origins'))
        row = {'year': year, 'sensors': len(ids), 'parent': parent, 'selection': selection,
               'selected_sensor_ids': ids[centers].tolist(), 'optimizer_steps': steps,
               'fisher_batches': fisher_batches, 'auxiliary_steps': auxiliary, 'training_skipped': skip,
               'history': history, 'epochs_completed': len(history), 'metrics': metrics,
               'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
               'checkpoint_exact_reload': True, 'seconds': time.time()-tic,
               'peak_gpu_gb': torch.cuda.max_memory_allocated()/1024**3 if device.type == 'cuda' else None}
        if isinstance(model, STRAP):
            row['pattern_sources'] = model.strap.source_counts
            row['pattern_origin_years'] = {kind: sorted(set(lib['origin_year'].tolist()))
                for kind, lib in model.strap.pattern_manager.payloads[year]['libraries'].items()}
        results.append(row)
        (out/'results.json').write_text(json.dumps({'status': 'running', 'years': results}, indent=2))
        previous_path, previous_ids, previous_data = checkpoint, ids.tolist(), data
        historical_ids.update(previous_ids)
    final = {'status': 'passed', 'method': m['method'], 'purpose': m['purpose'], 'years': results,
             'source_sha256': m['source_sha256'], 'protocol_hash': ph}
    (out/'results.json').write_text(json.dumps(final, indent=2))
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.manifest).read_text()))
