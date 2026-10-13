"""Train-only STKEC influence selection using damped Gauss-Newton inverse.

The released detector reads test labels. Here its query is a disjoint final
training block. The positive-semidefinite GGN curvature is an explicit stable
approximation to the released stochastic Hessian recursion, not JS drift.
"""
import numpy as np
import torch
from torch.func import functional_call, jvp
from benchmark_v2.data import split_origins


def conjugate_gradient(operator, rhs, iterations=40, tolerance=1e-3):
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rr = torch.dot(r, r)
    initial = float(rr.sqrt())
    if not initial:
        return x, {'iterations': 0, 'relative_residual': 0.}
    used = 0
    for used in range(1, iterations+1):
        ap = operator(p)
        denom = torch.dot(p, ap)
        if not torch.isfinite(denom) or denom <= 0:
            raise ValueError('Influence curvature must be finite positive definite')
        alpha = rr/denom
        x = x+alpha*p
        r = r-alpha*ap
        next_rr = torch.dot(r, r)
        if float(next_rr.sqrt()) <= tolerance*initial:
            rr = next_rr
            break
        p = r+(next_rr/rr)*p
        rr = next_rr
    residual = float((operator(x)-rhs).norm())/initial
    if not torch.isfinite(x).all() or residual > max(5*tolerance, .02):
        raise ValueError(f'Influence inverse solve did not converge: {residual}')
    return x, {'iterations': used, 'relative_residual': residual}


def select_influence(model, data, adj, changes, cfg, rng, tensor_batch):
    origins = split_origins(data, 'train')[::data['metadata']['output_length']]
    query_count = min(16, max(1, len(origins)//4))
    query_origins = origins[-query_count:]
    # A 12-slot exclusion gap keeps inputs and targets of the two pools separate.
    pool = origins[origins+data['metadata']['output_length'] < query_origins[0]-data['metadata']['input_length']]
    pool = pool[-int(cfg.get('influe_length', 168)):]
    if not len(pool):
        raise ValueError('Insufficient disjoint training blocks for influence')
    curvature_origins = pool[np.linspace(0, len(pool)-1, min(16, len(pool)), dtype=int)]
    saved_mode = model.training
    model.eval()
    try:
        params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        names, shapes, sizes = list(params), [p.shape for p in params.values()], [p.numel() for p in params.values()]
        def flatten(values):
            return torch.cat([v.reshape(-1) for v in values])
        def unpack(vector):
            return {k: v.reshape(s) for k, v, s in zip(names, vector.split(sizes), shapes)}
        scale = float(data['metadata']['scaler_std'])
        def batch(orig):
            return tensor_batch(data, orig, adj.device)[:3]
        curvature = [batch(curvature_origins[i:i+4]) for i in range(0, len(curvature_origins), 4)]
        query = [batch(query_origins[i:i+4]) for i in range(0, len(query_origins), 4)]
        def prediction(parameters, x):
            out = functional_call(model, parameters, (x, adj))
            return out[0] if isinstance(out, tuple) else out
        query_num = sum(int(mask.sum()) for _, _, mask in query)
        rhs = torch.zeros(sum(sizes), device=adj.device)
        for x, y, mask in query:
            residual = (prediction(params, x).reshape(y.shape)-y)/scale
            loss = residual[mask].square().sum()/max(query_num, 1)
            grads = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
            rhs += flatten([torch.zeros_like(p) if g is None else g for p, g in zip(params.values(), grads)])
        count = sum(int(mask.sum()) for _, _, mask in curvature)
        def operator(vector):
            out = float(cfg.get('influence_damping', .1))*vector
            direction = unpack(vector)
            for x, y, mask in curvature:
                def f(p):
                    return prediction(p, x).reshape(y.shape)/scale
                _, tangents = jvp(f, (params,), (direction,))
                pred = f(params)
                weights = torch.where(mask, tangents.detach(), torch.zeros_like(tangents))
                grads = torch.autograd.grad(pred, tuple(params.values()), grad_outputs=weights,
                                            allow_unused=True)
                out = out+2*flatten([torch.zeros_like(p) if g is None else g for p, g in zip(params.values(), grads)])/max(count, 1)
            return out.detach()
        solution, solve = conjugate_gradient(operator, rhs, int(cfg.get('influence_cg_steps', 80)))
        direction = unpack(solution)
        scores = torch.zeros(len(data['sensor_ids']), device=adj.device)
        counts = torch.zeros_like(scores)
        for i in range(0, len(pool), 4):
            x, y, mask = batch(pool[i:i+4])
            def node_losses(p):
                residual = (prediction(p, x).reshape(y.shape)-y)/scale
                return torch.where(mask, residual.square(), torch.zeros_like(residual)).sum((0, 2))
            _, derivative = jvp(node_losses, (params,), (direction,))
            scores += derivative.detach()
            counts += mask.sum((0, 2))
        benefit = scores/counts.clamp_min(1)
        candidates = [i for i in changes['retained_current'] if counts[i] > 0]
        ranked = sorted(candidates, key=lambda i: (-float(benefit[i]), int(data['sensor_ids'][i])))
        selected = set(map(int, changes['entered_current']))
        selected.update(ranked[:int(cfg.get('infl_topk', 10))])
        # Author main.py fills to 20%, using stable IDs and retaining isolated nodes.
        deficit = int(.2*len(benefit))-len(selected)
        if deficit > 0:
            remaining = sorted(set(range(len(benefit)))-selected)
            selected.update(rng.choice(remaining, deficit, replace=False).tolist())
        report = {'policy': 'train_only_damped_GGN_inverse_influence', 'uses_test': False,
                  'query_origins': query_origins.tolist(), 'pool_origins': pool.tolist(),
                  'curvature_origins': curvature_origins.tolist(), 'solve': solve,
                  'benefit_scores': [[int(data['sensor_ids'][i]), float(benefit[i])] for i in ranked],
                  'damping': cfg.get('influence_damping', .1),
                  'direction': 'descending gradient_dot_inverse_query_gradient; predicted reduction in query loss'}
        return np.array(sorted(selected), dtype=int), report
    finally:
        model.train(saved_mode)
