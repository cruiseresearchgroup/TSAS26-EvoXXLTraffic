import os, re, json, torch
import os.path as osp
import numpy as np
from Bio.Cluster import kcluster


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)


def _numeric_checkpoint(directory):
    """Finite numeric validation loss, rather than filename ordering."""
    candidates = []
    for filename in os.listdir(directory):
        if not filename.endswith('.pkl'):
            continue
        try:
            value = float(filename[:-4])
        except ValueError:
            continue
        if np.isfinite(value) and os.path.isfile(osp.join(directory, filename)):
            candidates.append((value, filename))
    if not candidates:
        raise ValueError('No finite numeric checkpoint in ' + directory)
    return osp.join(directory, min(candidates)[1])


def load_best_model(args):
    # The external first-year initialization applies only to the first transition.
    if args.load_first_year and args.year == args.begin_year + 1:
        load_path = args.first_year_model_path
    else:
        load_path = _numeric_checkpoint(osp.join(args.model_path, args.logname+'-'+str(args.seed), str(args.year-1)))
    loss = osp.basename(load_path)[:-4]

    args.logger.info("[*] load from {}".format(load_path))  # Recording Load Path
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]  # Loading the model state dictionary
    
    model = args.methods[args.method](args)  # Initialize the model
    
    if args.method == 'EAC':
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    
    if args.method == 'Universal' and args.use_eac == True:
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    
    model.load_state_dict(state_dict)  # Load the state dictionary into the model
    model = model.to(args.device)  # Move the model to the specified device
    return model, loss  # Returns the model and the minimum loss value


def load_test_best_model(args):
    if args.load_first_year and args.year == args.begin_year:
        load_path = args.first_year_model_path
    else:
        load_path = _numeric_checkpoint(osp.join(args.model_path, args.logname+'-'+str(args.seed), str(args.year)))
    loss = osp.basename(load_path)[:-4]

    args.logger.info("[*] load from {}".format(load_path))  # Recording Load Path
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]  # Loading the model state dictionary
    
    model = args.methods[args.method](args)  # Initialize the model
    
    if args.method == 'EAC':
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx + 1])
    
    if args.method == 'Universal' and args.use_eac == True:
        if args.year == args.begin_year:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx, _ in enumerate(range(args.year - args.begin_year)):
                model.expand_adaptive_params(args.graph_size_list[idx])
    
    model.load_state_dict(state_dict)  # Load the state dictionary into the model
    model = model.to(args.device)  # Move the model to the specified device
    return model, loss  # Returns the model and the minimum loss value



def long_term_pattern(args, long_pattern):
    # Guard against datasets whose begin_year has fewer nodes than args.cluster
    # (otherwise Bio.Cluster.kcluster raises "more clusters than items to be clustered").
    # Cap args.cluster in-place so that downstream STKEC `memory` tensor
    # (defined as [args.cluster, gcn.out_channel]) keeps a consistent shape.
    n_items = long_pattern.shape[0]
    if args.cluster > n_items:
        msg = "[long_term_pattern] cluster cap: requested {} > n_nodes {}, fallback to {}".format(args.cluster, n_items, n_items)
        if hasattr(args, "logger") and args.logger is not None:
            args.logger.warning(msg)
        else:
            print(msg)
        vars(args)["cluster"] = n_items
    attention, _, _ = kcluster(long_pattern, nclusters=args.cluster, dist='u')  # [number of nodes, average number of days per day] -> [number of nodes] ranges from 0 to k-1
    np_attention = np.zeros((len(attention), args.cluster))  # [number of nodes, clusters]
    for i in attention:
        np_attention[i][attention[i]] = 1.0
    return np_attention.astype(np.float32)


def get_max_columns(matrix):
    tensor = torch.tensor(matrix)
    max_columns, _ = torch.max(tensor, dim=1)
    return max_columns