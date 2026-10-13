"""Pinned full STRAP with checkpoint-contained, read-only evaluation libraries."""
import copy
import torch
from .vendor.strap_model import RAP_Model


class MemoryLibraryManager:
    """Same library API as upstream; persistence is part of the run checkpoint."""
    def __init__(self):
        self.payloads = {}

    @property
    def metadata(self):
        return {str(y): [{}] for y in self.payloads}

    def update_library(self, year, libraries, params, source_counts):
        self.payloads[int(year)] = {
            'libraries': {k: {n: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                             for n, v in lib.items()} for k, lib in libraries.items()},
            'params': copy.deepcopy(params), 'source_counts': copy.deepcopy(source_counts)}

    def get_library_for_year(self, year, device):
        value = self.payloads.get(int(year))
        if value is None:
            return None
        value = copy.deepcopy(value)
        for lib in value['libraries'].values():
            for key, v in lib.items():
                if isinstance(v, torch.Tensor):
                    lib[key] = v.to(device)
        return value


class STRAP(RAP_Model):
    def __init__(self, args):
        super().__init__(args)
        self.strap.pattern_manager = MemoryLibraryManager()

    def get_extra_state(self):
        return {'year': self.current_year,
                'active_year': self.strap.current_year,
                'payloads': self.strap.pattern_manager.payloads}

    def set_extra_state(self, state):
        self.current_year = state['year']
        self.strap.pattern_manager.payloads = copy.deepcopy(state['payloads'])
        # Force reactivation: the upstream banks are nonpersistent buffers.
        self.strap.current_year = None
        if state['active_year'] is not None:
            if not self.strap.switch_to_year(state['active_year']):
                raise ValueError('Checkpoint omitted the active STRAP library')
