"""Label availability for online calibration, in timestamp units rather than batches."""
class LabelDelayQueue:
    def __init__(self):
        self.pending=[]
        self.last_origin=None
    def advance(self,origin):
        if self.last_origin is not None and origin<=self.last_origin:
            raise ValueError('Online predictions must have strictly increasing origins')
        self.last_origin=origin
        ready=[payload for end,payload in self.pending if end<=origin]
        self.pending=[(end,payload) for end,payload in self.pending if end>origin]
        return ready
    def add_after_prediction(self,target_end,payload):
        if self.last_origin is None or target_end<=self.last_origin:
            raise ValueError('Enqueue only future targets, after issuing a prediction')
        self.pending.append((target_end,payload))
