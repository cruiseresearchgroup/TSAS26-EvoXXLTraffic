"""Identity transitions, including departures and returning sensors."""
import numpy as np

def transition(previous_ids, current_ids, historical_ids):
    prev,cur,hist=list(previous_ids),list(current_ids),set(historical_ids)
    if len(set(prev))!=len(prev) or len(set(cur))!=len(cur):
        raise ValueError('Duplicate sensor IDs')
    if not set(prev)<=hist:
        raise ValueError('History must include the previous year')
    p={sid:i for i,sid in enumerate(prev)}
    current=set(cur)
    def idx(condition):return np.array([i for i,sid in enumerate(cur) if condition(sid)],dtype=np.int64)
    entered=idx(lambda sid:sid not in p)
    return dict(retained_current=idx(lambda sid:sid in p),
                retained_previous=np.array([p[sid] for sid in cur if sid in p],dtype=np.int64),
                entered_current=entered,
                first_seen_current=idx(lambda sid:sid not in hist),
                returning_current=idx(lambda sid:sid not in p and sid in hist),
                departed_previous=np.array([i for i,sid in enumerate(prev) if sid not in current],dtype=np.int64))
