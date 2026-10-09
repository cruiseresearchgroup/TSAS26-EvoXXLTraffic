"""Pooled sums/counts with an explicit observed-label mask.

MAE/RMSE include observed zero flows. MAPE has a separately reported denominator
and excludes |y| <= epsilon. Empty cohorts are missing, never zero error.
"""
import numpy as np

class MetricAccumulator:
    def __init__(self, mape_epsilon=1e-6):
        if mape_epsilon<0:raise ValueError('Negative MAPE threshold')
        self.epsilon=mape_epsilon
        self.n=self.mape_n=0
        self.a=self.s=self.p=0.0
    def update(self,prediction,target,observed):
        p,y,m=np.asarray(prediction),np.asarray(target),np.asarray(observed)
        if p.shape!=y.shape or m.shape!=y.shape or m.dtype!=np.bool_:
            raise ValueError('Aligned prediction, target and boolean mask required')
        if not np.isfinite(y[m]).all() or not np.isfinite(p[m]).all():
            raise ValueError('Nonfinite prediction/target on observed entries; do not silently discard')
        y=y[m].astype(np.float64);p=p[m].astype(np.float64);err=np.abs(y-p)
        self.n+=len(y);self.a+=float(err.sum());self.s+=float(np.square(err).sum())
        eligible=np.abs(y)>self.epsilon
        self.mape_n+=int(eligible.sum());self.p+=float((err[eligible]/np.abs(y[eligible])).sum())
    def result(self):
        return dict(MAE=self.a/self.n if self.n else None,
                    RMSE=(self.s/self.n)**.5 if self.n else None,
                    MAPE=100*self.p/self.mape_n if self.mape_n else None,
                    observed_count=self.n,mape_count=self.mape_n,mape_epsilon=self.epsilon)

def score(prediction,target,observed,mape_epsilon=1e-6):
    acc=MetricAccumulator(mape_epsilon);acc.update(prediction,target,observed);return acc.result()

def horizons(prediction,target,observed,steps=(3,6,12)):
    """Arrays (windows,horizon,nodes); cumulative 1..h plus a direct all-horizon score.

    The old Avg averaged repeated cumulative prefixes, overweighting short leads.
    V2 Avg pools each lead exactly once; historical Avg is not interchangeable.
    """
    p,y,m=map(np.asarray,(prediction,target,observed))
    if p.ndim!=3 or p.shape!=y.shape or p.shape!=m.shape:
        raise ValueError('Expected aligned (windows,horizon,nodes) arrays')
    if any(h<1 or h>y.shape[1] for h in steps):raise ValueError('Invalid horizon')
    out={str(h):score(p[:,:h],y[:,:h],m[:,:h]) for h in steps}
    out['Avg']=score(p,y,m)
    return out
