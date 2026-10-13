TSAS fidelity repair, version 3 (development validation)
========================================================

Scope and acceptance
--------------------

The v2 data, scores, protocols, checkpoints and seven unaffected methods remain
immutable. This version repairs six implementations only. It first validates
PEMS05 2005 -> 2006 with seeds 47, 48, 49: 18 bounded development chains, two
epochs, four logical training batches per epoch and 32 chronological validation
and test origins. These measurements are not replacement paper scores.

Only after all mechanism tests, all 18 pilot receipts, checkpoint continuity,
and full-size resource validation succeed may a formal protocol be frozen.
The formal scope is six methods x (PEMS04 five seeds + PEMS05 three seeds),
48 complete annual chains. There is no automatic expansion to other districts,
gap tasks, or the KDD model. The allocation-side entry rejects formal work
until that gate is implemented and supplied with verified results.

Mechanisms and adaptations
-------------------------

* DCRNN: two layers of DCGRU encode each of 12 input times and autoregressively
  decode 12 outputs. Diffusion enters the reset/update and candidate gates.
  Separate row normalization of A and A.T precedes transpose support creation.
  K=2 directional Chebyshev recurrences each start from the original signal;
  this avoids the cross-support temporary reuse in the published TF loop.
  Inverse-sigmoid scheduled sampling uses k=1000 and observed training labels
  only. Decoder outputs/teacher inputs use the frozen train scaler; forecasts
  and training MSE are in original units. Learning rate .01 and clipping 5 are
  declared before development results, rather than inheriting .03 blindly.

* TGCN: self-loop symmetric-normalized graph convolution acts on x_t and h_t
  at all 12 times. Reset/update gates split the feature axis per node, avoiding
  the flattened node/feature split in the author's PyTorch cell. A shared
  linear head maps the final recurrent state to 12 horizons.

* ASTGCN: the pinned author's ASTGCN_r (recent-only, two blocks, K=3, 64 graph
  and temporal channels, time stride 1) includes sample-specific spatial and
  temporal attention, graph Chebyshev convolution, temporal convolution,
  residuals and layer normalization. Graph polynomials use matrix recurrence
  (the public utility uses elementwise multiplication for order >=2).
  The experiment must say recent-only; no daily/weekly branch is invented.

* PECPM: the full predictor matches the pinned author's forward numerically:
  trainable 64x12 memory, temporal embedding, two gated temporal convolutions,
  softmax retrieval and residual forecast. Missing external attention arrays
  are replaced explicitly with training-prefix K-means/cosine/top-5 labels.
  Previous centers initialize next-year consolidation when the center count
  agrees. Memory is fixed-width, as in the published predictor; no unsupported
  claim about a public dynamic memory expansion implementation is made.
  Daily observed-training Wasserstein drift selects 1% evolved / 9% replay
  nodes plus entrants, with the published 20%/15% cap and a one-node minimum
  for tiny graphs. Stable IDs replace index-prefix assumptions; isolated
  nodes are retained. EWC uses current-training masked Fisher, weight .001.
  These replacements must be reported as a benchmark adaptation.

* STRAP: pinned author STGNN backbone and full three-library implementation
  are included with unchanged class bodies. Keys and actual backbone values
  share historical sample indices. Spatial, temporal, spatiotemporal,
  topology and fusion paths are enabled. The library manager is checkpoint
  backed instead of relying on external files; evaluation cannot build a
  library from validation/test inputs. Build samples are the first 128
  chronological training origins. History ratio .3, history limit 10000,
  16 nearest patterns and fusion .7 are explicit.

* STKEC: the audited predictor and differentiable first-year cluster NLL are
  retained. The author detector is present under the extensionless path
  ``src/model/detect2``; its main path reads test labels. The repaired selector
  uses the last 16 nonoverlapping training query windows and an earlier,
  disjoint training pool. It solves a damped GGN inverse (damping .1, up to
  80 CG iterations) as an explicit stable influence approximation, reports
  residuals and rejects failed solves. Candidate scores are directional loss
  derivatives, not JS drift. It ranks predicted query-loss reduction, adds
  new sensors, and fills to the author's 20% with random retained candidates.
  Training-only queries and GGN curvature must be named in paper/reply.

Shared protocol
---------------

Version 2 observation masks, train-only scalers, chronological 60/20/20 splits,
12->12 windows, full-graph single-origin evaluation, metric accumulation and
stable sensor IDs are reused. Optimizer is fresh AdamW each year. Prediction
losses are weighted by observed target entries, while auxiliary losses are
weighted by valid node-windows. DCRNN shares one scheduled-sampling decision
across all memory chunks of a logical batch. Tiny graphs use the full logical
batch. Sparse/dense storage switches at 10% graph density without changing
edges; predictions and gradients are checked against both representations.
These changes preserve the published effective batch. Retrained models initialize every
year. PECPM, STRAP and STKEC inherit a SHA-bound prior-year checkpoint. Every
checkpoint is reconstructed and predictions must match exactly before test.

Source and compute provenance
-----------------------------

``provenance.json`` records pinned author commits and local port differences.
Every allocation verifies a source-tree digest, archive SHA, and all selected
NPZ hashes. CUDA, host, scheduler ID, seeds, elapsed time and peak allocation
are recorded. Local tests run through IONA; HPC trials use its durable
single-winner race gate. glab requires project CRUISE, NCI requires hn98 and
jobfs extraction, and the shared Katana account limit is eight GPUs (the
user's October 10 instruction supersedes the temporary seven-GPU limit).

No corrected score, canonical-method equivalence, or formal completion is
claimed on the strength of unit tests or a queued pilot alone.
