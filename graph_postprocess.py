"""A small directional graph correction; no labels are read here."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.stats import rankdata


def predecessor_correction(edges, txids, scores, alpha=0.02):
    ids = pd.Index(np.asarray(txids))
    scores = np.asarray(scores, dtype=float)
    if not ids.is_unique or len(ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('IDs must be unique and aligned with finite scores')
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    src = ids.get_indexer(edges.iloc[:, 0])
    dst = ids.get_indexer(edges.iloc[:, 1])
    keep = (src >= 0) & (dst >= 0)
    # Row i contains predecessors j with an original edge j -> i.
    incoming = coo_matrix((np.ones(int(keep.sum())), (dst[keep], src[keep])),
                          shape=(len(ids), len(ids))).tocsr()
    incoming.data[:] = 1  # Duplicate edges must not duplicate a neighbor's vote.
    count = np.asarray(incoming.sum(axis=1)).ravel()
    own_rank = rankdata(scores, method='average') / len(scores)
    neighbor_mean = np.divide(incoming @ own_rank, count,
                              out=own_rank.copy(), where=count > 0)
    corrected = (1-alpha)*own_rank + alpha*neighbor_mean
    details = {
        'target_nodes': len(ids),
        'target_to_target_edges': int(incoming.nnz),
        'nodes_with_observed_predecessors': int((count > 0).sum()),
        'predecessor_coverage': float(np.mean(count > 0)),
        'alpha': alpha,
        'direction': 'predecessor',
        'no_predecessor_rule': 'retain own percentile rank',
    }
    return corrected, details
