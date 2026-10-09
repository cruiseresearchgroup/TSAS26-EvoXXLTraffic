import unittest
import numpy as np
from benchmark_v2.graph_identity_audit import distance_sigma
from benchmark_v2.graph_source import haversine_cross,validate_full_graph


class GraphAuditTests(unittest.TestCase):
    def test_blocked_sigma_and_identity_sensitive_reproduction(self):
        coords=np.array([[35.1,-120.1],[35.3,-120.4],[36.2,-121.1],[34.6,-119.1],[35.1,-120.1]])
        d=haversine_cross(coords,coords);reference=d[d>0].std()
        for block in [1,2,20]:self.assertAlmostEqual(distance_sigma(coords,block),reference,places=11)
        adj=np.exp(-d*d/reference**2);adj[adj<.1]=0;np.fill_diagonal(adj,0)
        self.assertTrue(validate_full_graph(adj.astype(np.float32),coords,distance_sigma(coords),2).passed)
        self.assertFalse(validate_full_graph(adj.astype(np.float32),coords[[1,0,2,3,4]],reference,2).passed)

    def test_coincident_catalog_rejected(self):
        with self.assertRaises(ValueError):distance_sigma(np.array([[35.,-120.],[35.,-120.]]))


if __name__=='__main__':unittest.main()
