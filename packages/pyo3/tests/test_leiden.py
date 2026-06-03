import os
import graspologic_native as gcn
import unittest

sbm_graph = os.path.join("..", "network_partitions", "tests", "sbm_network.csv")
simple_path = os.path.join("..", "network_partitions", "tests", "simple_org_graph.csv")
seed = 12345


def _get_edges(path):
  edges = []
  with open(path, "r") as sbm_graph_io:
    for line in sbm_graph_io:
      source, target, weight = line.strip().split(",")
      edges.append((source, target, float(weight)))
  return edges


class TestLeiden(unittest.TestCase):
    def test_leiden(self):
        edges = _get_edges(sbm_graph)
        modularity, partitions = gcn.leiden(edges=edges, seed=seed)

    def test_reiterative_leiden(self):
        """
        Initially I had thought I could write a proper equality test for this but we won't be able to
        Each time we call leiden through graspologic native, we create a new XorgRandomShift PRNG.  If no
        seed is provided, it seeds itself.

        This state is discarded at the end of the leiden function, so seeding it the first time and not seeding it
        on any subsequent runs won't achieve the same behavior

        So instead we just test that the modularity and partitions produced where trials=10 is superior
        """
        edges = _get_edges(sbm_graph)
        single_modularity, single_partitions = gcn.leiden(edges, seed=seed)

        repetitive_modularity, repetitive_partitions = gcn.leiden(edges, seed=seed, trials=10)
        self.assertTrue(single_modularity < repetitive_modularity)

    def test_provided_clusters(self):
        edges = _get_edges(simple_path)
        # this graph has two connected components, so first we'll try it with a reasonable clustering, and then we'll
        # try it with an invalid clustering
        communities = {
            "dwayne": 0,
            "nick": 0,
            "jon": 0,
            "carolyn": 0,
            "bryan": 0,
            "patrick": 0,
            "chris": 1,
            "david": 1,
            "amber": 1,
            "nathan": 1
        }

        gcn.leiden(edges, starting_communities=communities, seed=seed) # we just want to make sure it runs, not
        # inspect values

        # this is a bug we found, and we're testing for it
        communities["dwayne"] = 2
        communities["nathan"] = 2
        # these two have no edges, and shouldn't really be in the same community as per leiden, but they can
        # absolutely be put in there due to other reasons, so we should presume it's possible

        _, partitions = gcn.leiden(edges, starting_communities=communities, seed=seed)
        self.assertNotEqual(partitions["dwayne"], partitions["nathan"])


class TestLeidenCsr(unittest.TestCase):
    def test_leiden_csr_basic(self):
        """Test leiden_csr on a simple two-triangle graph connected by a weak bridge."""
        import numpy as np

        # Two triangles: {0,1,2} and {3,4,5} connected by weak edge 2-3
        # Build symmetric CSR for 6 nodes
        rows = []
        cols = []
        weights = []
        edges = [
            (0, 1, 10.0), (0, 2, 10.0),
            (1, 2, 10.0),
            (2, 3, 0.01),
            (3, 4, 10.0), (3, 5, 10.0),
            (4, 5, 10.0),
        ]
        for (r, c, w) in edges:
            rows.extend([r, c])
            cols.extend([c, r])
            weights.extend([w, w])

        n_nodes = 6
        # Build CSR from COO
        from scipy.sparse import csr_matrix
        mat = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

        indptr = mat.indptr.astype(np.int64)
        indices = mat.indices.astype(np.int32)
        data = mat.data.astype(np.float64)

        modularity, partitions = gcn.leiden_csr(
            indptr=indptr,
            indices=indices,
            data=data,
            n_nodes=n_nodes,
            resolution=1.0,
            randomness=0.01,
            iterations=2,
            use_modularity=True,
            seed=42,
            trials=1,

            max_local_moving_iterations=None,
        )

        # Should find 2 communities
        communities = set(partitions.values())
        self.assertEqual(len(communities), 2)
        # Nodes in same triangle should be in same community
        self.assertEqual(partitions[0], partitions[1])
        self.assertEqual(partitions[0], partitions[2])
        self.assertEqual(partitions[3], partitions[4])
        self.assertEqual(partitions[3], partitions[5])
        # The two triangles should be in different communities
        self.assertNotEqual(partitions[0], partitions[3])

    def test_leiden_csr_deterministic_with_seed(self):
        """Same seed should produce same results."""
        import numpy as np
        from scipy.sparse import csr_matrix

        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 0.1), (3, 4, 1.0), (4, 3, 1.0)]
        rows, cols, weights = [], [], []
        for (r, c, w) in edges:
            rows.extend([r, c])
            cols.extend([c, r])
            weights.extend([w, w])

        mat = csr_matrix((weights, (rows, cols)), shape=(5, 5))
        indptr = mat.indptr.astype(np.int64)
        indices = mat.indices.astype(np.int32)
        data = mat.data.astype(np.float64)

        kwargs = dict(
            indptr=indptr, indices=indices, data=data, n_nodes=5,
            resolution=1.0, randomness=0.01, iterations=2,
            use_modularity=True, seed=99, trials=1,
            max_local_moving_iterations=None,
        )
        _, p1 = gcn.leiden_csr(**kwargs)
        _, p2 = gcn.leiden_csr(**kwargs)
        self.assertEqual(p1, p2)

    def test_leiden_csr_trials_improves_or_equals(self):
        """Multiple trials should produce quality >= single trial."""
        import numpy as np
        from scipy.sparse import csr_matrix

        edges = [(0, 1, 5.0), (1, 2, 5.0), (2, 0, 5.0),
                 (3, 4, 5.0), (4, 5, 5.0), (5, 3, 5.0),
                 (2, 3, 0.1)]
        rows, cols, weights = [], [], []
        for (r, c, w) in edges:
            rows.extend([r, c])
            cols.extend([c, r])
            weights.extend([w, w])

        mat = csr_matrix((weights, (rows, cols)), shape=(6, 6))
        indptr = mat.indptr.astype(np.int64)
        indices = mat.indices.astype(np.int32)
        data = mat.data.astype(np.float64)

        kwargs = dict(
            indptr=indptr, indices=indices, data=data, n_nodes=6,
            resolution=1.0, randomness=0.01, iterations=2,
            use_modularity=True, seed=42,
            max_local_moving_iterations=None,
        )
        q1, _ = gcn.leiden_csr(trials=1, **kwargs)
        q5, _ = gcn.leiden_csr(trials=5, **kwargs)
        self.assertGreaterEqual(q5, q1)
