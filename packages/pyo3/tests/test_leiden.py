import os
import graspologic_native as gpn
import unittest

sbm_graph = os.path.join("..", "network_partitions", "tests", "sbm_network.csv")
seed = 12345


def _get_edges():
  edges = []
  with open(sbm_graph, "r") as sbm_graph_io:
    for line in sbm_graph_io:
      source, target, weight = line.strip().split(",")
      edges.append((source, target, float(weight)))
  return edges


class TestLeiden(unittest.TestCase):
    def test_leiden(self):
        edges = _get_edges()
        modularity, partitions = gpn.leiden(edges, seed=seed)

    def test_reiterative_leiden(self):
        """
        Initially I had thought I could write a proper equality test for this but we won't be able to
        Each time we call leiden through graspologic native, we create a new XorgRandomShift PRNG.  If no
        seed is provided, it seeds itself.

        This state is discarded at the end of the leiden function, so seeding it the first time and not seeding it
        on any subsequent runs won't achieve the same behavior

        So instead we just test that the modularity and partitions produced where trials=10 is superior
        """
        edges = _get_edges()
        single_modularity, single_partitions = gpn.leiden(edges, seed=seed)

        repetitive_modularity, repetitive_partitions = gpn.leiden(edges, seed=seed, trials=10)
        self.assertTrue(single_modularity < repetitive_modularity)

