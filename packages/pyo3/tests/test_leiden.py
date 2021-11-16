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
        modularity, partitions = gcn.leiden(edges, seed=seed)

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
