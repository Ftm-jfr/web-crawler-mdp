"""Tests: VI and PI must agree, and the MDP must be well formed.

Run with:  python -m unittest discover -s tests -v   (or: pytest)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mdp_project as m  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CORPORA = ["corpus0", "corpus1"]


class TestMDP(unittest.TestCase):
    def test_transition_probabilities_sum_to_one(self):
        for c in CORPORA:
            pages, links, term = m.crawl_mdp(os.path.join(DATA, c))
            P, _, actions = m.build_mdp(pages, links, term)
            for s in pages:
                for a in actions[s]:
                    self.assertAlmostEqual(sum(P[s][a].values()), 1.0, places=9)

    def test_terminal_values_equal_reward(self):
        for c in CORPORA:
            pages, links, term = m.crawl_mdp(os.path.join(DATA, c))
            P, R, actions = m.build_mdp(pages, links, term)
            V, _ = m.value_iteration(pages, P, R, actions, term)
            for s in pages:
                if term[s] is not None:
                    self.assertEqual(V[s], term[s])


class TestAlgorithmsAgree(unittest.TestCase):
    def test_vi_and_pi_give_same_values_and_policy(self):
        for c in CORPORA:
            with self.subTest(corpus=c):
                V1, p1 = m.solve(os.path.join(DATA, c), "vi")
                V2, p2 = m.solve(os.path.join(DATA, c), "pi")
                self.assertEqual(p1, p2)
                for s in V1:
                    self.assertAlmostEqual(V1[s], V2[s], places=4)

    def test_known_values_corpus0(self):
        V, policy = m.solve(os.path.join(DATA, "corpus0"), "vi")
        self.assertAlmostEqual(V["1.html"], 0.645479, places=5)
        self.assertAlmostEqual(V["6.html"], 0.838521, places=5)
        self.assertEqual(policy["1.html"], "2.html")
        self.assertEqual(policy["6.html"], "9.html")


if __name__ == "__main__":
    unittest.main()
