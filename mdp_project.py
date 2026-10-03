"""Web-page crawling as a Markov Decision Process (MDP).

Course project for "Fundamentals and Applications of Artificial Intelligence",
University of Isfahan.

A directory of HTML pages is interpreted as an MDP:

* state   - a page (file name)
* action  - clicking one of the page's outgoing links
* terminal states - pages carrying ``<meta name="reward" content="...">``

Two classic dynamic-programming algorithms compute the optimal values and a
greedy optimal policy: Value Iteration (VI) and Policy Iteration (PI).
"""

import argparse
import os
import sys
from html.parser import HTMLParser

# ──────────────────────────────────────────────────────────────────────────────
# MDP parameters
# ──────────────────────────────────────────────────────────────────────────────
GAMMA = 0.97             # discount factor
THRESHOLD = 1e-6         # convergence threshold
ACTION_PENALTY = -0.05   # reward for every non-terminal transition


# ──────────────────────────────────────────────────────────────────────────────
# 1) PARSE HTML CORPUS
# ──────────────────────────────────────────────────────────────────────────────
class _PageParser(HTMLParser):
    """Collect ``<a href>`` targets and the optional ``reward`` meta tag."""

    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.reward = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            href = attrs["href"].split("#", 1)[0]
            if href:
                self.hrefs.append(href)
        elif tag == "meta" and attrs.get("name") == "reward":
            try:
                self.reward = float(attrs.get("content"))
            except (TypeError, ValueError):
                pass


def crawl_mdp(directory):
    """Parse every ``*.html`` file in *directory*.

    Returns
    -------
    pages : set[str]
        File names of all pages.
    links : dict[str, set[str]]
        Outgoing links of each page (only links to pages inside the corpus).
    terminal_reward : dict[str, float | None]
        Reward of terminal pages, ``None`` for non-terminal pages.
    """
    pages = {fn for fn in os.listdir(directory) if fn.endswith(".html")}
    links, terminal_reward = {}, {}

    for fn in pages:
        with open(os.path.join(directory, fn), encoding="utf-8") as f:
            parser = _PageParser()
            parser.feed(f.read())
        links[fn] = {h for h in parser.hrefs if h in pages}
        terminal_reward[fn] = parser.reward

    return pages, links, terminal_reward


# ──────────────────────────────────────────────────────────────────────────────
# 2) BUILD TRANSITION / REWARD
# ──────────────────────────────────────────────────────────────────────────────
def build_mdp(pages, links, terminal_reward):
    """Build transition probabilities ``P``, rewards ``R`` and action lists.

    * Terminal and dead-end pages have a single no-op action ``None`` that
      loops back to the same state with probability 1 and reward 0.
    * On a regular page, clicking link ``a`` lands on ``a`` with probability
      0.6 (0.9 if it is the only link), stays on the page with probability 0.1
      (0.1 if it is the only link), and spreads the remaining 0.3 uniformly
      over the other links.
    * Every regular transition costs ``ACTION_PENALTY``.
    """
    P, R, actions = {}, {}, {}

    for s in pages:
        P[s], R[s], actions[s] = {}, {}, []
        out = links[s]

        if terminal_reward[s] is not None or not out:
            actions[s].append(None)
            P[s][None] = {s: 1.0}
            R[s][None] = {s: 0.0}
            continue

        for a in sorted(out):  # sorted -> deterministic tie-breaking
            actions[s].append(a)

            dist = {}
            others = sorted(out - {a})
            if len(out) == 1:
                dist[a], dist[s] = 0.90, 0.10
            else:
                dist[a], dist[s] = 0.60, 0.10
                share = 0.30 / len(others)
                for s2 in others:
                    dist[s2] = share
            P[s][a] = dist
            R[s][a] = {s2: ACTION_PENALTY for s2 in dist}

    return P, R, actions


# ──────────────────────────────────────────────────────────────────────────────
# 3) SOLVERS
# ──────────────────────────────────────────────────────────────────────────────
def _q_value(s, a, V, P, R, gamma):
    """Expected return of taking action *a* in state *s* (Bellman backup)."""
    return sum(p * (R[s][a][s2] + gamma * V[s2]) for s2, p in P[s][a].items())


def value_iteration(pages, P, R, actions, terminal_reward,
                    gamma=GAMMA, theta=THRESHOLD):
    """Value Iteration: repeat the Bellman optimality backup until convergence."""
    order = sorted(pages)
    V = {s: 0.0 for s in pages}
    policy = {}

    while True:
        delta = 0.0
        for s in order:
            if terminal_reward[s] is not None:
                V[s] = terminal_reward[s]
                policy[s] = None
                continue

            best_value, best_action = float("-inf"), None
            for a in actions[s]:
                q = _q_value(s, a, V, P, R, gamma)
                if q > best_value:
                    best_value, best_action = q, a

            delta = max(delta, abs(V[s] - best_value))
            V[s] = best_value
            policy[s] = best_action

        if delta < theta:
            break

    return V, policy


def policy_iteration(pages, P, R, actions, terminal_reward,
                     gamma=GAMMA, theta=THRESHOLD):
    """Policy Iteration: iterative policy evaluation + greedy improvement."""
    order = sorted(pages)
    V = {s: 0.0 for s in pages}
    policy = {}

    for s in order:
        if terminal_reward[s] is not None:
            policy[s] = None
            V[s] = terminal_reward[s]
        else:
            policy[s] = actions[s][0]

    while True:
        # Policy evaluation (iterative, approximate)
        while True:
            delta = 0.0
            for s in order:
                if terminal_reward[s] is not None:
                    continue
                new_v = _q_value(s, policy[s], V, P, R, gamma)
                delta = max(delta, abs(V[s] - new_v))
                V[s] = new_v
            if delta < theta:
                break

        # Policy improvement
        policy_stable = True
        for s in order:
            if terminal_reward[s] is not None:
                continue

            old_action = policy[s]
            best_value, best_action = float("-inf"), None
            for a in actions[s]:
                q = _q_value(s, a, V, P, R, gamma)
                if q > best_value + 1e-12:  # tolerance avoids float-tie flip-flopping
                    best_value, best_action = q, a
            # Keep the current action when it is (numerically) as good as the best.
            if abs(_q_value(s, old_action, V, P, R, gamma) - best_value) <= 1e-12:
                best_action = old_action

            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return V, policy


ALGORITHMS = {"vi": value_iteration, "pi": policy_iteration}


# ──────────────────────────────────────────────────────────────────────────────
# 4) DRIVER
# ──────────────────────────────────────────────────────────────────────────────
def solve(corpus_dir, algo="vi", gamma=GAMMA, theta=THRESHOLD):
    """Load *corpus_dir*, build the MDP and solve it with *algo* (vi | pi)."""
    pages, links, terminal_reward = crawl_mdp(corpus_dir)
    P, R, actions = build_mdp(pages, links, terminal_reward)
    return ALGORITHMS[algo](pages, P, R, actions, terminal_reward, gamma, theta)


def _natural_key(name):
    stem = name.rsplit(".", 1)[0]
    return (0, int(stem), name) if stem.isdigit() else (1, 0, name)


def print_result(title, V, policy):
    print(f"=== {title} ===")
    print("State Values (V):")
    for s in sorted(V, key=_natural_key):
        print(f"  {s:20s} : {V[s]:.6f}")
    print("\nGreedy Optimal Policy (click this link in each state):")
    for s in sorted(policy, key=_natural_key):
        print(f"  at {s:20s} -> {policy[s] or '-'}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Solve an HTML-crawling MDP with Value/Policy Iteration.")
    parser.add_argument("corpus_dir", help="directory containing *.html pages")
    parser.add_argument("--algo", choices=["vi", "pi", "both"], default="both",
                        help="algorithm to run (default: both, with comparison)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.corpus_dir):
        sys.exit(f"Error: '{args.corpus_dir}' is not a directory")

    names = {"vi": "Value Iteration", "pi": "Policy Iteration"}
    algos = ["vi", "pi"] if args.algo == "both" else [args.algo]
    results = {}
    for algo in algos:
        results[algo] = solve(args.corpus_dir, algo)
        print_result(names[algo], *results[algo])

    if len(results) == 2:
        (V1, p1), (V2, p2) = results["vi"], results["pi"]
        max_diff = max(abs(V1[s] - V2[s]) for s in V1)
        same = all(p1[s] == p2[s] for s in p1)
        print(f"Comparison: max |V_vi - V_pi| = {max_diff:.2e}; "
              f"policies identical: {same}")


if __name__ == "__main__":
    main()
