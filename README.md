# Web Crawling as a Markov Decision Process

Course project for **Fundamentals and Applications of Artificial Intelligence**
at the **University of Isfahan**.

A folder of HTML pages is modelled as a Markov Decision Process (MDP). Some
pages are *terminal* and carry a reward; the goal is to find, for every page,
which link to click so that the expected discounted return is maximal. The
problem is solved with two classic dynamic-programming algorithms:

- **Value Iteration (VI)**
- **Policy Iteration (PI)**

Both algorithms converge to the same optimal values and the same policy on the
provided corpora, which is verified by the test suite.

## MDP model

| Element | Definition |
|---|---|
| State | An HTML page (file name) |
| Action | Click one of the page's outgoing links |
| Terminal state | A page with `<meta name="reward" content="x">`; its value is `x` |
| Dead-end page | A non-terminal page without links; treated as absorbing with reward 0 |
| Transition | Chosen link: 0.6 · stay on page: 0.1 · other links share 0.3 equally. With a single link: 0.9 / 0.1 |
| Step reward | −0.05 for every regular transition |
| Discount | γ = 0.97 |
| Stopping threshold | θ = 1e-6 |

The parameters are constants at the top of `mdp_project.py`.

### Pipeline

1. `crawl_mdp` parses the HTML files (standard-library `html.parser`) and
   extracts outgoing links and terminal rewards.
2. `build_mdp` constructs the transition model `P`, rewards `R` and the action
   sets.
3. `value_iteration` / `policy_iteration` compute the optimal value function
   `V` and a greedy optimal policy.

Policy Iteration uses *iterative* policy evaluation (repeated sweeps until the
change is below θ) rather than solving the linear system exactly, which is
cheaper for larger state spaces.

## Usage

Requires Python 3.8+ and no third-party packages.

```bash
# Run both algorithms and compare them (default)
python mdp_project.py data/corpus0

# Run a single algorithm
python mdp_project.py data/corpus1 --algo vi
python mdp_project.py data/corpus1 --algo pi
```

### Example output (`data/corpus0`, excerpt)

```
=== Value Iteration ===
State Values (V):
  1.html               : 0.645479
  ...
  9.html               : 1.000000

Greedy Optimal Policy (click this link in each state):
  at 1.html               -> 2.html
  at 3.html               -> 6.html
  at 6.html               -> 9.html
  ...

Comparison: max |V_vi - V_pi| = 1.17e-08; policies identical: True
```

## Corpus format

A page is a plain HTML file. Links are ordinary anchors; terminal pages add a
reward meta tag:

```html
<a href="2.html">Go to 2</a>                 <!-- action -->
<meta name="reward" content="-0.5">          <!-- terminal page -->
```

Two sample corpora are included in `data/` (`corpus0`: 15 pages, `corpus1`:
16 pages). Links that point outside the corpus are ignored.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests check that transition probabilities sum to 1, that terminal values
equal their rewards, that VI and PI return the same policy and (almost) the
same values on both corpora, and that known reference values are reproduced.

## Project structure

```
.
├── mdp_project.py     # parsing, MDP construction, VI, PI, CLI
├── data/
│   ├── corpus0/       # sample corpus (15 pages)
│   └── corpus1/       # sample corpus (16 pages)
├── tests/
│   └── test_mdp.py
└── README.md
```

## Notes

- Actions are enumerated in sorted order so results are deterministic.
- In Policy Iteration, the current action is kept when another action is
  numerically tied with it, which prevents endless policy flip-flopping.
- Pages are scanned with Python's `html.parser`; only `<a href>` anchors and the
  `reward` meta tag are interpreted.

