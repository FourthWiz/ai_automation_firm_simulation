# Phase 4 — Welfare Metrics, Sensitivity Sweeps, Writeup

**Weekends:** 8–9.
**Deliverables:** paper draft (~10 pages), polished post thread, public GitHub release.
**Status of model after phase 4:** done. Project shipped.

> Read `00_overview.md` and the prior phase docs first.

---

## 1. Phase goal

Layer welfare analysis on top of the phase-3 model. Run the final sensitivity sweeps. Write the post thread and the paper draft. This phase is more writing than coding — but the writing depends on having clean simulations from phase 3.

---

## 2. Welfare metrics

The whole point of the welfare layer is that there's no single right metric — different metrics tell different stories, and the gap between them is part of the finding. Compute multiple, compare.

All metrics are time-averages over the second half of the simulation (after transients have died down).

| Metric | Formula | What it captures |
|---|---|---|
| **Total surplus** | `Σ_F π_F + Σ_w wage_w · employed_w` | Aggregate efficiency |
| **Mean firm profit** | `mean(π_F)` over firms in run | Shareholder view |
| **Median worker income** | `median(income_w)` where `income_w = mean wage when employed × employment_rate_w` | Median-worker view |
| **Income Gini** | Gini coefficient of `income_w` | Inequality among workers |
| **Employment rate** | `mean(employed / K_total)` | Job availability |
| **Variance-penalized welfare** | `median income − λ · std(income)`, `λ = 0.5` | Risk-aware welfare |
| **Log utility** | `Σ_w log(income_w + ε)`, `ε = 0.01` | Diminishing marginal utility |
| **Worker turnover** | `mean(switches_per_worker)` | Disruption / churn |

**Why so many.** Total surplus says "automation usually wins" because automation lifts output. Median worker income says "augmentation often wins" because augment-strategies retain employment. The **gap** between them is the welfare wedge — and quantifying it is the paper's distinctive contribution.

---

## 3. Final sensitivity sweeps

Three sweeps for the paper's three figures.

**Sweep 1 — Strategy-region winners by `(c_auto, s)`.**
The phase-3 heatmap, polished. Re-run with 50 replicates per cell instead of 20 for clean medians. ~5 hours overnight.

Subplots:
- Panel A: by total surplus (winning archetype).
- Panel B: by median worker income (winning archetype).
- Panel C: difference between A and B (where do shareholders and workers disagree?).

**Sweep 2 — Welfare gap by `(c_auto, g)`.**
X = automation cost. Y = augmentation gain. Color = (profit-max strategy's worker income) − (worker-income-max strategy's worker income), normalized.

This isolates the "welfare wedge" finding. A blank/zero plot = no wedge. A strong gradient = strong finding.

10 × 10 grid, 30 replicates. ~3 hours.

**Sweep 3 — Layoff cascades by `(firing_aggression, λ_peer)`.**
X = mean firing aggressiveness across firms (`mean(θ_f[2])`). Y = peer signal weight `λ_peer`. Color = mean cascade size, defined as: when one firm fires `≥ 5%` of its workforce in a period, how many additional layoffs happen at other firms within the next 3 periods (above background rate).

Tests "Coordination failure" finding. Strong cascade where peer signal is high and firing aggressive = positive evidence.

10 × 10 grid, 30 replicates. ~3 hours.

---

## 4. Paper draft

Target: ~10 pages, working-paper format. Suitable for SSRN, arXiv `econ.GN`, or a workshop. **Do not target a top journal first** — get one round of public feedback, then polish.

### 4.1 Outline

**1. Introduction (~1.5 pages)**
- Hook: the AI revolution forces tech firms to choose between augmenting human labor and replacing it. What strategies win, and for whom?
- Key tension: shareholder value vs. worker welfare can diverge under automation.
- Three pre-committed candidate findings (state explicitly which were confirmed, partially confirmed, or contradicted).
- Contribution: ABM with task-based production + endogenous labor market with peer-driven reservation wages.

**2. Related work (~1 page)**
- Acemoglu-Restrepo task-based framework as the building block.
- Korinek, Stiglitz, Brynjolfsson on AI and inequality.
- Mortensen-Pissarides for labor matching intuition.
- ABM in economics: Tesfatsion, Dawid, Delli Gatti's *Macroeconomics from the Bottom-Up*.
- What's missing in the literature: explicit comparison of firm strategy archetypes inside one labor market with peer effects.

**3. Model (~3 pages)**
- 3.1 Tasks and production function.
- 3.2 Firm strategy parameterization (`θ_f`).
- 3.3 Workers and reservation wages.
- 3.4 Labor-market dynamics (hiring, firing, quitting).
- 3.5 Peer signals and the reservation-wage update.
- 3.6 Demand and bankruptcy.

Use the notation table from `00_overview.md`. State equations precisely. Promise: a reader following the equations could replicate the model.

**4. Simulation results (~3 pages)**
- 4.1 Baseline (single firm, two firm). Show phase-1 and phase-2 figures as warmup.
- 4.2 Sweep 1: strategy-region winners. The headline heatmap. One paragraph interpreting the boundary.
- 4.3 Sweep 2: welfare wedge. Quantify the gap.
- 4.4 Sweep 3: cascades. Show conditions under which firing decisions propagate.

**5. Discussion (~1 page)**
- What the simulations say about the three findings.
- Limitations (this is honest, not defensive — see §6 below).
- Extensions worth pursuing.

**6. Code and data availability**
Link to the GitHub repo. State: "All code, sweep results, and figures are reproducible from the repository at [URL] using `python -m firm_ai_abm.sweep`."

### 4.2 Limitations to acknowledge in the paper

State these directly in §5. Reviewers trust papers that acknowledge limits.

- Production function additivity (no task complementarity).
- No capital, no financial constraints.
- Worker skill is one-dimensional.
- Peer network static.
- Calibration illustrative, not estimated.
- Equilibrium concept informal — firms use heuristics rather than best-responding.
- One good, one demand curve.

These don't invalidate the findings — they sharpen them. The model's claims should be qualitative ("there exists a regime where..."), not quantitative point estimates.

### 4.3 What to NOT claim

- Don't claim policy implications. A stylized model can't inform policy. Phrase any policy-adjacent statements as "questions worth empirical investigation."
- Don't claim the augment/automate threshold is at any specific number. The threshold is a function of parameters that you've assumed; the *existence* of a threshold and its *direction of movement* with key parameters is what the model establishes.
- Don't call it a forecast.

### 4.4 Optional: an analytical limit

If you have time, derive one analytical result for credibility. Easiest: in the limit of infinite workers and zero switching cost, with homogeneous tasks (`α_i = α`, `β_i = β`), the augment-vs-automate boundary is at:

```
q_a · α = q_h · (1 + g · β) − c_aug + c_auto
```

This is the deterministic "cost-equalized" boundary. The simulation departs from it because of friction, peer effects, and heterogeneity — and quantifying the departure is part of the contribution. Including this gives reviewers a sanity-checkable benchmark.

---

## 5. Posting strategy

The paper is one output. The thread is the other.

| Post # | Timing | Content | Asset |
|---|---|---|---|
| 1 | After weekend 2 | "Three strategies for tech firms in the AI era — here's how their profits diverge over time" | Phase 1 line chart + small-multiples |
| 2 | After weekend 4 | "Watching workers move between firms under different AI strategies" | Phase 2 animation (GIF) + dual profit chart |
| 3 | After weekend 7 | "The map of who wins" | Phase 3 heatmap + skill-stratified subplots |
| 4 | After weekend 9 | Paper drop. Thread of findings. | All figures + paper PDF + GitHub link |

Each post links to the running GitHub repo. By post 4 you have a portfolio.

**Tone:** confident but humble. "Here's what the model says" not "here's what AI will do." Engage replies but don't argue with people who haven't read the post carefully.

**Where to post:** X/Twitter for visibility, LinkedIn for professional reach, a personal blog or Substack for the long-form companion piece. Cross-link.

---

## 6. Code release

Public GitHub repo. Cleaning up before release:

- [ ] Code in `firm_ai_abm/` package, importable.
- [ ] Notebooks in `notebooks/`, numbered, with markdown explanation cells (not just code).
- [ ] Sweep results in `results/sweeps/` as parquet. Document the schema in a README.
- [ ] Figures in `results/figures/` as PNG (300dpi) and SVG.
- [ ] Tests in `tests/` covering production function, mass conservation, key validations. `pytest` should run them.
- [ ] `pyproject.toml` with pinned dependencies (numpy, pandas, matplotlib, plotly, scipy).
- [ ] `README.md` with quick-start, reproduction instructions, link to paper.
- [ ] License: MIT.
- [ ] Citation file (`CITATION.cff`).

Reproduction instructions in README:
```bash
git clone <repo>
cd firm-ai-abm
pip install -e .
python -m firm_ai_abm.sweep --config sweep1.yml --out results/sweeps/sweep1.parquet
python -m firm_ai_abm.figures sweep1
```

---

## 7. Phase 4 schedule

**Weekend 8:**
- Day 1: implement welfare metrics module. Add to summary function. Add post-processing scripts that read sweep parquets and produce welfare summaries.
- Day 2: run sweep 2 (welfare gap) and sweep 3 (cascades). These run while you write — kick off, then come back. Polish sweep 1 figure.

**Weekend 9:**
- Day 1: write the paper. Have all figures finalized first, then the words flow around them. Aim for a complete first draft by end of day. Don't polish — get full skeleton with placeholder text where needed.
- Day 2: revise paper. Write the post thread. Clean up code for release. Push to GitHub. Post.

If you slip a weekend, the paper goes out a week later. That's fine. Don't release a half-baked paper to hit a self-imposed deadline.

---

## 8. Open design questions resolvable in phase 4

1. **Which welfare metric to feature most prominently.** **Recommendation: median worker income, with total surplus as a robustness check.** Or feature both side-by-side — that emphasizes the wedge finding.
2. **Whether to attempt the analytical limit.** Worth 2–4 hours if it works cleanly. Skip if it gets messy.
3. **Whether to include policy framings.** **Recommendation: skip.** Too easy to overclaim. A line at the end like "questions worth empirical investigation" is plenty.
4. **Pre-print or workshop-first?** **Recommendation: pre-print on SSRN + arXiv econ.GN.** Workshops involve gatekeeping that slows feedback.
5. **Is this a paper or a "research note"?** Honest answer at pet-project scale: probably a research note (~6–8 pages) rather than a full paper. Frame accordingly. The post thread is more visible than the paper anyway.

---

## 9. After phase 4 — possible extensions

If the project gathers interest and you want to keep going:

- **Strategy learning.** Firms update `θ_f` based on observed performance. Imitation, gradient ascent, or evolutionary updating.
- **Multi-good markets.** Different output goods, firms differentiate.
- **Skill investment by workers.** Workers retrain in response to job-market signals. Adds a beautiful loop.
- **Dynamic peer networks.** Workers re-form connections when switching firms.
- **Calibration to actual data.** Use BLS, Burning Glass, etc. to ground key parameters. Hard.
- **Policy interventions.** UBI, retraining subsidies, automation taxes — but only if you can model them with discipline.
- **Network of firms in a supply chain.** Firms produce intermediate goods for each other.

Each is a separate project. Don't bolt them onto the original.

---

## 10. Definition of "phase 4 done" (= project done)

- Welfare metrics implemented and validated.
- Three final sweeps run, results saved as parquet.
- Three final figures saved as PNG + SVG.
- Paper draft (~6–10 pages) written and proofread once.
- GitHub repo public, documented, license added.
- Post thread (4 posts) drafted and at least the first one published.
- One sentence describing the headline finding written down somewhere prominent.

The headline finding is what people will remember. Spend disproportionate time getting that sentence right.
