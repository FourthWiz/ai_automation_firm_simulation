# Phase 1 observation

The most striking result from `fig1_primary_lines` (default params, `q_a=1.2`) is that
`greedy_with_switching` — the supposedly more sophisticated strategy that explicitly
accounts for switching costs — finishes with *lower* cumulative profit (6400.54) than
the simpler `greedy_profit` (6401.88), a gap of ~1.34 units. The explanation lies in
the decision-vs-payment asymmetry: `greedy_with_switching` uses a smooth per-task
amortization of switching costs in its argmax decision rule, which causes it to switch
*slightly less aggressively* than `greedy_profit` in early periods — but the firm pays
the same lumpy hire/fire costs regardless of which strategy decided the switch. The
result is that the "cautious" strategy foregoes some early profit (by delaying
profitable mode transitions) without actually saving on adjustment costs paid.

A second surprise from `fig2_small_multiples_q_a` is that the top-4 strategy ranking
(`greedy_profit` > `greedy_with_switching` > `all_A` > `all_H`) holds identically
across all three `q_a` panels (0.8, 1.2, 1.6), despite the wide range in absolute
scale. Only `all_T` behaves qualitatively differently: deeply negative at `q_a=0.8`
(−88 cumulative profit, below even `all_H`'s 5100) and positive at `q_a=1.6` (2544),
yet it never cracks the top-4. This suggests the tipping point for `all_T` to
challenge the greedy strategies requires a higher `q_a` than 1.6 — a natural
parameter sweep for Phase 3.
