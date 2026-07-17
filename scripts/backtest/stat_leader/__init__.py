"""Stat-leader backtest seam: the fair-value-to-tradeable bridge for the pts / reb
/ ast per-game NBA leader books. stat_samples produces the calibrated P(lead)
samples (with the CVaR pool) over the union field; stat_pricejoin resolves markets
to player_ids and attaches D+1 execution prices. Both are read-only against the
DB. The sizer / region / ledger wiring is the master's Phase 3."""
