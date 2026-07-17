# Feature glossary (stat-leader arm)

One-line definitions for every base signal in `awards.db`, grouped by source
table. Season is the starting year (2022 = 2022-23). Where a definition is
inferred from the NBA stat convention rather than confirmed in code it is marked
(verify).

## Reading the names: two conventions that clash

Two families of "percentage" columns look alike and mean different things. This
is the single most common confusion.

- `X_pct` (`reb_pct`, `oreb_pct`, `dreb_pct`, `ast_pct`) is a **rate**: the share
  of *available opportunities* the player converted while on the floor. `reb_pct`
  = of the rebounds available while he played, the fraction he grabbed. `ast_pct`
  = of teammate field goals while he played, the fraction he assisted.
- `pct_X` (`pct_stl`, `pct_blk`, `pct_dreb`, `pct_ast`, `pct_pts`) is a **team
  share**: the player's count divided by his *team's* count. `pct_stl` = player
  steals / team steals. `pct_dreb` = player DREB / team DREB.
- `pct_pts_Y` (`pct_pts_paint`, `pct_pts_3pt`, `pct_pts_mr`, `pct_pts_ft`,
  `pct_pts_fb`) is a **self share**: of the player's *own* points, the fraction
  from that source. Different denominator again (his own points, not the team's).

So `reb_pct` (of available boards, how many he got) and `pct_dreb` (of the
team's DREB, his share) are unrelated denominators; do not treat them as the
same rebounding signal.

## Rolling-window suffixes (per-game and hustle/defence tables)

Each base signal below in `box_asof`, `hustle_asof`, `defend_asof` carries eight
as-of forms; the base name alone is the season-to-date view.

- `_std` season-to-date average as-of the snapshot (the level)
- `_l5 / _l10 / _l20 / _l30` trailing-N-game average as-of
- `_ema` exponentially weighted trailing average
- `_l10_vs_l30` recent-versus-medium momentum (l10 minus l30)
- `_l10_vs_std` recent-versus-season momentum (l10 minus season)

The `_l10_vs_*` forms are the momentum signals a leading indicator would lean on:
a rising `_l10_vs_std` says the player's recent form exceeds his season baseline.

## Box counting, per game (`stg_nba_player_game_logs`, 1996+)

Raw single-game box line. `points, rebounds, assists, steals, blocks,
turnovers, fga, fgm, fg3a, fg3m, fta, ftm, minutes` are the standard box counts.
`ts_pct` true-shooting % for the game. `opp_team_id` opponent. `usage_rate`
exists but is empty; use `usg_pct` from the advanced table.

## Per-game rolling (`stg_nba_box_asof`, 1996+)

As-of per-game averages with the eight rolling forms above.
- `ppg, rpg, apg, spg, bpg` points / rebounds / assists / steals / blocks per game
- `mpg` minutes per game
- `pra` points + rebounds + assists per game (combined)
- `fg_pct, fg3_pct, ft_pct` shooting percentages (fg3/ft partial in early seasons)
- `ts_pct` true shooting %; `efg_pct` effective FG% (weights threes 1.5x)

## Advanced / efficiency (`stg_nba_player_advanced_asof`, 1996+)

As-of season-to-date advanced rates.
- `off_rating, def_rating, net_rating` points produced / allowed per 100 possessions, and their difference
- `usg_pct` usage %: share of team possessions the player used (shot, turnover, or FT trip) while on floor
- `ts_pct` true shooting %
- `pace` team possessions per 48 while the player is on
- `pie` Player Impact Estimate: the player's share of the sum of box contributions across the game
- `dreb, stl, blk` per-game defensive rebounds / steals / blocks (advanced pull)
- `gp, min` games and minutes as-of
- `plus_minus, def_rim_fgm, def_rim_fga` present but empty here (rim defence lives in `asof_ext`)

## Role, share, shot profile (`stg_nba_player_asof_ext`, 1996+ core)

- `dd2, td3` count of double-doubles / triple-doubles as-of
- `pfd` personal fouls drawn per game
- `blka` own field-goal attempts blocked by opponents ("blocked attempts against")
- `poss` possessions played as-of
- `pct_pts` player points as a share of team points (usage-dashboard share)
- `pct_ast` player assists as a share of team assists
- `pct_stl, pct_blk, pct_dreb` player share of team steals / blocks / defensive rebounds
- `pct_pts_paint, pct_pts_3pt, pct_pts_mr, pct_pts_ft, pct_pts_fb` of the player's
  own points, share from paint / three / mid-range / free throws / fast break
- `pct_uast_fgm` share of the player's made field goals that were unassisted
- `ast_pct` assist %: teammate FGs assisted while on floor
- `ast_to` assist-to-turnover ratio
- `reb_pct, oreb_pct, dreb_pct` rebound % (total / offensive / defensive): share of
  available rebounds grabbed while on floor
- `pts_off_tov, pts_fb, pts_2nd_chance, pts_paint` points off turnovers / fast break /
  second chance / in the paint (counts)
- `opp_pts_paint, opp_pts_fb, opp_pts_2nd_chance, opp_pts_off_tov` opponent equivalents (on-court context)
- `def_ws` present but empty

### Tracking sub-block (same table, 2013+, ~49% fill)

- `potential_ast` potential assists per game: passes that would be assists if the shot were made
- `ast_pts_created` points created by the player's assists per game
- `secondary_ast` secondary ("hockey") assists per game: the pass before the assist
- `time_of_poss` minutes of ball possession per game
- `touches` touches per game
- `front_ct_touches` front-court touches per game
- `avg_sec_per_touch` average seconds held per touch
- `avg_drib_per_touch` average dribbles per touch

### Rim defence sub-block (same table, 2013+, ~48% fill)

- `def_rim_fga` opponent field-goal attempts at the rim the player defended, per game
- `def_rim_fgm` opponent rim makes allowed, per game
- `def_rim_pct` opponent FG% at the rim when this player defends
- `def_rim_freq` share of the player's defensive matchups that were at the rim

## Hustle (`stg_nba_hustle_asof`, 2016+, full fill; rolling forms)

- `defl` deflections per game (loose-ball-inducing touches of the ball on defence)
- `charge` charges drawn per game
- `cont2` contested two-point shots per game (defender within contest range)
- `cont3` contested three-point shots per game
- `conttot` total contested shots per game
- `dloose` loose balls recovered on defence per game
- `oloose` loose balls recovered on offence per game
- `scrast` screen assists per game: screens that directly free a scorer

## Defence, matchup (`stg_nba_defend_asof`, 2013+, full fill; rolling forms)

Opponent shooting when this player is the closest defender.
- `dpct_overall` opponent FG% defended, all shots
- `dfga_overall` opponent FGA defended per game, all shots (defensive volume)
- `dpct_fg2, dfga_fg2` opponent two-point FG% allowed and two-point FGA defended per game
- `dpct_fg3, dfga_fg3` opponent three-point FG% allowed and three-point FGA defended per game

## Availability (`stg_nba_availability_asof`, 1996+)

- `team_games_asof` team games elapsed while the player was rostered, as-of
- `games_played_asof, games_missed_asof` player games played / missed, as-of
- `availability_pct_asof` games played / team games (availability rate)
- `missed_last_10_team_games, missed_last_30_team_games` recent absences
- `current_absence_streak` consecutive team games currently missed
- `on_65_game_pace_flag` whether the player is tracking to clear the 65-game award qualifier

## Team (`stat_team_game`, `stat_qualifier`, `stat_team_fg_asof`, 1996+)

- `stat_team_game` per team-game: box aggregate, offensive and defensive possessions, per-48 pace
- `stat_qualifier` per team-season: `team_games` and `q = ceil(0.70 * team_games)`, the soft leader-market qualifier floor
- `stat_team_fg_asof` cumulative team FG makes / attempts (overall and three), the assist-conversion prior target

## Derived substrate (`stat_rate_counts_asof`, 1996+)

The as-of cumulative sufficient statistics the Monte Carlo consumes.
- `gp_played_asof, min_asof` cumulative games and minutes banked
- `used_fga, used_ft_trip (=0.44*FTA), used_tov` usage-possession numerators
- `fg3a, fg3m, fg2a, fg2m` three- and two-point attempts / makes; `fg2a_rim, fg2m_rim, fg2a_mid, fg2m_mid` the rim/mid split (proxy)
- `fta, ftm` free-throw attempts / makes
- `reb` cumulative rebounds; `ast` cumulative assists; `stl, blk` cumulative steals / blocks
- `potential_ast_asof` cumulative potential assists (2013+)
