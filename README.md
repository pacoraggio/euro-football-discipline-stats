# European Football Discipline Statistics

A data science project analysing disciplinary behaviour in European football, focusing on yellow and red cards across national leagues and international competitions.

The project is designed to be both analytical and pedagogical: it demonstrates how a real data analysis is conducted — starting from a research question, adapting to the data available, and drawing conclusions only from what can be trusted.

---

## Research Question

> Given a team, is the number of yellow (or red) cards it receives statistically different from other teams — and does this pattern persist across different competitions?

The analysis normalises cards by fouls committed to control for differences in playing style, and checks whether any anomalous behaviour identified in one league replicates across others.

---

## Project Structure

```
euro-football-discipline-stats/
│
├── notebooks/
│   ├── 00_research_framework.ipynb     # Scientific goals, methodology, known limitations
│   ├── 01_data_scraping.ipynb          # Data collection from multiple sources
│   ├── 02_eda_and_wrangling.ipynb      # Data quality checks, exploration, reshaping
│   ├── 03_single_season_test.ipynb     # Hypothesis test pipeline — one league, one season
│   ├── 04_multi_season_analysis.ipynb  # Season-by-season extension — Serie A (10 seasons)
│   ├── 05_cross_league_analysis.ipynb  # Cross-league comparison — PL, La Liga, Bundesliga
│   └── 06_forced_cards_analysis.ipynb  # Reverse direction: fouls received → cards forced
│
├── src/
│   ├── plots.py                        # Reusable plotting functions (EDA and results)
│   ├── hypothesis_tests.py             # Hypothesis test pipeline functions
│   ├── multi_season.py                 # Multi-season pooled analysis utilities
│   ├── discipline_pipeline.py          # Parameterised pipeline (forward & reverse direction)
│   └── scrapers/
│       ├── scraper_football_data_co_uk.py
│       ├── scraper_local_html.py
│       └── scraper_espn_data.py
│
├── data/
│   ├── raw/                            # Raw scraped data (not tracked by git)
│   └── processed/                      # Cleaned and reshaped data (tracked by git)
│
└── backup/                             # Draft notebooks and working files (not tracked by git)
```

---

## Data Sources

### National Leagues
Match-level data for 5 major European leagues over 15 seasons (2011/12 – 2025/26):
- Serie A (Italy)
- Premier League (England)
- La Liga (Spain)
- Bundesliga (Germany)
- Ligue 1 (France)

Source: [football-data.co.uk](https://www.football-data.co.uk/)

Each record contains per-match statistics for both teams: fouls, yellow cards, red cards, shots, corners, and match result.

### International Competitions
Season-aggregate data for UEFA club competitions (Champions League, Europa League, Europa Conference League).

Source: FBref / ESPN

---

## Notebooks

| Notebook | Description |
|---|---|
| `00_research_framework.ipynb` | Lays out the scientific goals, data structure, analytical approach, and known limitations before any code is written |
| `01_data_scraping.ipynb` | Collects raw data from multiple sources via web scraping |
| `02_eda_and_wrangling.ipynb` | Checks data quality (nulls, duplicates, outliers), explores distributions, and reshapes data into a team-oriented structure for analysis |
| `03_single_season_test.ipynb` | Step-by-step hypothesis test for one league and season: Bernoulli model, z-test, exact binomial, assumption checks, bootstrap, Mann-Whitney — applied to four teams (Juventus, Napoli, Inter, Udinese) |
| `04_multi_season_analysis.ipynb` | Season-by-season extension of the single-season pipeline across 10 Serie A seasons; identifies teams with persistent high/low card rates and applies pooled tests |
| `05_cross_league_analysis.ipynb` | Cross-league comparison extending the analysis to Premier League, La Liga, and Bundesliga; applies Z-screening, bootstrap, and Mann-Whitney per team and per season |
| `06_forced_cards_analysis.ipynb` | Reverse direction analysis: studies fouls received vs yellow cards forced by opponent, identifying teams that are unusually effective (or ineffective) at drawing bookings |

---

## Key Design Decisions

- **Cards are normalised by fouls** to separate disciplinary outcomes from playing style
- **Yellow and red cards are modelled separately** — red cards are rare count data with a very different distribution
- **Home/away venue is tracked** as a potential confounder (away teams tend to receive more cards)
- **Match result is included** as it may influence in-game disciplinary behaviour
- The cleaned, reshaped dataset (`team_matches.pkl`) is saved to `data/processed/` for use in analysis notebooks

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scipy
```

Notebooks were developed using Python 3.13 (Miniconda).