# move-alerts

> **Running on an always-on server?** See `SERVER_SETUP.md`. The server adds:
> - a live squawk listener
> - exact 15-minute timings
> - Telegram commands
> - holdings alerts
> - data-release alerts
> - CFTC positioning
> - cross-asset divergences
> - fade follow-ups and a scorecard
> - the Sunday week-ahead

Telegram alerts for large idiosyncratic moves in US large caps, with a Claude-written
read on *why* the stock moved and whether the move looks fadeable.

## What it does

1. **Scan** the S&P 500 (optionally filtered to `revolut_whitelist.txt`). Prices come
   from yfinance.
2. **Flag** names whose move clears all of these:
   - Idiosyncratic move ≥ 3σ: the move net of beta to the stock's sector SPDR ETF,
     scaled by 60-day residual vol.
   - Raw move ≥ 3%.
   - Price above $5.
   - At least $20m average daily dollar volume.
   - Capped at 10 names per run.
3. **Context**: pull the last 3 days of headlines and the earnings calendar (±7 days)
   from Finnhub.
4. **Explain**: Claude classifies the catalyst using only those headlines, with
   numbered citations. It labels the move as *information* or *flow* and gives a
   verdict: **fade candidate**, **don't fade** or **unclear**.
5. **Alert**: one Telegram message per new name, plus a closing summary after the
   US close.
6. **Log**: every alert is appended to `alerts_log.csv`. Running `evaluate.py` scores
   whether fade candidates actually reverted over 1, 5 and 20 days.

## Schedule

GitHub Actions runs on weekdays:

- **Intraday:** every 15 minutes from 09:45 to 15:45 New York time. A gate step
  skips runs outside that window, so US and UK clock changes need no edits.
- **End of day:** 21:35 UTC, after the close.

GitHub cron can start 5–20 minutes late at busy times, so expect alerts within
roughly 15–35 minutes of a move. The script exits cleanly on US holidays. A name is
alerted at most once per day.

**Actions minutes:** roughly 25 runs a day at 1–2 minutes each is about 1,000–1,300
minutes a month. That fits inside the 2,000 free minutes for private repos. To halve
it, change `*/15` to `*/30` in `.github/workflows/alerts.yml`.

## Macro alerts: USD, metals and bitcoin, 24/7 (`macro.py`)

This runs every hour, every day. It flags 1-hour or 4-hour moves of at least 3σ
against the instrument's own volatility, provided the move is also above a minimum
size.

Instruments covered:
- **USD:** DXY, EURUSD, USDJPY, USDCNH
- **Metals:** gold, silver, platinum and palladium futures
- **Bitcoin:** via Kraken

When gold futures are closed, including weekends, it watches **PAXG** on Kraken
instead. PAXG is a gold-backed token that trades 24/7.

Each alert has two parts:
- headlines from FinancialJuice and Finnhub
- a Claude assessment: the catalyst, whether other markets confirm the move,
  information vs flow, and a fade verdict

A repeat alert on the same instrument within 6 hours only fires if the move
extends further.

## Breaking news every 30 minutes (`news.py`)

This job runs in the same workflow as the macro alerts.

**Sources** (about 17 feeds plus Finnhub, all listed in `FEEDS`):
- FinancialJuice
- CNBC
- MarketWatch
- WSJ Markets
- Yahoo Finance
- Investing.com
- FXStreet
- CoinDesk
- Mining.com
- Fed, ECB and Bank of England press releases
- Google News searches, which surface Reuters, Bloomberg and FT headlines

**How it works:**
- Duplicate stories are merged across sources.
- Claude Haiku keeps only market-relevant stories and scores them 🔴 top-tier or 🟠 relevant.
- You get one digest per run, and only when something qualifies.

**Controls:**
- Set the repo variable or secret `NEWS_MIN_IMPORTANCE=3` for top-tier only.
- Run `python news.py --sources` to see which feeds respond.

The move alerts and the daily brief use the same source list.

## Daily brief at 07:00 UK, every day (`daily.py`)

- A hello, plus a health check that reads the last 20 runs of the other two jobs.
- Levels for USD, metals, BTC, equity futures, US 10-year and Brent.
- Today's high-impact economic calendar from ForexFactory, in UK time.
- S&P 500 names reporting earnings today.
- An overnight summary, the top headlines with links, and what to watch today.

If the brief doesn't arrive one morning, something is broken.

**Actions minutes:** with news every 30 minutes, usage is roughly 2,500–3,000 a month.
That's more than the 2,000 free minutes for a private repo. Either make the repo
public, which gives unlimited free minutes and keeps secrets hidden, or use GitHub
Pro, which allows 3,000 minutes.

## Setup (about 15 minutes)

1. **Telegram bot**
   - In Telegram, message **@BotFather**, send `/newbot`, and copy the token.
   - Send your new bot any message.
   - Open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy
     `message.chat.id`.
2. **Finnhub**: get a free API key at finnhub.io. The free tier covers US company
   news and the earnings calendar.
3. **Anthropic**: get an API key from console.anthropic.com. This costs a few pence
   per day at roughly 10 alerts.
4. **GitHub**
   - Create a **private** repo and push this folder to it.
   - Go to Settings → Secrets and variables → Actions and add these secrets:
     - `ANTHROPIC_API_KEY`
     - `FINNHUB_API_KEY`
     - `TELEGRAM_BOT_TOKEN`
     - `TELEGRAM_CHAT_ID`
5. **Test**: go to the Actions tab, open "US move alerts", choose "Run workflow" and set
   mode to `eod`. That run scans the last close, so you get output straight away.

Local test:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=... FINNHUB_API_KEY=... TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
python scanner.py --test-telegram
python scanner.py --mode eod --dry-run
```

## Restricting to what Revolut offers

Create `revolut_whitelist.txt` with one ticker per line. Lines starting with `#` are
ignored. If the file exists and isn't empty, only those names are scanned.

## Tuning

Thresholds are at the top of `scanner.py`:

- `MIN_IDIO_Z`
- `MIN_ABS_RET`
- `MIN_DOLLAR_VOL`
- `MAX_ALERTS_PER_RUN`

The analyst framework Claude applies is in `SYSTEM_PROMPT`.

## Caveats

- yfinance is unofficial and can break or lag. If it becomes unreliable, swap
  `download_prices()` for a paid feed; the rest of the code is unchanged.
- Intraday relative volume is projected linearly over the session, so it overstates
  RVOL early in the day.
- Finnhub's free news feed is broad but not Bloomberg. "No identifiable catalyst"
  sometimes means the feed missed the story, so check before fading.
- `evaluate.py` uses the alert-day close as the entry price.
