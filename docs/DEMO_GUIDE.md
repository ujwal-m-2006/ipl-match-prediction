# Weekly demonstration guide

A runbook for presenting this project: what to run beforehand, what to show,
what to say, and what to do when something goes wrong in front of an examiner.

---

## Before every demo — run this one command

```bash
python scripts/verify.py
```

It takes about 90 seconds and checks **51 things**: every dependency, the
database contents, all four trained models, every prediction path, all nine
dashboard pages rendered against the real data, every button on every page,
fourteen API endpoints including their error handling, and the full test suite.

It prints one verdict:

```
ALL 51 CHECKS PASSED (72s)

The system is ready to demonstrate.
```

**If anything says FAIL, do not start the demo** — the output names the exact
check and shows the last few lines of the error.

Short on time? `python scripts/verify.py --quick` runs the essentials in ~20
seconds.

---

## Starting up (do this 5 minutes early)

Terminal 1 — the dashboard:

```bash
streamlit run streamlit_app.py
```

Terminal 2 — the API, if you plan to show it:

```bash
python scripts/run_api.py
```

Open **http://localhost:8501** and leave it on the Home page. Open
**http://localhost:8000/docs** in a second browser tab.

> **Minimise the terminal window before you present.** The app logs normal INFO
> lines while it runs; they are not errors, but a scrolling terminal behind your
> slides invites questions you do not need.

---

## A 10-minute walkthrough

### 1. Home (1 min) — establish the scale

Point at the four tiles: **19 seasons, 1,246 matches, 1,457 players, 280,125
deliveries**. Say where it came from:

> "The primary source is the official IPL website. It's a JavaScript app that
> renders from public JSON feeds, so I read those feeds directly rather than
> scraping HTML — same data, one step earlier, and it doesn't break when they
> restyle the site. That covers 2019 to 2026. The official host doesn't index
> older seasons, so Cricsheet back-fills 2008 to 2018."

Then open the **Toss impact** tab. The line sits near 50%, and there's a
reference line drawn at exactly 50%.

> "Winning the toss is worth almost nothing. That's a real finding, and it's
> foreshadowing the main result."

### 2. Schedule & Results (1 min) — show the data is real

Pick **2026 → League table**. Points and net run rate, computed to IPL rules.
Switch to the **Scorecard** tab and open any match: full batting and bowling
cards, extras, Player of the Match.

> "This isn't a summary table someone downloaded. It's ball-by-ball data
> aggregated into scorecards, and it reconciles exactly with the published
> ones."

### 3. Predictions → Chase simulator (2 min) — **your strongest demo**

Set target 180, 120 runs, 3 wickets, 13 overs. Press **Simulate**.

Show the gauge, then the sensitivity curve underneath.

> "This model scores 0.897 ROC-AUC on seasons it never saw in training. Watch
> what happens as I add runs —"

Change wickets from 3 to 7 and re-simulate. The probability drops sharply.

> "It has learned that wickets in hand matter more than the run rate at this
> stage, which is exactly what a commentator would tell you."

### 4. Predictions → Match winner (2 min) — **the intellectual centrepiece**

Run any fixture. Then read the warning box aloud rather than hiding it.

> "This one is barely better than a coin toss — about 0.55 AUC. I want to be
> straight about that, because it's the most interesting result in the project.
>
> I checked it properly: I re-evaluated at six different cut-off points, from
> holding out the last two seasons up to the last eight — 541 test matches. It
> stayed between 0.50 and 0.55 every time. That's not noise.
>
> Then I looked at why. No pre-match feature correlates with the result above
> r = 0.09. Not form, not head-to-head, not home advantage, not squad quality.
> The home side wins 52% of IPL matches. There simply isn't much to predict
> before the toss.
>
> I didn't stop there — I added Playing XI strength from each player's career
> record, era-aware scoring levels, and I mirror every training fixture so the
> model can't exploit which team happens to be listed first. That moved it from
> *below* chance up to chance. It didn't invent signal that isn't in the data.
>
> The contrast with the chase model is the point: same pipeline, same
> algorithms, 0.90 versus 0.55. The difference is information, not technique."

This paragraph is worth more marks than a fake 78% accuracy. It shows you
understand evaluation, not just fitting.

### 5. Model Comparison (2 min) — the required comparison

All six algorithms per task, with accuracy, precision, recall, F1 and ROC-AUC.
Open the expander titled *"Why the pre-match winner model scores near 0.50"*.

Scroll to the **calibration** chart.

> "Accuracy isn't enough when you're reporting probabilities. This checks
> whether a '70% confident' prediction is actually right 70% of the time."

Then **feature importance** for the chase model — wickets in hand and required
run rate dominate, which is the sanity check that it learned cricket rather
than noise.

### 6. Admin + API (2 min) — the engineering

**Admin → Health**: row counts, last pipeline run, which models are trained.

> "The pipeline is incremental — a match that's already stored and already
> finished gets skipped. A daily refresh costs one request per season. And
> everything is rate-limited to one request every 0.6 seconds with caching, so
> it's polite to the source."

Then http://localhost:8000/docs — expand `/predict/chase` and hit **Try it out**.

> "Same models, served over REST. The dashboard and the API both call one
> prediction service, so they can't drift apart."

---

## Questions you should expect

**"Why is the accuracy so low?"**
Covered above. Emphasise: it was verified across 541 test matches, the ceiling
is set by the data, and the chase model at 0.90 proves the pipeline is sound.

**"Why not use a neural network?"**
With 1,246 training rows and features correlating below r = 0.09, a neural
network overfits faster and predicts no better. The gradient-boosted models are
already at the information ceiling. The honest constraint is data, not model
capacity.

**"How do you know there's no data leakage?"**
Two things. The split is by *season*, not random — the model never sees a match
from the test years. And features are built by walking matches in date order,
emitting each row *before* folding that match's result into the rolling state.
There's a test that asserts the head-to-head count on row *n* equals exactly the
number of prior meetings.

**"Is scraping the IPL site allowed?"**
It reads public endpoints the site serves to every visitor, one request at a
time with a 0.6-second delay and on-disk caching, with a descriptive
User-Agent. It's an independent educational project, stated in the README and
LICENSE.

**"What would you improve next?"**
Ball-by-ball player matchups (this batter versus this bowler), venue pitch
conditions, and reconciling player identity across the two sources — Cricsheet
writes "V Kohli", the official feed writes "Virat Kohli", and matching on
surname plus initial would collide on "R Sharma".

---

## If something breaks mid-demo

| Symptom | Fix |
|---|---|
| A page shows an error | Press **R** to rerun. If it persists, switch pages and come back — the caches rebuild. |
| Dashboard won't start | Port in use. `streamlit run streamlit_app.py --server.port 8502` |
| "No matches in the database" | Wrong directory. `cd` to the project root and restart. |
| "Model has not been trained" | `python scripts/train_models.py` (~2 minutes) |
| Everything is broken | `git stash && python scripts/verify.py` — returns to the last known-good commit's code. |

**Have a backup.** Before the demo, screenshot the key pages. If the laptop
misbehaves, you can still talk through the results.

---

## Between demos — adding work each week

Refresh the data (seconds, thanks to caching):

```bash
python scripts/ingest.py --seasons 2026
```

Retrain if new matches landed:

```bash
python scripts/train_models.py
```

Then commit, so the marker can see week-on-week progress:

```bash
git add -A && git commit -m "Week N: <what you added>"
```

### Ideas, roughly in order of effort against marks

| Week | Addition | Where |
|---|---|---|
| — | Player-vs-bowler matchup stats from `deliveries` | new `analytics/matchups.py` |
| — | Hyperparameter tuning with `GridSearchCV`, reported on the comparison page | `models/train.py` |
| — | SHAP explanations to replace the current driver table | `models/predict.py` |
| — | A win-probability *timeline* across a whole chase | `dashboard/views/predictions.py` |
| — | Cluster venues by scoring behaviour (k-means) | new `analytics/clustering.py` |
| — | Deploy publicly on Streamlit Cloud and demo from a phone | `docs/DEPLOYMENT.md` |

Each is self-contained, and `scripts/verify.py` will tell you immediately if it
broke anything.

---

## One-line summary, if you only get 30 seconds

> "An end-to-end ML system on 19 IPL seasons — 1,246 matches and 280,000
> deliveries collected from the official IPL feeds, with six algorithms compared
> on held-out seasons. The in-play chase model reaches 0.90 AUC. The pre-match
> winner model reaches only 0.55, and I can show you why that's the honest
> answer rather than a bug."
