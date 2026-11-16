# Motorcycle Insurance Pricing

Claim frequency and severity models for Swedish motorcycle insurance (Wasa, 1994 to 1998):
a Poisson GLM for frequency, a Gamma GLM for claim cost, and a test of whether gradient
boosting beats them under nested cross-validation.

**Live page:** https://YOUR-USERNAME.github.io/motorcycle-pricing/

## What it shows

- A Poisson GLM reduces held-out deviance by 12.9% versus a no-model baseline; gradient boosting
  reaches 12.4% and does not beat the GLM in any of 10 folds.
- Out-of-fold predictions are well calibrated: the riskiest decile claims about 6.0 times per 100
  policy years against 0.2 for the safest.
- Owner age, geographic zone, bike age and bike class drive frequency; bike age also drives cost per claim.
- Frequency times severity gives a pure premium that tracks actual cost by quintile.

## Reproduce

```bash
pip install -r requirements.txt
./fetch_data.sh          # downloads data/ohlsson.csv
python analysis.py       # about 3 minutes; writes results/
```

`analysis.py` writes the figures and `results/results.json`. The web page (`index.html`) shows those
figures, so rerun the script before editing numbers on the page.

## Data

Ohlsson, E. and Johansson, B. (2010), *Non-life insurance pricing with generalized linear models*,
Springer. Accessed through the Rdatasets collection (R package `insuranceData`). The data file is not
stored in this repository; `fetch_data.sh` downloads it.

## Notes

- 2,074 cells with zero exposure (holding 4 claims) are dropped.
- Sex is in the original data. It is no longer permitted as a rating factor in EU and UK insurance pricing.
- Analysis code was written with help from Claude (Anthropic); the author reviewed the method and results.
