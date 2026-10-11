# GitHub, Resume & Interview Guide

## 1. Push to GitHub (beginner-friendly)

1. On github.com click **New repository**, name it `retainiq-customer-analytics`, leave it **empty** (no README/.gitignore/license), click **Create**.
2. Open the project folder in VS Code (**File → Open Folder**), then open the terminal (**Terminal → New Terminal**) and run:

```bash
git init
git add .
git status                      # check: no venv/, outputs/ or data/raw/*.xlsx listed
git commit -m "Initial commit: RetainIQ revenue forecasting and retention analytics"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/retainiq-customer-analytics.git
git push -u origin main
```

If git asks who you are (first time only):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

GitHub will ask you to sign in in the browser (or use a Personal Access Token as the password). Afterwards, repeat changes with `git add . && git commit -m "message" && git push`.

Before pushing, run the pipeline on the real data, fill in the **Results** section of `README.md`, add 2-3 dashboard screenshots to `docs/screenshots/`, and add a repo description and topics (`data-analytics`, `streamlit`, `scikit-learn`, `customer-retention`).

## 2. Resume entry

**Title:** RetainIQ: Customer Revenue Forecasting & Retention Analytics

**Description (2-3 lines):** End-to-end analytics project on the UCI Online Retail dataset that cleans transaction data, engineers customer-level features, forecasts 90-day revenue and repeat-purchase probability, and delivers a Streamlit dashboard with RFM segments and recommended retention actions.

**Tech stack:** Python, Pandas, NumPy, scikit-learn, SQL (SQLite), Plotly, Streamlit, pytest, Git

**Bullets** (fill the brackets with numbers from YOUR real run; delete any bullet whose number you did not measure):

- Built a data-cleaning and validation pipeline for [N] raw retail transactions, removing cancellations, guest checkouts, fee codes and duplicates, and logged an audit report of rows removed at each step.
- Engineered 13 leakage-safe customer features (RFM, tenure, purchase velocity, rolling revenue) and trained a gradient-boosting revenue model tuned with RandomizedSearchCV (5-fold CV) on a time-based train/test split, reaching log-scale R² of [X.XX] vs. [Y.YY] for a mean baseline.
- Delivered a Streamlit dashboard with RFM segmentation, churn-risk × value-tier action rules, cohort retention heatmap and SQL-driven KPIs, backed by 29 automated tests.

**Skills demonstrated:** data cleaning, feature engineering, time-series-aware validation, hyper-parameter tuning, model evaluation against baselines, SQL (CTEs, window functions), cohort analysis, dashboarding, testing, Git.

**Only claim what your run shows.** The numbers in other people's READMEs (including the reference project's R² of ~0.81) are not yours. If your own R² is lower, that is fine. Being able to explain why (see Q5) is a stronger interview answer than a number you cannot reproduce.

## 3. Likely interview questions

**1. Walk me through the project.** (60-second pitch below.)

**2. Why a time-based split instead of a random split?** The goal is to predict the future. A random split lets the model see customers' later behaviour during training (leakage) and overstates performance. I train on features as of `T−2H`, test on features as of `T−H`, and the targets come from the windows after each cutoff. A unit test checks that features never use data after the cutoff.

**3. How did you define churn?** In non-contractual retail there is no cancellation event, so churn = no purchase within the next 90 days. The classifier predicts P(purchase), and churn probability = 1 − that. It is a proxy, and I say so in the README.

**4. Why log-transform the revenue target?** Customer revenue is extremely right-skewed. Training on `log1p(revenue)` stops a few huge customers dominating the loss; predictions are converted back with `expm1`.

**5. Why report several metrics, and why might R² be low?** Raw-scale R² is dominated by a handful of very large customers, so I also report log-scale R², MAE, and "top-20% revenue capture" (share of real revenue earned by the customers the model ranks highest), which is closer to the business use. I compare against baselines (mean, last-90-days persistence, Ridge) so the gain from the model is visible.

**6. What is RandomizedSearchCV and why use it?** It samples hyper-parameter combinations from distributions and scores each with k-fold CV. It covers a large space far cheaper than a full grid search.

**7. What leakage risks did you consider?** Using future transactions in features, tuning on the test set, and refitting on test data before evaluation. Features are cut off by date; tuning uses CV on the training table only; the refit on train+test happens after evaluation, only for the final scoring.

**8. How do the RFM segments differ from the model outputs?** RFM is descriptive (what the customer did); the models are predictive (what they will do). Combining a predicted-value tier with churn risk gives the action matrix (e.g. high value + high risk → priority win-back).

**9. How would you prove the recommended actions work?** Run an A/B test: contact a random half of the high-risk/high-value group and compare repurchase and revenue against the untouched half. The model ranks customers, but only an experiment shows campaign uplift.

**10. What would you improve?** Multi-year data for seasonality, rolling multi-cutoff training, probability calibration, SHAP explanations, scheduled refreshes, and CI.

**11. Why SQL if you have pandas?** Analysts are expected to extract insight with SQL. The queries (CTEs, `NTILE` window function) are cross-checked against pandas in the tests.

**12. How did you test it?** Unit tests with hand-computed feature values, a leakage test, cleaning tests for each invalid-row type, SQL-vs-pandas equality checks, end-to-end pipeline tests, invalid-input tests, and Streamlit `AppTest` smoke tests.

## 4. 60-second placement pitch

"RetainIQ answers three questions for an online retailer: who will spend in the next 90 days, who is likely to stop buying, and where retention effort should go. I took the UCI Online Retail transactions, cleaned them with an audited pipeline, and engineered customer-level features such as recency, frequency, monetary value, and rolling revenue. To keep the evaluation honest I used a time-based split with a date cutoff so the model never sees the future. I tuned a gradient-boosting model with randomized search and 5-fold cross-validation and compared it with simple baselines. I also built a repeat-purchase classifier, combined both outputs with RFM segments into an action matrix, and put it all in a Streamlit dashboard with SQL insights and cohort retention. The pipeline is covered by 29 automated tests. The main limitation is that the data covers only about a year, so I can't learn seasonality, which is the first thing I'd fix with more data."
