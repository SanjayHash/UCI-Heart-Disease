Heart Disease Risk Prediction

Binary classification of heart disease presence using the UCI Heart Disease (Cleveland) dataset, with an interactive Streamlit app for live predictions and EDA exploration.

Live demo: https://uci-heart-disease-ekkwc8pjg3xjrekdb7slqg.streamlit.app/

Problem

303 patient records, 13 clinical features (age, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.), binary target (0 = no disease, 1 = disease present). Classes are close to balanced (164 vs. 139), so accuracy is a reasonable primary metric.

EDA and data cleaning

  --> No missing values, no duplicate rows. 
  -->Verified categorical-coded-as-numeric columns (  cp  ,   restecg  ,   slope  ,   ca  ,   thal  ) were not silently corrupted by placeholder strings, despite   isnull()   showing clean — confirmed via   .unique()   checks on each.
  --> Age vs. target (boxplot):  the disease-present group has a higher median age (~57) than the no-disease group (~52), with one low outlier — read as a general age trend, not a hard cutoff.
  --> Correlation heatmap vs. target:  no single feature dominates —   thal   (0.52),   ca   (0.46),   exang   (0.43),   oldpeak   (0.42),   cp   (0.41),   thalach   (-0.42) are the strongest, all in the moderate range. This spread (no 0.8+ correlation) reflects that heart disease is genuinely multi-factorial, not attributable to one measurement — unlike the mushroom project, where   odor   alone was a near-deterministic reflectors of the target.
  --> Multicollinearity noted:    oldpeak   and   slope   correlate with each other at 0.58, the highest non-target pairwise value on the heatmap.
  --> Distribution shapes:    chol   is right-skewed (long tail toward high values),   thalach   is left-skewed,   oldpeak   is heavily right-skewed (large spike near 0, long tail to 6). Not corrected via transformation, since the chosen model (tree-based) doesn't require it — see reasoning below.
  --> Outliers were intentionally not removed.  In clinical data, an extreme value (e.g. very high cholesterol) is frequently the actual signal of disease risk rather than noise — removing it would strip exactly the information the model needs. Only checked for biologically impossible values (e.g. resting BP of 0), of which there were none.

Why tree-based / ensemble

  -->All features are numeric or ordinal-as-numeric — tree splits work natively on both without needing separate encoding pipelines.
  -->Skewed distributions (  chol  ,   oldpeak  ) don't need transformation, since trees split on thresholds regardless of distribution shape.
  -->Multicollinearity (  oldpeak  /  slope   at 0.58) doesn't destabilize tree-based models the way it can for linear/logistic regression.
  -->Signal is spread across six moderately-correlated features rather than concentrated in one dominant column — this is exactly the setting where an  ensemble  (Random Forest) has an advantage over a single Decision Tree: averaging across many trees reduces the variance/overfitting risk of any single tree committing hard to one early split.

Modeling process (including a real overfitting fix)

| Model | Train Acc. | Test Acc. |
|---|---|---|
| Decision Tree (default) | — | 0.738 |
| Random Forest (default, unconstrained) | 1.00 | 0.885 |
| Random Forest (tuned:   max_depth=5  ,   min_samples_split=10  ,   min_samples_leaf=4  ) | 0.884 | 0.918 |

The unconstrained Random Forest hit 100% training accuracy — a clear overfitting signal, since the trees had grown deep enough to memorize the 242 training rows rather than learn generalizable patterns. Constraining tree depth and minimum samples per split/leaf fixed this: training accuracy dropped to 88.4% while test accuracy *improved* to 91.8%, confirming the model was generalizing better, not just fitting less.

 5-fold cross-validation on the tuned model:  scores ranged 0.783–0.885,  mean 0.828 . This is reported as the honest expected performance, since the single 91.8% test score came from one particular 61-row split and was somewhat optimistic — cross-validation corrects for that by averaging across 5 different splits.

 Feature importance (tuned model) vs. correlation heatmap — consistency check: 
  thal  ,   cp  ,   ca  ,   thalach  , and   oldpeak   rank highest in both the model's feature importances and the raw correlation-with-target values — agreement between two independent analyses, which supports that these are genuinely the most informative features rather than an artifact of one method.

Streamlit app

Two-tab interactive app built on the tuned Random Forest:
-->  Prediction tab  — sidebar inputs for all 13 features, returns a risk classification with confidence %.
-->  EDA Dashboard tab  — live target distribution, feature importance chart, full correlation heatmap, and an interactive dropdown to explore any numeric feature's distribution split by target.

*This is a portfolio demonstration only and is not a medical diagnostic tool.*

Tech stack

Python, pandas, scikit-learn (  DecisionTreeClassifier  ,   RandomForestClassifier  ,   cross_val_score  ), seaborn/matplotlib, Streamlit, joblib

Dataset

[UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

Running locally

bash
pip install -r requirements.txt
streamlit run app.py
