import streamlit as st

st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="centered")

st.title("✈️ US Domestic Flight Delay Prediction")
st.caption("DS 4420 Final Project · Spring 2026")

st.markdown("""
## Research Question
> *Can we predict whether a US domestic flight will be delayed by 15 or more minutes
> before departure, and what factors drive that risk?*

## Dataset
We use the **Bureau of Transportation Statistics (BTS) On-Time Performance** dataset,
covering all ~6.9 million US domestic flights in 2025. The target variable is `DEP_DEL15` —
a binary indicator of whether a flight departed 15 or more minutes late.

The dataset exhibits **moderate class imbalance**: approximately 78% of flights depart on
time and 22% are delayed.

## Methods

### 1 · Multilayer Perceptron (MLP)
A feed-forward neural network implemented in Keras, trained on a stratified 600k-flight
sample. Architecture:

| Layer | Width | Activation |
|-------|-------|------------|
| Input | ~175 features | — |
| Hidden 1 | 64 | ReLU + Dropout 0.3 |
| Hidden 2 | 32 | ReLU + Dropout 0.3 |
| Output | 1 | Sigmoid |

Trained with Adam (lr=0.001), binary cross-entropy loss, and class weights to correct for
imbalance. Achieved **ROC-AUC = 0.715** on the held-out test set.

### 2 · Bayesian Logistic Regression (Manual MCMC)
A Bayesian logistic regression model implemented from scratch in R using a
**Metropolis-Hastings** sampler — no pre-built ML packages. Trained on a stratified 50k
subsample due to MCMC computational constraints. Posterior inference over regression
coefficients provides uncertainty estimates alongside predictions.

## Key Findings
- Delay risk rises sharply through the day — early morning departures (5–7 AM) have
  the lowest delay rates (~10%), while evening departures (8–9 PM) exceed 35%.
- June and July carry the highest seasonal delay risk (~27%), driven by weather and
  peak travel volume.
- Carrier identity and origin airport are meaningful predictors even after controlling
  for time-of-day and seasonality.
- The primary ceiling on model performance is the absence of real-time weather data,
  which the literature identifies as the strongest individual predictor of flight delays.

---
👈 **Use the sidebar to explore the interactive model demo.**
""")
