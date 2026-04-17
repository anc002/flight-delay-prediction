# Flight Delay Prediction

This project predicts flight delays using historical airline operational data. The goal is to evaluate different machine learning approaches for delay prediction and compare their predictive performance.

## Models

The project implements two modeling approaches:

Multi-Layer Perceptron (MLP) neural network

Bayesian classification model

The MLP is used to model nonlinear relationships between flight features, while the Bayesian model provides probabilistic predictions and parameter uncertainty.

## Data

The models use historical flight data from the U.S. Bureau of Transportation Statistics (BTS) On-Time Performance dataset, which contains operational flight information such as:

- origin and destination airports

- scheduled departure time

- day of week and month

- airline carrier

- route distance

- delay outcomes

These features are used to predict whether a flight will be delayed.

Data was downloaded one month at a time and combined into a master dataset too big to upload to github (~ 7 million rows). Creation process can be seen in Dataset_creation.ipynb. T_ONTIME_REPORTING.csv is an example dataset for 1 month

SOURCE: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr

## Objective

The objective is to compare the predictive performance of a neural network model and a Bayesian model for flight delay prediction using structured airline operational data.
