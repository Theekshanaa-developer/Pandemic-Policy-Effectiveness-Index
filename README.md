# Pandemic Policy Effectiveness (PPEI)

Short: code and analysis to compute a Pandemic Policy Effectiveness Index (PPEI), per-country lag estimation (CCF), an India case study, and simple forecasting pipelines (LSTM/GRU/TCN).

---

## Project structure
pandemic-policy-effectiveness/
├── 1_india_case_study/
│ ├── india_analysis.py
│ ├── india_visualizations.ipynb
│ └── outputs/
├── 2_ppei_global_analysis/
│ ├── ppei_compute.py
│ ├── ppei_waves.py
│ └── outputs/
├── 3_lag_optimization_ccf/
│ ├── ccf_compute.py
│ └── outputs/
├── 4_deep_learning_forecasting/
│ ├── lstm_model.py
│ ├── gru_model.py
│ └── outputs/
├── utils/
│ ├── preprocess.py
│ ├── feature_engineering.py
│ └── common.py
├── data/ # raw & processed data (ignored by default)
├── .github/workflows/ # optional CI
├── README.md
└── .gitignore
