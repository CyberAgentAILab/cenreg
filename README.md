# The `cenreg` Package

The Python package `cenreg` is a repository for probabilistic forecasts such as quantile regression and distribution regression and for censored regression such as survival analysis and interval-censored data analysis.

Features:
+ **Tree-based models** run fast and output accurate predictions.
+ **Neural network models** are implemented both for structured data (e.g., tabular data) and non-structured data (e.g., image data).
+ Both models can handle **competing risks**.
+ Both models are based on the (conditional) independence assumption or the non-informative assuption, but they can also handle **dependent censoring** based on assumed copula.
+ **Strictly proper scoring rules** are implemented to evaluate the discrimination performances of prediction models.  The scoring rules can handle right-censored and interval-censored data.
+ **Binning-free calibration metrics** are implemented to evaluate the calibration performances of prediction models.  The calibration metrics can handle right-censored and interval-censored data.


## Getting Started

### Prerequisites

You first need to install `SurvSet` via pip
```
pip install SurvSet
```

Additionally, denpending on the models you want to use, you also need to install
+ LightGBM
+ PyTorch

### Installation

You can install `cenreg` via pip:
```
pip install cenreg
```

### Run Sample Code

You can find our sample codes in the `notebooks` directory.

### Documentation

Read the [documentation](https://cyberagentailab.github.io/cenreg/) to get started.


## Citation

+ H. Yanagisawa and S. Akiyama, [A Strictly Proper Scoring Rule and a Calibration Metric for Interval-Censored Data Analysis](https://icml.cc/virtual/2026/poster/65937), ICML 2026 (Paper in [OpenReview](https://openreview.net/forum?id=8jViL8YrB1))
+ H. Yanagisawa and S. Akiyama, [Survival Analysis via Density Estimation](https://icml.cc/virtual/2025/poster/43491), ICML 2025 (Paper in [OpenReview](https://openreview.net/forum?id=z9SRjXPf8T))
+ H. Yanagisawa, [Proper Scoring Rules for Survival Analysis](https://proceedings.mlr.press/v202/yanagisawa23a/yanagisawa23a.pdf), ICML 2023
