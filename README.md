# The `cenreg` Package

The Python package `cenreg` is repository for probabilistic forecasts such as quantile regression and distribution regression and for censored regression such as survival analysis and interval-censored data analysis.


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

[1] [H. Yanagisawa and S. Akiyama, Survival Analysis via Density Estimation, ICML 2025](https://icml.cc/virtual/2025/poster/43491)

```
  @InProceedings{yanagisawa2025survival,
    author    = {Yanagisawa, Hiroki and Akiyama, Shunta},
    booktitle = {Proceedings of ICML 2025},
    title     = {Survival Analysis via Density Estimation},
    year      = {2025},
  }
```

[2] [H. Yanagisawa, Proper Scoring Rules for Survival Analysis, ICML 2023](https://proceedings.mlr.press/v202/yanagisawa23a/yanagisawa23a.pdf)

```
@InProceedings{yanagisawa2023proper,
   author    = {Yanagisawa, Hiroki},
   booktitle = {Proceedings of ICML 2023},
   title     = {Proper Scoring Rules for Survival Analysis},
   year      = {2023},
}
```
