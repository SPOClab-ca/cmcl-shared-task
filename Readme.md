# TorontoCL at CMCL 2021 Shared Task

This repository contains the code for the TorontoCL submission to the [CMCL 2021 Shared Task](https://competitions.codalab.org/competitions/28176) on eye tracking prediction. We fine-tune RoBERTa-base with a custom token-level regression head, and leverage data from the [Provo](https://osf.io/sjefs/) eye tracking corpus for task-adaptive pretraining prior to fine-tuning. Our model ranked 3rd place out of 13 teams in the competition.

Team: Bai Li, Frank Rudzicz.

## Instructions to run

Run single model

```
PYTHONPATH=. python scripts/run_roberta.py --mode=submission --num-ensembles=1 --use-provo=True
```

Run ensemble of 10 models

```
PYTHONPATH=. python scripts/run_roberta.py --mode=submission --num-ensembles=10 --use-provo=True
PYTHONPATH=. python scripts/ensemble.py
```

## Notebooks

* ProvoProcess.ipynb: preprocesses the Provo data to have a similar form as ZuCo training data.
* MedianBaseline.ipynb: implements median, linear regression, and SVR baselines.

