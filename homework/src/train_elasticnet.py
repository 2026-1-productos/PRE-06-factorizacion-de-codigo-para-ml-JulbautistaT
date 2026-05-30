from sklearn.linear_model import ElasticNet

from ._internals.run_experiment import run_experiment

run_experiment(
    ElasticNet(
        alpha=0.5,
        l1_ratio=0.5,
        random_state=12345,
    )
)
