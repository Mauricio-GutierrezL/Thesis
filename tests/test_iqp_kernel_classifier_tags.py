from qml_benchmarks.models.iqp_kernel import (
    IQPKernelClassifier,
    IQPKernelClassifierHalfSeparable,
    IQPKernelClassifierSeparable,
)


def test_iqp_kernel_classifier_reports_classifier_tags():
    estimator = IQPKernelClassifier()

    assert estimator._estimator_type == "classifier"
    assert estimator.__sklearn_tags__().estimator_type == "classifier"


def test_iqp_kernel_variants_report_classifier_tags():
    for estimator_cls in (
        IQPKernelClassifierHalfSeparable,
        IQPKernelClassifierSeparable,
    ):
        estimator = estimator_cls()
        assert estimator._estimator_type == "classifier"
        assert estimator.__sklearn_tags__().estimator_type == "classifier"
