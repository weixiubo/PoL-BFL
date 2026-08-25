import pytest

torch = pytest.importorskip("torch")

from experiments.scripts.utils.models import create_model


def test_femnist_two_layer_cnn_matches_paper_scale_and_output():
    model = create_model("TwoLayerCNN", num_classes=62, input_channels=1)
    assert sum(parameter.numel() for parameter in model.parameters()) == 399_998
    output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 62)


def test_cifar_reference_models_have_expected_class_heads():
    assert create_model("ResNet18", num_classes=10).fc.out_features == 10
    assert create_model("ResNet34", num_classes=100).fc.out_features == 100
