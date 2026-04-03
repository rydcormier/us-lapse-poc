import torch

from lapse_poc.models.tabular import TabularNet


def test_tabular_forward_shapes():
    model = TabularNet(cat_cardinalities=[5, 3, 10], n_num=2)
    x_cat = torch.randint(0, 5, (8, 3))  # batch size 8, 3 categorical features
    x_num = torch.randn(8, 2)  # batch size 8, 2 numerical features
    y = model(x_cat, x_num)
    assert y.shape == (8,)  # assuming output is a single value per instance
