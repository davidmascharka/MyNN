from mynn.layers import dense, conv


def test_dense_init_not_constant():
    layer = dense(2, 3)
    assert not layer.bias.constant
    assert not layer.weight.constant


def test_conv_init_not_constant():
    layer = conv(2, 3)
    assert not layer.bias.constant
    assert not layer.weight.constant
