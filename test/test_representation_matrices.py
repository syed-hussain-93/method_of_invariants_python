import pytest

from representation_matrices import direct_sum, D3h


@pytest.fixture
def d3h():
    return D3h()


@pytest.fixture
def d3h_double():
    return D3h(is_double_group=True)


@pytest.fixture
def d3h_double_irreps(d3h_double):
    return d3h_double.irreducible_representations


def test_block_sum(d3h_irreps):
    assert len(d3h_irreps) == 6
