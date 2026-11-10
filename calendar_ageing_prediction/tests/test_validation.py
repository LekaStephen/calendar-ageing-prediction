import pytest
from battery_ageing.validation import validate_inputs

def test_valid_values_are_normalized():
    assert validate_inputs(25, 50, 8760) == (25.0, 50.0, 8760.0)

@pytest.mark.parametrize("values", [(-1,50,100), (61,50,100), (25,-1,100), (25,101,100), (25,50,-1), (25,50,21242)])
def test_out_of_range_values_fail(values):
    with pytest.raises(ValueError):
        validate_inputs(*values)

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_fail(bad):
    with pytest.raises(ValueError):
        validate_inputs(bad, 50, 100)

def test_non_numeric_value_fails():
    with pytest.raises(ValueError):
        validate_inputs("warm", 50, 100)
