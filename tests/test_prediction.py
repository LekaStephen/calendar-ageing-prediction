import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prediction import predict_capacity_loss

class DummyModel:
    def predict(self, X):
        assert list(X.columns) == ["storage_time_days", "temperature_c", "soc_pct"]
        return [2.5]

def test_prediction_and_retention():
    out = predict_capacity_loss(DummyModel(), 25, 50, 240)
    assert out["capacity_loss_pct"] == 2.5
    assert out["capacity_retention_pct"] == 97.5

def test_out_of_range_temperature_rejected():
    with pytest.raises(ValueError):
        predict_capacity_loss(DummyModel(), 65, 50, 240)

def test_out_of_range_time_rejected():
    with pytest.raises(ValueError):
        predict_capacity_loss(DummyModel(), 25, 50, 22000)

def test_invalid_soc_rejected():
    with pytest.raises(ValueError):
        predict_capacity_loss(DummyModel(), 25, 105, 240)

def test_negative_storage_time_rejected():
    with pytest.raises(ValueError):
        predict_capacity_loss(DummyModel(), 25, 50, -1)

def test_non_finite_input_rejected():
    with pytest.raises(ValueError):
        predict_capacity_loss(DummyModel(), float('nan'), 50, 240)


def test_saved_model_loads_and_predicts_finite_result():
    from prediction import load_model
    model, metadata = load_model()
    result = predict_capacity_loss(model, 25, 50, 8760)
    assert metadata['model'] == 'RBF Support Vector Regression'
    assert set(result) == {'capacity_loss_pct', 'capacity_retention_pct'}
    assert all(__import__('math').isfinite(v) for v in result.values())
    assert result['capacity_loss_pct'] + result['capacity_retention_pct'] == pytest.approx(100.0)


def test_processed_dataset_integrity():
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / 'data/processed/reconstructed_capacity_analysis.csv')
    assert len(df) == 595
    assert df['condition_id'].nunique() == 17
    assert df.groupby('condition_id').size().eq(35).all()
    assert not df.duplicated(['condition_id', 'storage_time_hours']).any()
    assert df['storage_time_hours'].min() == pytest.approx(0.0)
    assert df['storage_time_hours'].max() == pytest.approx(21241.0)
    assert df['capacity_loss_pct'].min() < 0  # negative measurement variation is retained


def test_saved_model_feature_schema_matches_metadata():
    from prediction import load_model
    model, metadata = load_model()
    assert list(model.feature_names_in_) == metadata['features']
    assert metadata['target'] == 'capacity_loss_pct'


def test_saved_model_representative_inference_regression_checks():
    from prediction import load_model
    model, _ = load_model()
    cases = [
        (25, 50, 8760, 3.0817906778),
        (40, 50, 8760, 5.8346871784),
        (60, 50, 21241, 19.9639758051),
        (0, 50, 0, -0.0898845260),
    ]
    for temperature, soc, hours, expected_loss in cases:
        result = predict_capacity_loss(model, temperature, soc, hours)
        assert result['capacity_loss_pct'] == pytest.approx(expected_loss, abs=1e-8)
        assert result['capacity_retention_pct'] == pytest.approx(100 - expected_loss, abs=1e-8)
