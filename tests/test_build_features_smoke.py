import pandas as pd

from lapse_poc.data.build_features import build_policy_quarter_panel


def test_build_features_smoke(tmp_path):
    # Minimal synthetic rows with required columns (matching R export column names)
    df = pd.DataFrame({
        'policy_id': [1, 2],
        'issue_date': ['2000-01-01', '2000-01-01'],
        'duration': [0.10, None],  # first looks scaled (-> 10 qtrs); second censored
        'acc.death.rider': ['NoRider', 'Rider'],
        'gender': ['Male', 'Female'],
        'premium.frequency': ['Annual', 'InfraAnnual'],
        'risk.state': ['NonSmoker', 'Smoker'],
        'underwriting.age': ['Middle', 'Young'],
        'living.place': ['Other', 'Other'],
        'annual.premium': [0.1, -0.2],
        'DJIA': [0.0, 0.0],
        'termination.cause': ['surrender', None],
        'surrender': [1, 0],
        'death': [0, 0],
        'other': [0, 0],
        'allcause': [1, 0],
    })
    raw = tmp_path / "raw.csv"
    out = tmp_path / "panel.parquet"
    df.to_csv(raw, index=False)

    manifest = build_policy_quarter_panel(str(raw), str(out), horizon_qtrs=1, include_djia=False)

    assert out.exists()
    assert "duration_scale" in manifest
    