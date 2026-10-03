from src import config


def test_shared_defaults_and_performance_bands_match_project_decisions() -> None:
    assert config.RANDOM_SEED == 42
    assert config.DEFAULT_ALPHA == 0.05
    assert config.DEFAULT_CONFIDENCE == 0.95
    assert config.LEAKY_COLUMNS == ["G1", "G2", "G3"]
    assert config.PERFORMANCE_BANDS == {
        "LOW_UPPER_EXCLUSIVE": 10,
        "MEDIUM_LOWER_INCLUSIVE": 10,
        "MEDIUM_UPPER_INCLUSIVE": 13,
        "HIGH_LOWER_INCLUSIVE": 14,
    }


def test_dataset_column_role_lists_cover_33_unique_variables() -> None:
    role_columns = (
        config.NUMERIC_COLUMNS
        + config.ORDINAL_COLUMNS
        + config.NOMINAL_COLUMNS
        + config.BINARY_COLUMNS
    )

    assert len(role_columns) == 33
    assert len(set(role_columns)) == len(role_columns)
