from __future__ import annotations

from pathlib import Path

from src.data.dataset import get_train_test_split, load_dataset


def test_load_dataset_and_split() -> None:
    dataset_path = Path("data/sample/hour_sample.csv")
    df = load_dataset(dataset_path)

    assert not df.empty
    assert "cnt" in df.columns

    train_df, test_df = get_train_test_split(df, test_size=0.2)

    assert len(train_df) > len(test_df)
    assert train_df["dteday"].max() <= test_df["dteday"].min()
