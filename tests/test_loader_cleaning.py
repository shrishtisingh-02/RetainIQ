import pandas as pd
import pytest

from retainiq import config
from retainiq.cleaning import clean_transactions
from retainiq.data_loader import find_raw_file, load_raw, standardize_columns, validate_schema


def _row(**kw):
    base = dict(InvoiceNo="1001", StockCode="A1", Description="x", Quantity=2,
                InvoiceDate="2011-01-05 10:00:00", UnitPrice=3.0, CustomerID=1.0,
                Country="United Kingdom")
    base.update(kw)
    return base


def test_find_raw_file_missing_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="archive.ics.uci.edu"):
        find_raw_file(tmp_path)


def test_load_raw_csv_and_alias_columns(tmp_path):
    df = pd.DataFrame([_row()]).rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price",
                                                "CustomerID": "Customer ID"})
    path = tmp_path / "online_retail.csv"
    df.to_csv(path, index=False)
    loaded = load_raw(path)
    assert set(config.REQUIRED_COLUMNS) <= set(loaded.columns)


def test_load_raw_missing_file_and_bad_extension(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")
    bad = tmp_path / "data.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        load_raw(bad)


def test_validate_schema_rejects_missing_columns_and_empty():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(pd.DataFrame({"InvoiceNo": [1]}))
    empty = pd.DataFrame(columns=config.REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="no rows"):
        validate_schema(standardize_columns(empty))


def test_cleaning_removes_each_invalid_type():
    rows = [
        _row(),                                   # valid
        _row(InvoiceNo="1002", CustomerID=None),  # missing customer
        _row(InvoiceNo="C1003"),                  # cancellation
        _row(InvoiceNo="1004", Quantity=-1),      # negative quantity
        _row(InvoiceNo="1005", UnitPrice=0),      # zero price
        _row(InvoiceNo="1006", StockCode="POST"), # fee code
        _row(InvoiceNo="1007", InvoiceDate="not a date"),
        _row(),                                   # exact duplicate of the valid row
    ]
    clean, report = clean_transactions(pd.DataFrame(rows))
    assert len(clean) == 1
    assert report["removed_missing_customer_id"] == 1
    assert report["removed_cancellations"] == 1
    assert report["removed_non_positive_qty_or_price"] == 2
    assert report["removed_non_product_codes"] == 1
    assert report["removed_unparseable_values"] == 1
    assert report["removed_exact_duplicates"] == 1
    assert clean.loc[0, "Revenue"] == pytest.approx(6.0)
    assert clean["CustomerID"].dtype.kind == "i"


def test_cleaning_synthetic_data_has_no_invalid_rows(clean_df):
    assert clean_df["Revenue"].gt(0).all()
    assert clean_df["CustomerID"].notna().all()
    assert not clean_df["InvoiceNo"].str.startswith("C").any()
    assert not clean_df.duplicated().any()
