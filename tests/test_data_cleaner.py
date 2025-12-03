import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):

    def test_example_trim_strings_with_pandas_testing(self):
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        result = cleaner.trim_strings(df, ["name"])
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        result = cleaner.drop_invalid_rows(df, ["name"])
        expected_name = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        pdt.assert_series_equal(result["name"], expected_name)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        result_df = cleaner.drop_invalid_rows(df, ["name", "age"])
        self.assertEqual(result_df["name"].isna().sum(), 0)
        self.assertEqual(result_df["age"].isna().sum(), 0)
        self.assertLess(len(result_df), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        df = make_sample_df()
        original_df = df.copy()
        cleaner = DataCleaner()
        result_df = cleaner.trim_strings(df, ["name"])
        self.assertEqual(result_df.loc[0, "name"], "Alice")
        self.assertEqual(result_df.loc[3, "name"], "Carol")
        pdt.assert_series_equal(result_df["age"], original_df["age"])
        pdt.assert_series_equal(result_df["city"], original_df["city"])
        self.assertEqual(original_df.loc[0, "name"], " Alice ")

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        result_df = cleaner.remove_outliers_iqr(df, "age")
        self.assertNotIn(120, result_df["age"].values)
        self.assertIn(25, result_df["age"].values)
        self.assertIn(35, result_df["age"].values)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr(df, "salary")

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city")


if __name__ == "__main__":
    unittest.main()

