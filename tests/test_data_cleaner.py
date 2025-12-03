import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos.
        
        Este test demuestra cómo usar pandas.testing.assert_frame_equal() para comparar
        DataFrames completos, lo cual es útil porque maneja correctamente los índices,
        tipos de datos y valores NaN de Pandas.
        """
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        # DataFrame esperado después de trim
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        
        # Usar pandas.testing.assert_frame_equal() para comparar DataFrames completos
        # Esto maneja correctamente índices, tipos y estructura de Pandas
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar Series.
        
        Este test demuestra cómo usar pandas.testing.assert_series_equal() para comparar
        Series completas, útil cuando queremos verificar que una columna completa tiene
        los valores esperados manteniendo los índices correctos.
        """
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name"])
        
        # Verificar que la columna 'name' ya no tiene valores faltantes
        # Los índices después de drop_invalid_rows son [0, 2] (se eliminó la fila 1)
        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        
        # Usar pandas.testing.assert_series_equal() para comparar Series completas
        # Esto verifica valores, índices y tipos correctamente
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        df = self.make_sample_df()
        cleaner = DataCleaner(df.copy())
        result_df = cleaner.drop_invalid_rows(["name", "age"])
        missing_in_name = result_df["name"].isna().sum()
        missing_in_age = result_df["age"].isna().sum()
        self.assertEqual(missing_in_name, 0, "La columna 'name' debería tener 0 valores faltantes")
        self.assertEqual(missing_in_age, 0,"La columna 'age' debería tener 0 valores faltantes")
        self.assertLess(len(result_df), len(df), "El DataFrame resultante debería tener menos filas que el original")

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        df = self.make_sample_df()
        cleaner = DataCleaner(df)
        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        df = self.make_sample_df()
        original_df = df.copy()
        cleaner = DataCleaner(df.copy())
        result_df = cleaner.trim_strings(["name"])
        self.assertEqual(original_df.loc[0, "name"], " Alice ","El DataFrame original debería mantener los espacios en blanco")
        self.assertEqual(result_df.loc[0, "name"], "Alice","El primer nombre debería estar sin espacios al inicio/final")
        self.assertEqual(result_df.loc[2, "name"], "Charlie","El tercer nombre debería estar sin espacios al final")
        pdt.assert_series_equal(result_df["city"], original_df["city"],"La columna 'city' no debería ser modificada")
        pdt.assert_series_equal(result_df["age"], original_df["age"],"La columna 'age' no debería ser modificada")

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        df = self.make_sample_df()
        cleaner = DataCleaner(df)
        with self.assertRaises(TypeError):
            cleaner.trim_strings(["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        df = self.make_sample_df()
        cleaner = DataCleaner(df.copy())
        result_df = cleaner.remove_outliers_iqr("age", factor=1.5)
        self.assertNotIn(120, result_df["age"].values,"El valor extremo 120 debería ser eliminado")
        has_normal_value = any(val in result_df["age"].values for val in [25, 35])
        self.assertTrue(has_normal_value,"Al menos uno de los valores normales (25 o 35) debería permanecer")
        normal_values_present = [val for val in [25, 35] if val in result_df["age"].values]
        self.assertGreater(len(normal_values_present), 0,"Debería haber al menos un valor normal presente")

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        df = self.make_sample_df()
        cleaner = DataCleaner(df)
        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr("salary")

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        df = self.make_sample_df()
        cleaner = DataCleaner(df)
        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr("city")


if __name__ == "__main__":
    unittest.main()
