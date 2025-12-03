import numpy as np
import numpy.testing as npt
import unittest

from src.statistics_utils import StatisticsUtils


class TestStatisticsUtils(unittest.TestCase):
    """Test suite for StatisticsUtils class."""

    def test_example_moving_average_with_numpy_testing(self):
        """Ejemplo de test usando numpy.testing para comparar arrays de NumPy.
        
        Este test demuestra cómo usar numpy.testing.assert_allclose() para comparar
        arrays de NumPy con tolerancia para errores de punto flotante, lo cual es
        esencial cuando trabajamos con operaciones numéricas.
        """
        utils = StatisticsUtils()
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = utils.moving_average(arr, window=3)
        
        # Valores esperados para media móvil con window=3
        expected = np.array([2.0, 3.0, 4.0])
        
        # Usar numpy.testing.assert_allclose() para comparar arrays de NumPy
        # Esto maneja correctamente errores de punto flotante con tolerancia
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_example_min_max_scale_with_numpy_testing(self):
        """Ejemplo de test usando numpy.testing para verificar transformaciones numéricas.
        
        Este test demuestra cómo usar numpy.testing.assert_allclose() para verificar
        que una transformación numérica produce los resultados correctos en todo el array,
        considerando errores de punto flotante en cálculos matemáticos.
        """
        utils = StatisticsUtils()
        arr = [10.0, 20.0, 30.0, 40.0]
        result = utils.min_max_scale(arr)
        
        # Valores esperados después de min-max scaling: (x - min) / (max - min)
        # min=10, max=40, range=30
        # [10->0.0, 20->0.333..., 30->0.666..., 40->1.0]
        expected = np.array([0.0, 1/3, 2/3, 1.0])
        
        # Usar numpy.testing.assert_allclose() para comparar arrays numéricos
        # La tolerancia relativa y absoluta permite errores pequeños de punto flotante
        npt.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_moving_average_basic_case(self):
        utils = StatisticsUtils()
        arr = [1, 2, 3, 4]
        result = utils.moving_average(arr, window=2)
        expected = np.array([1.5, 2.5, 3.5])
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7,err_msg="El cálculo de la media móvil no es correcto")
        self.assertEqual(result.shape, (3,),f"El shape del resultado es {result.shape}, debería ser (3,)")

    def test_moving_average_raises_for_invalid_window(self):
        utils = StatisticsUtils()
        arr = [1, 2, 3]
        with self.assertRaises(ValueError) as context:
            utils.moving_average(arr, window=0)
        self.assertIn("window must be a positive integer", str(context.exception),"Debería lanzar ValueError para window <= 0")
        with self.assertRaises(ValueError) as context:
            utils.moving_average(arr, window=5)
        self.assertIn("window must not be larger than the array size", str(context.exception),"Debería lanzar ValueError para window > len(arr)")

    def test_moving_average_only_accepts_1d_sequences(self):
        utils = StatisticsUtils()
        arr_2d = [[1, 2], [3, 4]]
        with self.assertRaises(ValueError) as context:
            utils.moving_average(arr_2d, window=2)
        self.assertIn("moving_average only supports 1D sequences", str(context.exception),"Debería lanzar ValueError para secuencias multidimensionales")

    def test_zscore_has_mean_zero_and_unit_std(self):
        utils = StatisticsUtils()
        arr = [10, 20, 30, 40]
        result = utils.zscore(arr)
        mean_result = np.mean(result)
        self.assertAlmostEqual(mean_result, 0.0, places=10,msg=f"La media de los z-scores es {mean_result}, debería ser 0.0")
        std_result = np.std(result)
        self.assertAlmostEqual(std_result, 1.0, places=10,msg=f"La desviación estándar de los z-scores es {std_result}, debería ser 1.0")
    

    def test_zscore_raises_for_zero_std(self):
        utils = StatisticsUtils()
        arr_constant = [5, 5, 5]
        with self.assertRaises(ValueError) as context:
            utils.zscore(arr_constant)
        self.assertIn("Standard deviation is zero; z-scores are undefined", str(context.exception),"Debería lanzar ValueError para desviación estándar cero")

    def test_min_max_scale_maps_to_zero_one_range(self):
        utils = StatisticsUtils()
        arr = [2, 4, 6]
        result = utils.min_max_scale(arr)
        min_result = np.min(result)
        self.assertAlmostEqual(min_result, 0.0, places=10,msg=f"El valor mínimo escalado es {min_result}, debería ser 0.0")
        max_result = np.max(result)
        self.assertAlmostEqual(max_result, 1.0, places=10,msg=f"El valor máximo escalado es {max_result}, debería ser 1.0")
        expected = np.array([0.0, 0.5, 1.0])
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7,err_msg="Los valores escalados no son correctos")
    def test_min_max_scale_raises_for_constant_values(self):
        utils = StatisticsUtils()
        arr_constant = [3, 3, 3]
        with self.assertRaises(ValueError) as context:
            utils.min_max_scale(arr_constant)
        self.assertIn("All values are equal; min-max scaling is undefined", str(context.exception), "Debería lanzar ValueError para secuencias constantes")


if __name__ == "__main__":
    unittest.main()
