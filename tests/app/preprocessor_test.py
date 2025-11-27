import unittest

from lukefi.metsi.app.preprocessor import slice_list_by_percentage, slice_list_by_size


class TestPreprocessor(unittest.TestCase):

    def test_slice_by_percentage(self):
        test_data = list(range(100))
        result = slice_list_by_percentage(test_data, 33)
        self.assertListEqual([list(range(33)), list(range(33, 66)), list(range(66, 99)), [99]], result)

    def test_slice_by_percentage_2(self):
        test_data = list(range(100))
        result = slice_list_by_percentage(test_data, 50)
        self.assertListEqual([list(range(50)), list(range(50, 100))], result)

    def test_slice_by_size(self):
        test_data = list(range(100))
        result = slice_list_by_size(test_data, 21)
        self.assertListEqual(
            [list(range(21)), list(range(21, 42)), list(range(42, 63)), list(range(63, 84)), list(range(84, 100))],
            result
        )
