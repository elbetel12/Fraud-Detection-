# tests/test_data_processor.py
import unittest
import pandas as pd
from src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.sample_data = pd.DataFrame({
            'amount': [100, 200, 300],
            'is_fraud': [0, 1, 0]
        })
    
    def test_clean_data_shape(self):
        cleaned = self.processor.clean_data(self.sample_data)
        self.assertEqual(cleaned.shape, (3, 2))