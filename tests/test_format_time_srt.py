import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from srt_utils import format_time_srt

class TestFormatTimeSrt(unittest.TestCase):
    def test_rounding_carries_seconds(self):
        self.assertEqual(format_time_srt(1.9996), "00:00:02,000")
    def test_negative_value_clamped(self):
        self.assertEqual(format_time_srt(-5.0), "00:00:00,000")
    def test_nan_value_clamped(self):
        self.assertEqual(format_time_srt(float('nan')), "00:00:00,000")
    def test_infinite_value_clamped(self):
        self.assertEqual(format_time_srt(float('inf')), "00:00:00,000")

if __name__ == '__main__':
    unittest.main()
