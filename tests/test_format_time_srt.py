import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from srt_utils import format_start_ms, format_end_ms

class TestFormatMs(unittest.TestCase):
    def test_start_ceil(self):
        self.assertEqual(format_start_ms(1.9996), "00:00:02,000")

    def test_end_floor(self):
        self.assertEqual(format_end_ms(1.9994), "00:00:01,999")

    def test_clamp_invalid(self):
        self.assertEqual(format_start_ms(-5.0), "00:00:00,000")
        self.assertEqual(format_end_ms(float('nan')), "00:00:00,000")

if __name__ == '__main__':
    unittest.main()
