import math
from mpv_parakeet.srt_utils import format_time_srt

def test_rounding_carries_seconds():
    assert format_time_srt(1.9996) == "00:00:02,000"

def test_negative_value_clamped():
    assert format_time_srt(-5.0) == "00:00:00,000"

def test_nan_value_clamped():
    assert format_time_srt(math.nan) == "00:00:00,000"

def test_infinite_value_clamped():
    assert format_time_srt(math.inf) == "00:00:00,000"
