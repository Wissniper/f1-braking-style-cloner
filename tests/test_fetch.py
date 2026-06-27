# tests/test_fetch.py
import numpy as np
import pytest
from data.fetch_telemetry import StraightSegment, extract_straight
    

def test_straight_segment_fields():
    seg = StraightSegment(s=0.0, v=80.0, u_expert=0.5, d_braking=500.0,
                          v_norm=80.0/92.0, d_norm=500.0/1100.0, a_recent=0.0)
    assert seg.v_norm == pytest.approx(80.0 / 92.0)
    assert seg.d_norm == pytest.approx(500.0 / 1100.0)

def test_extract_straight_returns_segments():
    # synthetic telemetry: 100 Hz, 2-second straight, braking starts at t=1.5s
    import pandas as pd
    n = 200
    t = np.linspace(0, 2, n)
    speed = np.full(n, 80.0)
    throttle = np.ones(n)
    brake = np.zeros(n)
    brake[150:] = 0.5  # brake starts at index 150
    distance = np.linspace(0, 1100, n)
    tel = pd.DataFrame({'Speed': speed * 3.6,  # fastf1 gives km/h
                        'Throttle': throttle,
                        'Brake': brake,
                        'Distance': distance})
    segments = extract_straight(tel, v_max=92.0, straight_length=1100.0)
    assert len(segments) > 0
    assert all(0.0 <= seg.v_norm <= 1.0 for seg in segments)
    assert all(0.0 <= seg.d_norm <= 1.0 for seg in segments)
    # first segment has zero a_recent (edge padding)
    assert segments[0].a_recent == 0.0