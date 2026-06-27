from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import fastf1 as ff1

V_MAX = 92.0 # m/s, top straight speed at Monza
STRAIGHT_LENGTH = 1100.0 # m
BRAKE_THRESHOLD = 0.1 # m/s^2, threshold for braking detection
A_RECENT_WINDOW = 3 # seconds, window for recent acceleration calculation
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

@dataclass
class StraightSegment:
    s: float # distance along the straight
    v: float # speed at the straight segment
    u_expect: float # expected speed at the straight segment
    d_braking: float # distance to braking point
    v_norm: float # normalized speed (v / V_MAX)
    d_norm: float # normalized distance (d_braking / STRAIGHT_LENGTH)
    a_recent: float # recent acceleration (average over A_RECENT_WINDOW seconds)

def extract_straight(tel: pd.DataFrame,
                     v_max: float = V_MAX,
                     straight_length: float = STRAIGHT_LENGTH) -> list[StraightSegment]:
    """
    Extract StraightSegments from a single-lap telemetry DataFrame.

    tel must have cols: Speed (km/h), Throttle (0-1), Brake (0-1), Distance (m).
    Returns empty list if no braking point
    """

    speed_ms = np.array((tel['Speed'] / 3.6).values)  # Convert speed from km/h to m/s
    throttle = np.array(tel['Throttle'].values)
    brake = np.array(tel['Brake'].values)
    distance = np.array(tel['Distance'].values)

    # Locate braking points where the car is decelerating
    brake_idx = np.where(brake > BRAKE_THRESHOLD)[0]
    if len(brake_idx) == 0:
        return []  # No braking points found
    braking_start = brake_idx[0]

    # Signed acceleration from throttle and brake inputs
    # throttle maps to [0, u_max], brake maps to [u_min, 0], where u_max and u_min are the maximum and minimum acceleration values
    u_raw = throttle * 10.0 - brake * 50.0 # This is a simplified model; in reality, the mapping would depend on the car's characteristics

    d_braking_ref = distance[braking_start]

    segments = []
    u_history = []

    # Iterate over the telemetry data up to the braking point

    for i in range(braking_start):
        s = distance[i]
        v = speed_ms[i]
        u_expect = u_raw[i]
        d_braking = d_braking_ref - s

        u_history.append(u_expect)
        if len(u_history) > A_RECENT_WINDOW:
            u_history.pop(0)
        a_recent = float(np.mean(u_history)) if len(u_history) == A_RECENT_WINDOW else 0.0

        segments.append(
            StraightSegment(
                s=float(s),
                v=float(v),
                u_expect=float(u_expect),
                d_braking=float(d_braking),
                v_norm=float(v / v_max),
                d_norm=float(d_braking / straight_length),
                a_recent=float(a_recent)
            )
        )
    
    return segments


