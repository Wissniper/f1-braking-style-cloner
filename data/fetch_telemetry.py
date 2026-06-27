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
    u_expert: float # expected speed at the straight segment
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
        u_expert = u_raw[i]
        d_braking = d_braking_ref - s

        u_history.append(u_expert)
        if len(u_history) > A_RECENT_WINDOW:
            u_history.pop(0)
        a_recent = float(np.mean(u_history)) if len(u_history) == A_RECENT_WINDOW else 0.0

        segments.append(
            StraightSegment(
                s=float(s),
                v=float(v),
                u_expert=float(u_expert),
                d_braking=float(d_braking),
                v_norm=float(v / v_max),
                d_norm=float(d_braking / straight_length),
                a_recent=float(a_recent)
            )
        )
    
    return segments


def load_straight_data(year: int = 2023,
                       gp: str = 'Monza',
                       driver: str = 'HAM',
                       session_type: str = 'R') -> tuple[list[list[StraightSegment]], float, float]:
    """
    Fetch telemetry data via fastf1, extract straights, cache as .npz.

    Returns (laps, v_setpoint, v_corner):
      laps: list of laps, each a list of StraightSegment
      v_setpoint: mean straight speed across all laps (m/s)
      v_corner: mean braking-zone entry speed across all laps (m/s)
    """

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{year}_{gp}_{driver}_{session_type}.npz")

    ff1.Cache.enable_cache(CACHE_DIR)
    session = ff1.get_session(year, gp, session_type)
    session.load(telemetry=True)
    laps_df = session.laps.pick_driver(driver).pick_quicklaps()


    # Extract straights for each lap
    all_laps : list[list[StraightSegment]] = []
    v_setpoints, v_corners = [], []

    for _, lap in laps_df.iterlaps():
        try:
            tel = lap.get_telemetry()
        except Exception as e:
            print(f"Error fetching telemetry for lap {lap['LapNumber']}: {e}")
            continue
        segs = extract_straight(tel)
        if len(segs) < 10:
            continue
        all_laps.append(segs)
        v_setpoints.extend([s.v for s in segs])
        v_corners.extend([s.v for s in segs if s.d_braking < 50]) 

    # Calculate mean speeds
    v_setpoint = float(np.mean(v_setpoints)) if v_setpoints else 0.0
    v_corner = float(np.mean(v_corners)) if v_corners else 0.0

    return all_laps, v_setpoint, v_corner