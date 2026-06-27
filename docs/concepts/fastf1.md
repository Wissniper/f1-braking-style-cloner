# FastF1

## What it is

FastF1 is a Python library that downloads and parses official F1 timing and telemetry data. It gives you per-lap, per-driver telemetry at up to ~50 Hz — speed, throttle, brake, distance, gear, GPS position, and more.

Free, no account required.

## What data is available

| Channel | Unit | What it means |
|---|---|---|
| Speed | km/h | Car speed |
| Throttle | 0–1 | Throttle pedal position |
| Brake | 0–1 | Brake pedal pressure (0 = not braking) |
| Distance | m | Distance along the lap |
| nGear | int | Current gear |
| RPM | rpm | Engine speed |
| DRS | 0/1 | DRS open/closed |
| X, Y, Z | m | GPS coordinates |

Telemetry is sampled at ~50 Hz (one reading every ~0.02 s), aligned to lap timing.

## How to use it

```python
import fastf1

fastf1.Cache.enable_cache('data/cache')          # cache downloads locally

session = fastf1.get_session(2023, 'Monza', 'R') # year, circuit, session type
session.load(telemetry=True)

laps = session.laps.pick_driver('HAM').pick_quicklaps()

for _, lap in laps.iterlaps():
    tel = lap.get_telemetry()   # DataFrame with all channels
    speed  = tel['Speed']       # km/h
    brake  = tel['Brake']       # 0–1
    dist   = tel['Distance']    # m along lap
```

Session types: `'R'` = Race, `'Q'` = Qualifying, `'FP1/2/3'` = Practice.

`pick_quicklaps()` filters out laps with pit stops, safety cars, or anomalous lap times — leaves only clean representative laps.

## Why Monza 2023

Monza has the longest full-throttle straight in F1 (~1.1 km, T1 exit to T1 braking zone). This means:
- The braking event at the end is clear and unambiguous
- The straight is long enough to show the full lifting and braking profile
- Behavior is consistent across laps — no tricky chicanes or variable wind effects

## In this project

We detect the braking point per lap using `Brake > 0.1`, average it across laps for a stable reference, then extract the segment from T1 exit to that braking point.

Speed is converted km/h → m/s (`/ 3.6`). A synthetic `u_expert` is fused from Throttle and Brake:

```python
u_expert ≈ throttle * 10.0 - brake * 50.0   # approximate m/s²
```

This is an approximation — real acceleration depends on gear, engine map, tyre state — but it's consistent enough for behavioral cloning at this scale.

## Caching

`fastf1.Cache.enable_cache(path)` stores downloads locally. First run: ~30 seconds, ~50 MB. Subsequent runs: instant. The cache directory is git-ignored.
