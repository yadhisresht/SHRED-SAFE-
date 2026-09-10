# PoultryGuard

**Smart Poultry Shed Protection & Early-Warning System**

An ESP32-based shed monitoring system that predicts heat stress instead of reacting to it. A small
machine-learning model runs on the microcontroller itself, forecasts shed conditions 30 minutes
ahead, and starts ventilation *before* the danger threshold is crossed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/PoultryGuard_ML_Prototype.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-ESP32-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The problem

Poultry birds are kept in closed sheds at high density, where heat builds up slowly and does real
damage: reduced feed intake, slower growth, and mortality.

Existing smart sheds handle this **reactively**. A sensor watches the temperature, and once it
crosses a limit — say 32 °C — the fan switches on and the farmer gets an alert. But a shed is a
building with thermal mass. By the time the fan starts, the birds have already been in heat stress,
and cooling the space takes time.

In short: **the alarm and the emergency arrive at the same moment.**

The deeper cause is that threshold logic looks at one reading in isolation. To such a system,
31 °C and falling is indistinguishable from 31 °C and climbing fast. It has no sense of direction.

## The solution

Add one step to the pipeline:

```
Monitor  →  PREDICT  →  Detect  →  Alert  →  Auto-Respond  →  Protect Birds
```

Instead of asking *"is it too hot?"*, the system asks *"will it be too hot in 30 minutes?"*

It reads the last 20 minutes of sensor history, forecasts temperature and humidity half an hour
out, converts that into a Temperature-Humidity Index risk level, and acts on the forecast. The
fan starts while the shed is still safe, and the farmer is warned while there is still time to do
something.

The entire model is **six floating-point numbers** stored in the firmware. No cloud, no TensorFlow
Lite, no extra hardware.

---

## Results

Validated on 10 days of simulated conditions the model had never seen.

| Metric | Result |
|---|---|
| Temperature forecast error (MAE, +30 min) | **0.38 °C** |
| R² | **0.987** |
| Improvement over persistence baseline | **27%** |
| Heat events detected in advance | **100%** |
| Mean early warning | **~70 minutes** |
| Alarm precision | **95%** |
| Heat-stress exposure reduction (closed loop) | **10%** |
| Additional fan runtime cost | **+4%** |
| On-device model footprint | **~24 bytes** |

![Reactive vs predictive control](docs/results.png)

*Same shed, same weather, same sensors. The only difference is that one system can see what is
coming.*

---

## How the model works

**Inputs (5):** temperature now, at −10 min, at −20 min; humidity now and at −20 min.
The three temperature lags let the model infer the trend on its own, so no explicit slope feature
is needed (and none is added, since it would be perfectly collinear with the lags).

**Output:** temperature and humidity at +30 minutes, via two separate linear regressions.

**Why linear regression?** The shed's thermal response is close to first-order, and a linear model
captures it with R² ≈ 0.99. A neural network would not fit better here but would need roughly a
hundred times the memory and could not run on an ESP32 without an ML runtime. Choosing the smallest
model that solves the problem is a deliberate engineering decision.

**On the device**, inference is five multiplications and an addition:

```c
float predictTemp(float t0, float t10, float t20, float h0, float h20) {
  return W[0]*t0 + W[1]*t10 + W[2]*t20 + W[3]*h0 + W[4]*h20 + B;
}

float computeTHI(float t, float rh) {
  return 0.8f * t + (rh / 100.0f) * (t - 14.4f) + 46.4f;
}
```

The notebook prints this file with the trained coefficients already filled in.

---

## Validation methodology

The evaluation is deliberately stricter than an accuracy number, and this is the part worth
reading if you are assessing the work:

- **Chronological train/test split** (first 21 days train, last 9 test). No shuffling — shuffling a
  time series leaks future readings into training and inflates the score.
- **Persistence baseline.** The model is measured against *"assume the temperature will not
  change"*, which is what you get with no model at all. Beating it by 27% is what shows the model
  learned real dynamics rather than echoing the current reading back.
- **Held-out weather.** Early-warning performance is measured on a separate simulation run with a
  different random seed.
- **Detection metrics, not just regression metrics.** Recall, mean lead time and precision on
  actual heat events — because a forecast error in °C does not tell you whether the system would
  have caught anything.
- **Closed-loop A/B test.** The same shed, same weather and same seed, run twice with only the
  control logic swapped, measuring heat-stress exposure against fan runtime cost.

---

## Repository structure

```
├── notebooks/
│   └── PoultryGuard_ML_Prototype.ipynb   # simulator, training, validation, C export
├── firmware/
│   └── poultryguard.ino                  # ESP32 sketch with the trained coefficients
├── data/
│   └── poultry_shed_synthetic.csv        # 8,640 readings, generated by the notebook
├── docs/
│   ├── results.png                       # reactive vs predictive comparison
│   └── PoultryGuard_Deck.pptx            # project presentation
└── README.md
```

---

## Quick start

No hardware required. The notebook runs end to end in about a minute.

1. Open `notebooks/PoultryGuard_ML_Prototype.ipynb` in Google Colab (badge above).
2. **Runtime → Run all.** Nothing to install; NumPy, pandas, matplotlib and scikit-learn are
   preinstalled.
3. The notebook will:
   - build a physics-based virtual shed and generate 30 days of sensor data
   - train the forecasting models and score them against the baseline
   - measure early-warning recall, lead time and precision
   - run the reactive and predictive controllers head to head
   - print a sample mobile alert and a mock dashboard
   - export ready-to-paste Arduino C with the trained coefficients

To train on real data instead, replace the simulator call with your own CSV:

```python
data = pd.read_csv("your_shed_readings.csv")
```

Everything downstream runs unchanged.

---

## Hardware

| Component | Purpose |
|---|---|
| ESP32 dev board | Controller, with built-in Wi-Fi |
| DHT22 | Temperature and humidity |
| MQ-135 | Air quality (ammonia proxy) |
| Soil moisture probe | Litter moisture |
| Water leakage strip | Leak detection |
| Relay + mini DC fan | Automatic ventilation |

Indicative build cost: **₹2,000 – ₹2,500 per shed**, one-time, with no recurring fees.

**Control parameters:** fan on at 32 °C, off at 30.5 °C (hysteresis to prevent relay chattering),
with a 15-minute minimum run time. THI bands: 78 warning, 82 critical.

---

## Limitations

Stated plainly, because they matter:

- **The dataset is synthetic.** It is generated by a physics model written for this project, not
  logged from a live shed. This demonstrates that the method and pipeline work end to end; it does
  **not** establish accuracy on any particular farm.
- **No sensor fault handling yet.** A failed or frozen sensor would currently feed bad values
  straight into the model.
- **The 32 °C threshold and THI bands are configurable defaults**, not values calibrated for a
  specific breed, bird age or region.

## Roadmap

- [ ] Validate on real logged data from a live shed (2–3 weeks of readings)
- [ ] Sensor plausibility checks with fallback to rule-based control
- [ ] Extend forecasting to litter moisture and ammonia
- [ ] Camera-based dead-bird detection
- [ ] Multi-shed dashboard
- [ ] On-device retraining as more data accumulates

---

## Team

Guhan Karthikeyan · Dhivya Shri · Yadhisresht Harikrishnan

## License

MIT
