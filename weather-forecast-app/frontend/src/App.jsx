// App.jsx
import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import Plot from "react-plotly.js";

const API_ROOT = import.meta.env.VITE_API_ROOT || "http://localhost:8000/api";

const METRIC_LABELS = {
  LongTermTemp: "Long-term Temperature Trend (2000–2026)",
  AvgTemp: "Average Temperature (°C)",
  AvgMaxTemp: "Max Temperature (°C)",
  AvgMinTemp: "Min Temperature (°C)",
  Rainfall2024: "Rainfall (2024 - Testing)",
  Rainfall2025: "Rainfall (2025 - Forecast)",
  AQI2024: "Delhi AQI (2024 - XGBoost Test)",
  AQI2025: "Delhi AQI (2025 - Hybrid Forecast)"
};

// Delhi-only metrics
const DELHI_ONLY_METRICS = new Set([
  "Rainfall2024",
  "Rainfall2025",
  "AQI2024",
  "AQI2025",
  "LongTermTemp"
]);

export default function App() {
  const [cities, setCities] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [selectedCities, setSelectedCities] = useState(["Delhi", "Mumbai"]);
  const [selectedMetric, setSelectedMetric] = useState("AvgTemp");
  const [forecasts, setForecasts] = useState([]);
  const [accuracy, setAccuracy] = useState([]);
  const [actuals, setActuals] = useState({});
  const [loading, setLoading] = useState(false);

  // dropdown state + ref for custom metric dropdown
  const [openMetricDropdown, setOpenMetricDropdown] = useState(false);
  const metricDropdownRef = useRef(null);

  // Load cities + metrics
  useEffect(() => {
    axios.get(`${API_ROOT}/cities`).then(res => {
      const apiCities = res.data.cities || [];
      const m = res.data.metrics || [];

      setCities(apiCities);
      setMetrics(m);
      setSelectedMetric("LongTermTemp");

      if (apiCities.length > 0) {
        setSelectedCities(apiCities.slice(0, 2));
      }
    });

    axios.get(`${API_ROOT}/accuracy`).then(res => {
      setAccuracy(res.data.accuracy || []);
    }).catch(() => { /* ignore */ });

    axios.get(`${API_ROOT}/actuals`).then(res => {
      setActuals(res.data.actuals || {});
    }).catch(() => { /* ignore */ });
  }, []);

  // close dropdown on outside click
  useEffect(() => {
    function handleClick(e) {
      if (openMetricDropdown && metricDropdownRef.current && !metricDropdownRef.current.contains(e.target)) {
        setOpenMetricDropdown(false);
      }
    }
    window.addEventListener("mousedown", handleClick);
    return () => window.removeEventListener("mousedown", handleClick);
  }, [openMetricDropdown]);

  // Enforce Delhi-only lock for certain metrics
  useEffect(() => {
    if (DELHI_ONLY_METRICS.has(selectedMetric)) {
      const delhi = cities.includes("Delhi") ? "Delhi" : "Delhi";
      setSelectedCities([delhi]);
    } else {
      if (!selectedCities.length) {
        setSelectedCities(cities.slice(0, 2));
      } else {
        const valid = selectedCities.filter(c => cities.includes(c));
        if (!valid.length) {
          setSelectedCities(cities.slice(0, 2));
        } else setSelectedCities(valid);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedMetric, cities]);

  useEffect(() => {
    fetchForecast();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedMetric, selectedCities]);

  function toggleCity(city) {
    if (DELHI_ONLY_METRICS.has(selectedMetric)) return;
    setSelectedCities(prev =>
      prev.includes(city) ? prev.filter(c => c !== city) : [...prev, city]
    );
  }

  // Fetch forecasts based on metric
  async function fetchForecast() {
    setLoading(true);
    try {
      // RAINFALL 2024
      if (selectedMetric === "Rainfall2024") {
        const r = await axios.get(`${API_ROOT}/rainfall_2024`);
        setForecasts([
          {
            city: "Delhi",
            metric: "Rainfall2024",
            months: r.data.months || [],
            values: r.data.predicted || [],
            actual: r.data.actual || []
          }
        ]);
        return;
      }

      // RAINFALL 2025
      if (selectedMetric === "Rainfall2025") {
        const r = await axios.get(`${API_ROOT}/rainfall_2025`);
        setForecasts([
          {
            city: "Delhi",
            metric: "Rainfall2025",
            months: r.data.months || [],
            values: r.data.predicted || [],
            actual: r.data.actual || []
          }
        ]);
        return;
      }

      // LONG TERM TEMP
      if (selectedMetric === "LongTermTemp") {
        const r = await axios.get(`${API_ROOT}/long_term_temp`);
        setForecasts([
          {
            city: "Delhi",
            metric: "LongTermTemp",
            traces: r.data.traces || [],
            layout: r.data.layout || {}
          }
        ]);
        return;
      }

      // AQI TEST 2024
      if (selectedMetric === "AQI2024") {
        const r = await axios.get(`${API_ROOT}/aqi_2024_test`);
        setForecasts([
          {
            city: "Delhi",
            metric: "AQI2024",
            dates: r.data.dates || [],
            actual: r.data.actual || [],
            predicted: r.data.predicted || []
          }
        ]);
        return;
      }

      // AQI FORECAST 2025
      if (selectedMetric === "AQI2025") {
        const r = await axios.get(`${API_ROOT}/aqi_2025`);
        setForecasts([
          {
            city: "Delhi",
            metric: "AQI2025",
            months: r.data.months || [],
            actual: r.data.actual || [],
            predicted: r.data.predicted || []
          }
        ]);
        return;
      }

      // Normal temperature metrics
      const res = await axios.post(`${API_ROOT}/forecasts`, {
        cities: selectedCities,
        metrics: [selectedMetric],
        months: 12
      });
      setForecasts(res.data.forecasts || []);
    } catch (err) {
      console.error("Error:", err);
      setForecasts([]);
    } finally {
      setLoading(false);
    }
  }

  // Build Plotly traces
  const traces = [];
  forecasts.forEach(f => {
    const { city, metric, months, values, actual, error } = f;

    if (error) return;

    if (metric === "Rainfall2024") {
      traces.push(
        {
          x: months,
          y: values,
          name: "Predicted Rainfall 2024",
          mode: "lines+markers",
          line: { width: 3, dash: "dash" }
        },
        {
          x: months,
          y: actual,
          name: "Actual Rainfall 2024",
          mode: "lines+markers",
          line: { width: 3 }
        }
      );
      return;
    }

    if (metric === "Rainfall2025") {
      traces.push({
        x: months,
        y: values,
        name: "Predicted Rainfall 2025",
        mode: "lines+markers",
        line: { width: 3, dash: "dash" }
      });
      if (actual.some(v => v !== null)) {
        traces.push({
          x: months,
          y: actual,
          name: "Actual Rainfall 2025",
          mode: "lines+markers",
          line: { width: 3 }
        });
      }
      return;
    }

    if (metric === "AQI2024") {
      traces.push(
        {
          x: f.dates,
          y: f.predicted,
          name: "Predicted AQI 2024",
          mode: "lines+markers",
          line: { width: 3, dash: "dash" }
        },
        {
          x: f.dates,
          y: f.actual,
          name: "Actual AQI 2024",
          mode: "lines+markers",
          line: { width: 3 }
        }
      );
      return;
    }

    if (metric === "AQI2025") {
      traces.push({
        x: f.months,
        y: f.predicted,
        name: "Predicted AQI 2025",
        mode: "lines+markers",
        line: { width: 3, dash: "dash" }
      });

      if (f.actual.some(v => v !== null)) {
        traces.push({
          x: f.months,
          y: f.actual,
          name: "Actual AQI 2025",
          mode: "lines+markers",
          line: { width: 3 }
        });
      }

      f.errorTrend = f.actual.map((a, i) =>
        a !== null && f.predicted[i] !== null ? a - f.predicted[i] : null
      );

      return;
    }

    traces.push({
      x: months,
      y: values,
      name: `${city} — Predicted`,
      mode: "lines+markers",
      line: { width: 3, dash: "dash" }
    });

    if (actuals[city] && actuals[city][metric]) {
      traces.push({
        x: actuals[city].months,
        y: actuals[city][metric],
        name: `${city} — Actual`,
        mode: "lines+markers",
        line: { width: 3 }
      });
    }
  });

  // Accuracy cards
  const accCards = selectedCities.map(city => {
    const row = accuracy.find(
      r => r.city === city && r.metric === selectedMetric
    );
    return {
      city,
      accuracy: row ? row.accuracy_pct : null,
      mape: row ? row.mape_pct : null
    };
  });

  const cityOptions = DELHI_ONLY_METRICS.has(selectedMetric)
    ? ["Delhi"]
    : cities;

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-slate-900 via-slate-800 to-black text-white font-inter">

      {/* Sidebar */}
      <aside className="w-72 backdrop-blur-xl bg-white/5 border-r border-white/10 p-6 space-y-7 shadow-2xl">

        <div className="space-y-1">
          <h1 className="text-2xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Weather Dashboard
          </h1>
          <p className="text-slate-400 text-xs tracking-wide">
            ML Forecasts • SARIMAX • Hybrid AQI
          </p>
        </div>

        {/* City */}
        <div>
          <h3 className="text-sm uppercase text-slate-300 mb-2 tracking-wider">City</h3>
          <div className="space-y-2 max-h-56 overflow-auto custom-scroll">
            {cityOptions.map(city => (
              <label
                key={city}
                className="flex items-center gap-3 text-sm bg-white/5 px-3 py-2 rounded-xl hover:bg-white/10 transition"
              >
                <input
                  type="checkbox"
                  checked={selectedCities.includes(city)}
                  onChange={() => toggleCity(city)}
                  disabled={DELHI_ONLY_METRICS.has(selectedMetric)}
                  className="w-4 h-4 accent-cyan-400"
                />
                {city}
              </label>
            ))}
          </div>

          {DELHI_ONLY_METRICS.has(selectedMetric) && (
            <div className="text-xs mt-2 text-yellow-300 flex items-center gap-1">
              ⚠️ This metric is available for Delhi only.
            </div>
          )}
        </div>

        {/* Metric */}
        <div>
          <h3 className="text-sm uppercase text-slate-300 mt-3 mb-2 tracking-wider">Metric</h3>

          {/* CUSTOM METRIC DROPDOWN */}
          <div className="relative" ref={metricDropdownRef}>
            <button
              onClick={() => setOpenMetricDropdown(o => !o)}
              className="w-full bg-white/10 backdrop-blur-xl border border-white/10 p-3 rounded-xl 
                 text-sm flex justify-between items-center hover:bg-white/20 transition"
            >
              {METRIC_LABELS[selectedMetric] || selectedMetric}
              <span className="text-lg">▾</span>
            </button>

            {openMetricDropdown && (
              <div
                className="absolute left-0 right-0 mt-2 bg-slate-900 border border-white/10 
                   rounded-xl shadow-xl z-50 max-h-64 overflow-auto custom-scroll"
              >
                {[
                  "LongTermTemp",
                  ...metrics,
                  "Rainfall2024",
                  "Rainfall2025",
                  "AQI2024",
                  "AQI2025"
                ].map(m => (
                  <button
                    key={m}
                    onClick={() => {
                      setSelectedMetric(m);
                      setOpenMetricDropdown(false);
                    }}
                    className={`w-full text-left px-4 py-2 text-sm hover:bg-white/10 transition
                      ${m === selectedMetric ? "bg-cyan-500/20 text-cyan-300" : ""}
                    `}
                  >
                    {METRIC_LABELS[m] || m}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <button
          onClick={fetchForecast}
          className="w-full bg-gradient-to-r from-cyan-400 to-blue-500 text-black py-2 rounded-xl mt-2 font-semibold shadow-lg hover:shadow-cyan-500/30 transition-all"
        >
          Refresh
        </button>

        {/* Accuracy */}
        {!["LongTermTemp", "AQI2024", "AQI2025"].includes(selectedMetric) && (
          <div>
            <h3 className="text-xs text-slate-300 uppercase tracking-wider">
              Accuracy
            </h3>
            <div className="mt-3 space-y-3">
              {accCards.map(a => (
                <div
                  key={a.city}
                  className="bg-white/5 rounded-xl p-3 shadow-inner border border-white/10"
                >
                  <div className="text-sm font-semibold mb-1">{a.city}</div>
                  <div className="text-xs">Accuracy: {a.accuracy ?? "N/A"}%</div>
                  <div className="text-xs">MAPE: {a.mape ?? "N/A"}%</div>
                </div>
              ))}
            </div>
          </div>
        )}

      </aside>

      {/* Main */}
      <main className="flex-1 p-8 space-y-6">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
          {METRIC_LABELS[selectedMetric] || selectedMetric}
        </h2>

        <div className="text-slate-400 mb-4 text-sm">
          Select Cities & Metrics (Some metrics are Delhi-only)
        </div>

        {/* Graph container */}
        <div className="backdrop-blur-xl bg-white/5 p-6 rounded-2xl shadow-2xl border border-white/10">
          {traces.length === 0 &&
          !(forecasts.length > 0 && forecasts[0].metric === "LongTermTemp") ? (
            <div className="text-center py-20 text-slate-400 text-xl">
              {loading ? "Loading Predictions..." : "Select options to view forecast"}
            </div>
          ) : (
            <>
              {forecasts.length > 0 && forecasts[0].metric === "LongTermTemp" ? (
                <Plot
                  data={forecasts[0].traces}
                  layout={{
                    ...(forecasts[0].layout || {}),
                    autosize: true,
                    margin: { t: 80, b: 40 }
                  }}
                  useResizeHandler
                  style={{ width: "100%", height: "600px" }}
                />
              ) : (
                <Plot
                  data={traces}
                  layout={{
                    autosize: true,
                    legend: { orientation: "h" },
                    margin: { t: 20, b: 40 },
                    hovermode: "x unified",
                    xaxis: { title: "Month" },
                    yaxis: {
                      title: DELHI_ONLY_METRICS.has(selectedMetric)
                        ? "Rainfall (mm)"
                        : "Temperature (°C)"
                    }
                  }}
                  useResizeHandler
                  style={{ width: "100%", height: "600px" }}
                />
              )}
            </>
          )}
        </div>

        {/* AQI 2025 Error Trend */}
        {selectedMetric === "AQI2025" &&
          forecasts.length > 0 &&
          forecasts[0].errorTrend && (
            <div className="bg-white/5 backdrop-blur-xl p-6 rounded-2xl shadow-2xl border border-white/10">
              <Plot
                data={[
                  {
                    x: forecasts[0].months,
                    y: forecasts[0].errorTrend,
                    type: "bar",
                    marker: { color: "rgba(80,120,255,0.9)" },
                    name: "AQI Error"
                  }
                ]}
                layout={{
                  title: "AQI Prediction Error Trend (2025)",
                  autosize: true,
                  margin: { t: 40, b: 40 }
                }}
                useResizeHandler
                style={{ width: "100%", height: "450px" }}
              />
            </div>
          )}

        {/* Scientific Explanation — Rainfall */}
        {selectedMetric === "Rainfall2025" && (
          <div className="mmt-8 bg-gradient-to-br from-slate-800/60 to-black/40 backdrop-blur-xl border border-white/10 p-8 rounded-2xl shadow-2xl space-y-4">
            <h2 className="text-xl font-bold text-cyan-400 mb-4">
              🌧️ Scientific Explanation for Delhi's Extreme Rainfall (July–August 2025)
            </h2>

            <p className="leading-relaxed text-slate-300 mb-4">
              During the <b>2025 monsoon season</b>, Delhi experienced exceptionally high rainfall
              in July and August due to an <b>unusually persistent monsoon trough</b> positioned
              farther north than usual. This system continually drew
              <b> moisture-laden southwesterly winds</b> from both the Arabian Sea and the Bay of Bengal,
              causing repeated heavy rain spells rather than short, isolated showers.
            </p>

            <p className="leading-relaxed text-slate-300 mb-4">
              At the same time, a <b>positive Indian Ocean Dipole (IOD)</b> phase and a weakening
              <b> El Niño</b> enhanced the monsoon’s strength over northern India, supplying
              additional moisture to the region. Localized factors such as
              <b> urban heat island effects</b> and <b>convective cloud formation</b>
              further amplified rainfall intensity over Delhi.
            </p>

            <p className="leading-relaxed text-slate-300">
              As a result, rainfall totals surged far beyond climatological norms, producing the
              large difference between our model’s predicted and observed values. The SARIMAX model,
              trained on historical averages, could not fully capture these
              <b> dynamic, multi-system-driven anomalies</b> that made 2025 an extraordinary rainfall year for Delhi.

            </p>
          </div>
        )}

        {/* Scientific Explanation for AQI 2025 */}
        {selectedMetric === "AQI2025" && (
          <div className="mmt-8 bg-gradient-to-br from-slate-800/60 to-black/40 backdrop-blur-xl border border-white/10 p-8 rounded-2xl shadow-2xl space-y-4">
            <h2 className="text-xl font-bold text-cyan-400 mb-4">
              🌫️ Why Predicted AQI ≠ Actual AQI in Delhi (2025)
            </h2>

            <p className="leading-relaxed text-slate-300 mb-4">
              Delhi’s AQI is strongly influenced by sudden, short-term pollution events that 
              machine learning models cannot predict in advance. Below are the scientific reasons 
              behind the mismatch between predicted and actual AQI:
            </p>

            <h3 className="text-lg font-semibold text-orange-300 mt-4">1. Stubble Burning (Oct–Nov Spike)</h3>
            <ul className="list-disc ml-6 text-slate-300 mb-4">
              <li>Crop residue burning in Punjab & Haryana releases massive PM2.5/PM10.</li>
              <li>Smoke reaches Delhi within 2–3 days depending on winds.</li>
              <li>Extent of burning varies yearly based on rainfall & crop cycles.</li>
              <li>Models cannot predict future burning → unpredictable spikes.</li>
            </ul>

            <h3 className="text-lg font-semibold text-orange-300 mt-4">2. Diwali Firecrackers</h3>
            <ul className="list-disc ml-6 text-slate-300 mb-4">
              <li>Crackers push AQI from ~300 to 600+ in a single night.</li>
              <li>Diwali date changes each year (lunar calendar).</li>
              <li>Actual cracker usage varies based on bans & public behavior.</li>
              <li>Models cannot estimate Diwali pollution impact.</li>
            </ul>

            <h3 className="text-lg font-semibold text-orange-300 mt-4">3. Meteorology Variability</h3>
            <ul className="list-disc ml-6 text-slate-300 mb-4">
              <li>Wind speed/direction</li>
              <li>Temperature inversion</li>
              <li>Humidity & mixing height</li>
              <li>Rainfall events</li>
              <li>Weather cannot be predicted months ahead → AQI uncertainty.</li>
            </ul>

            <h3 className="text-lg font-semibold text-orange-300 mt-4">4. Traffic & Construction Dust</h3>
            <p className="text-slate-300 mb-4">
              Daily changes in traffic, road closures, construction bans, and holidays 
              cause unpredictable PM10 increases the model cannot foresee.
            </p>

            <h3 className="text-lg font-semibold text-orange-300 mt-4">5. GRAP & Government Actions</h3>
            <p className="text-slate-300 mb-4">
              Emergency actions like school closures, construction bans, and truck bans 
              vary every year. Models cannot anticipate these interventions.
            </p>

            <h3 className="text-lg font-semibold text-orange-300 mt-4">6. Dust Storms & Wildfires</h3>
            <p className="text-slate-300 mb-4">
              Dust storms from Rajasthan and wildfire smoke can raise PM levels massively. 
              These events are sudden and unpredictable.
            </p>

            <h3 className="text-lg font-semibold text-orange-300 mt-4">7. Climate Change Effects</h3>
            <ul className="list-disc ml-6 text-slate-300 mb-4">
              <li>Delayed monsoon</li>
              <li>Longer summers</li>
              <li>Reduced winter rains</li>
              <li>These change pollution dispersion patterns unpredictably.</li>
            </ul>

            <h3 className="text-lg font-semibold text-orange-300 mt-4">8. Machine Learning Limitations</h3>
            <p className="text-slate-300 mb-4">
              The model uses only past AQI patterns and month seasonality. 
              But actual AQI depends on many external factors like wind, fires, 
              humidity, festival traffic, and crop burning—which the model cannot know in advance.
            </p>

            <h3 className="text-lg font-bold text-green-300 mt-6">⭐ SUMMARY</h3>
            <p className="leading-relaxed text-slate-300">
              Delhi AQI is driven by extreme, episodic pollution events that do not repeat 
              in a predictable manner each year. Machine learning can capture seasonal trends 
              but cannot foresee sudden events like stubble burning, Diwali firecrackers, 
              dust storms, or emergency pollution-control actions. 
              Therefore, a difference between predicted and actual AQI is scientifically expected.
            </p>
          </div>
        )}


      </main>
    </div>
  );
}
