import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================================
# PLOT FONT SETTINGS
# =====================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10
})

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(layout="wide")

# Background
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f7fb;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================
# LOAD FILES
# =====================================
model = joblib.load("temperature_forecast_model.pkl")
features = joblib.load("temperature_model_features.pkl")
df = pd.read_csv("temperature_data.csv", index_col=0, parse_dates=True)

# =====================================
# FORECAST
# =====================================
last_row = df.iloc[-1]
X_last = last_row[features].values.reshape(1, -1)

forecast_temp = model.predict(X_last)[0]

last_time = df.index[-1]
next_time = last_time + pd.Timedelta(hours=1)

# =====================================
# HEAT INDEX FUNCTION
# =====================================
def heat_index(T, RH):
    return (
        -8.784695 + 1.61139411*T + 2.338549*RH
        - 0.14611605*T*RH - 0.012308094*T**2
        - 0.016424828*RH**2 + 0.002211732*T**2*RH
        + 0.00072546*T*RH**2 - 0.000003582*T**2*RH**2
    )

if "Heat_Index" not in df.columns:
    df["Heat_Index"] = heat_index(
        df["Temperature"],
        df["Relative Humidity"]
    )

# =====================================
# TITLE
# =====================================
st.markdown("# 🌡 HVAC STRESS – UPCOMING HOUR")

# =====================================
# INPUT SECTION
# =====================================
col_l, col_m, col_r = st.columns([1,3,1])

with col_m:
    st.markdown("### ⚙️ Configure Analysis")

    location = st.selectbox(
        "📍 Select Location",
        ["Kochi", "Chennai", "Mumbai", "Delhi"]
    )

    threshold_option = st.selectbox(
        "🔥 Select Threshold Method",
        [
            "Default (90th percentile)",
            "High Sensitivity (85th percentile)",
            "Low Sensitivity (95th percentile)",
            "Custom Value"
        ]
    )

    if threshold_option == "Custom Value":
        custom_val = st.number_input("Enter threshold (°C)", value=35.0)
    else:
        custom_val = None

    run_button = st.button("🔍 Check HVAC Stress Risk")

st.markdown("---")

# =====================================
# THRESHOLD
# =====================================
if custom_val is not None:
    HI_threshold = custom_val
else:
    if threshold_option == "Default (90th percentile)":
        HI_threshold = df["Heat_Index"].quantile(0.90)
    elif threshold_option == "High Sensitivity (85th percentile)":
        HI_threshold = df["Heat_Index"].quantile(0.85)
    else:
        HI_threshold = df["Heat_Index"].quantile(0.95)

TEMP_threshold = df["Temperature"].quantile(0.90)

RH_last = df["Relative Humidity"].iloc[-1]
HI_forecast = heat_index(forecast_temp, RH_last)

# =====================================
# OUTPUT
# =====================================
if run_button:

    st.markdown(f"### 📅 {next_time.strftime('%d-%b-%Y %H:%M')}")

    col1, col2 = st.columns([2,1])

    # ================= LEFT PANEL =================
    with col1:
        st.markdown("## 🌡 Forecast Heat Index (°C)")
        st.markdown(f"# {HI_forecast:.2f} °C")

        st.markdown(f"### HI Threshold: {HI_threshold:.2f} °C")

        if HI_forecast > HI_threshold:
            st.error("🚨 HVAC STRESS ALERT")
        else:
            st.success("✅ Normal Conditions")

        st.markdown(f"### Forecast Temperature: {forecast_temp:.2f} °C")

        # ✅ SPACING FIX (THIS IS THE KEY CHANGE)
        st.markdown("<div style='margin-top: 70px;'></div>", unsafe_allow_html=True)

    # ================= 24H PLOT =================
    with col2:
        fig1, ax = plt.subplots(figsize=(6,3.8))

        ax.plot(df.index[-24:], df["Heat_Index"].iloc[-24:], label="Observed HI")
        ax.axhline(HI_threshold, linestyle=":", label="Threshold")

        ax.scatter(next_time, HI_forecast, s=90, label="Forecast HI")

        ax.plot(
            [df.index[-1], next_time],
            [df["Heat_Index"].iloc[-1], HI_forecast],
            linestyle="--"
        )

        ax.annotate(
            next_time.strftime('%d-%b %H:%M'),
            (next_time, HI_forecast),
            textcoords="offset points",
            xytext=(0,18),
            ha='center',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        ax.set_xlim(df.index[-24], next_time + pd.Timedelta(hours=2))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%H:%M'))

        ax.set_title("24-Hour Heat Index + Forecast")
        ax.set_ylabel("Heat Index (°C)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

        st.pyplot(fig1)

    # ================= 7 DAY HI =================
    fig2, ax2 = plt.subplots(figsize=(12,2.2))

    y_hi = df["Heat_Index"].iloc[-7*24:]

    ax2.plot(df.index[-7*24:], y_hi, label="Observed HI")
    ax2.axhline(HI_threshold, linestyle=":", label="Threshold")

    ax2.scatter(next_time, HI_forecast, s=60)

    ax2.plot(
        [df.index[-1], next_time],
        [y_hi.iloc[-1], HI_forecast],
        linestyle="--"
    )

    ax2.set_ylim(y_hi.min()-1, y_hi.max()+1)
    ax2.set_title("7-Day Heat Index + Forecast")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)

    # ================= 7 DAY TEMP =================
    fig3, ax3 = plt.subplots(figsize=(12,2.2))

    y_temp = df["Temperature"].iloc[-7*24:]

    ax3.plot(df.index[-7*24:], y_temp, label="Observed Temp")
    ax3.axhline(TEMP_threshold, linestyle=":", label="Threshold")

    ax3.scatter(next_time, forecast_temp, s=60)

    ax3.plot(
        [df.index[-1], next_time],
        [y_temp.iloc[-1], forecast_temp],
        linestyle="--"
    )

    ax3.set_ylim(y_temp.min()-1, y_temp.max()+1)
    ax3.set_title("7-Day Temperature + Forecast")
    ax3.legend(loc="upper left")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)
