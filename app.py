import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_data():
    df = pd.read_csv('data/data_stats.csv', index_col=0, parse_dates=True)
    df = df.dropna(subset=['pm10_mean'])
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def categorize_pm10(pm10):
    if pm10<=50:
        return st.success(f'Good - {pm10:.2f} µg/m³')
    elif pm10<=100:
        return st.info(f'Moderate - {pm10:.2f} µg/m³')
    elif pm10<=260:
        return st.warning(f'Unhealthy for sensitive groups - {pm10:.2f} µg/m³')
    elif pm10<=400:
        return st.error(f'Unhealthy - {pm10:.2f} µg/m³')
    elif pm10<=520:
        return st.error(f'Very unhealthy - {pm10:.2f} µg/m³')
    elif pm10>520:
        return st.error(f'Hazardous - {pm10:.2f} µg/m³')
data = load_data()
eval_pack = load_model('models/pm10_lgbm_eval.joblib')
model_eval = eval_pack['model']
features = eval_pack['features']

model_full = load_model('models/pm10_lgbm.joblib')['model']

tab1, tab2, tab3 = st.tabs(['Forecast', 'Test results', 'About'])

with tab1:
    date = st.date_input('Pick a date', format='YYYY-MM-DD')
    
    month_data = data[data.index.month==date.month]
    
    pm10_lag1 = st.number_input("Yesterday's PM10 (µg/m³)", value=month_data['pm10_lag1'].median(), min_value=0.0)
    min_temp, max_temp = st.slider(
        "Temperature range (°C)",
        min_value=-20.0,
        max_value=40.0,
        value=(month_data['temp_min'].median(), month_data['temp_max'].median()),
    )
    humidity_mean = st.number_input('Mean humidity (%)', value=month_data['humidity_mean'].median(), min_value=0.0, max_value=100.0)
    wind_speed_mean = st.number_input('Mean wind speed (km/h)', value=month_data['wind_speed_mean'].median(), min_value=0.0)
    deg = np.degrees(np.arctan2(month_data['wind_sin'].median(), month_data['wind_cos'].median())) % 360
    wind_dir = st.number_input('Wind direction (°, 0 = north)', min_value=0, max_value=360, value=int(deg))
    precipitation = st.number_input('Precipitation (mm)', value=month_data['precipitation_sum'].median(), min_value=0.0)
     
    temp_mean = (max_temp + min_temp)/2
    temp_dif = max_temp - min_temp
    wind_dir_rad = np.radians(wind_dir)
    wind_sin = np.sin(wind_dir_rad)
    wind_cos = np.cos(wind_dir_rad)
    U_wind = wind_speed_mean * wind_sin
     
    
    medians = month_data[['pressure_mean', 'cloud_cover_mean', 'wind_speed_max', 'wind_gusts_max', 'pm10_lag7', 'pm10_roll7', '3days_max']].median()
    medians['3days_max'] = max(pm10_lag1, medians['3days_max'])
    medians_df = pd.DataFrame(medians).T
    
    row = {
               'pm10_lag1' : pm10_lag1,
               'max_temp' : max_temp,
               'min_temp' : min_temp,
               'humidity_mean' : humidity_mean,
               'wind_speed_mean' : wind_speed_mean,
               'wind_dir' : wind_dir,
               'precipitation_sum' : precipitation,
               'temp_mean' : temp_mean,
               'temp_dif' : temp_dif,
               'temp_min' : min_temp,
               'temp_max' : max_temp,
               'dayofyear' : pd.Timestamp(date).dayofyear,
               'wind_sin' : wind_sin,
               'wind_cos' : wind_cos,
               'U_wind' : U_wind
           }
    
    df = pd.concat([pd.DataFrame([row]), medians_df], axis=1)
    pred = np.expm1(model_full.predict(df[features])[0])
    categorize_pm10(pred)
    st.caption("Inputs not shown are filled with typical values for the selected month. This is a what-if demo, not a live forecast.")
    st.caption("Categories follow Turkey's National Air Quality Index (daily PM10).")
    

with tab2:
    test = data.loc['2025-10-01':]

    mask = test['pm10_lag1'].notna()
    baseline_mae = (test.loc[mask,'pm10_lag1'] - test.loc[mask, 'pm10_mean']).abs().mean()

    x_test = test.loc[mask, features]
    y_test = test.loc[mask, 'pm10_mean']

    pred_test = pd.Series(np.expm1(model_eval.predict(x_test)), index=y_test.index)
    model_mae = (y_test - pred_test).abs().mean()
    improvement = (model_mae - baseline_mae) / baseline_mae * 100


    st.header('Test results')
    st.caption("Model trained on data until Sep 2025, evaluated on Oct 2025 – Sep 2026.")

    col1, col2 = st.columns(2)
    col1.metric(
            label='Model MAE',
            value=f'{model_mae:.2f}',
            delta=f"{improvement:.0f}%",
            delta_color="inverse"
        )

    col2.metric(
            label="Baseline MAE (yesterday's value)",
            value=f'{baseline_mae:.2f}'
        )

    df_comparative = pd.DataFrame({
            'Actual':y_test,
            'Prediction':pred_test
        })

    st.subheader('Actual vs predicted daily PM10', text_alignment='center')
    st.line_chart(df_comparative, color=('blue', 'red'), y_label='PM10 (µg/m³)')
    
with tab3:
    st.header("About")

    st.subheader("What it does")
    st.markdown(
        "Predicts the next day's mean PM10 for the Ankara Bahçelievler "
        "air quality station."
    )

    st.subheader("Data")
    st.markdown("""
- Hourly air quality data from the Ministry of Environment monitoring network (Sep 2020 – Oct 2026).
- Hourly weather data from Open-Meteo.
- The weather data is reanalysis (what actually happened), not a forecast. In real use the model would get forecast weather, so the error would be a bit higher.
""")

    st.subheader("Model")
    st.markdown(f"""
- LightGBM with 19 features: yesterday's PM10, recent PM10 trend, the day's weather and the day of the year.
- Trained on log(PM10) and checked with 3-fold time-series cross-validation (one year per fold).
- On the test year (Oct 2025 – Sep 2026) the MAE is {model_mae:.2f} µg/m³, {abs(improvement):.0f}% lower than just using yesterday's value ({baseline_mae:.2f}).
""")

    st.subheader("What matters most")
    st.image("figures/feature_ablation.png")
    st.markdown(
    "Each bar shows how much the error goes up when that feature is removed. "
    "Yesterday's PM10 is by far the most important one (not shown, it would "
    "stretch the chart). After that come wind direction, the daily temperature "
    "range (max - min), wind speed and humidity."
)

    st.subheader("Limitations")
    ceiling = np.expm1(model_full.predict(data[features])).max()
    st.markdown(f"""
- It underestimates high pollution days. The model rarely predicts above ~{ceiling:.0f} µg/m³.
- It can't see sudden pollution episodes coming (e.g. dust transport), because nothing in the inputs signals them.
- Relative error is higher in winter than in summer.
""")
    
    