import os
import sys
import streamlit as st
import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils import load_data_from_data_base, generate_model_dict, split_forecasts_by_model, create_line_plot,create_bar_chart,combine_backtest_forecast


st.markdown(
    """
    <style>
        .block-container {
            text-align: left;
            max-width: 1200px;
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Thaink² Technical test")
st.write("This app connects to a database, generates forecasts, and displays interactive results.")

variable_descriptions = {
    "pce": "Personal Consumption Expenditures (in billions of dollars).",
    "pop": "Total population (in thousands).",
    "psavert": "Personal savings rate (percentage of disposable income).",
    "uempmed": "Median duration of unemployment (in weeks).",
    "unemploy": "Number of unemployed individuals (in thousands)."
}


@st.cache_data
def get_data():
    st.write("Connecting to the database...")
    df = load_data_from_data_base()
    return df

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if st.button("Load Data from Database") or st.session_state.data_loaded:
    if not st.session_state.data_loaded:
        st.session_state.data = get_data()
        st.session_state.data_loaded = True

    data = st.session_state.data
    st.write("Data Loaded:")
    st.dataframe(data)

    dropdown_options = [f"{key} - {value}" for key, value in variable_descriptions.items()]
    selected_option = st.selectbox("Select a variable for forecasting:", dropdown_options)
    variable = selected_option.split(" - ")[0]
    fcast_horizon = st.slider("Select forecasting horizon (months):", 1, 90, 12)
    model_options = ["xgboost", "arima", "random_forest"]
    selected_models = st.multiselect("Select forecasting models for comparison:", model_options, default=["xgboost"])

    if st.button("Generate Forecast"):
        if not selected_models:
            st.error("Please select at least one forecasting model before proceeding.")
        else:
            with st.spinner("calculating forecasts, please wait..."):
                # Filter data for the selected variable
                filtered_data = data[data['variable'] == variable]
                filtered_data['date'] = pd.to_datetime(filtered_data['date'])

                backtest_data = filtered_data.iloc[:-fcast_horizon]

                model_dict = generate_model_dict(selected_models)

                # Combine backtest and forecast for original values
                forecast_original_df = combine_backtest_forecast(
                    filtered_data, backtest_data, fcast_horizon, "value", selected_models
                )

                # Combine backtest and forecast for normalized values
                forecast_normalized_df = combine_backtest_forecast(
                    filtered_data, backtest_data, fcast_horizon, "value01", selected_models
                )

                min_value, max_value = filtered_data['value'].min(), filtered_data['value'].max()
                forecast_normalized_df['value'] = (
                                                          forecast_normalized_df['value01'] * (max_value - min_value)
                                                  ) + min_value

                forecasts_original = split_forecasts_by_model(forecast_original_df, model_dict)
                forecasts_normalized = split_forecasts_by_model(forecast_normalized_df, model_dict)

                zoom_range = [filtered_data['date'].iloc[-20], forecast_original_df['date'].iloc[-1]]

                # Create line plots
                fig_original = create_line_plot(
                    filtered_data, forecasts_original,
                    f"Original Values Forecast Comparison with Backtest for {variable}",
                    "Value", zoom_range
                )
                fig_normalized = create_line_plot(
                    filtered_data, forecasts_normalized,
                    f"Normalized Values Forecast Comparison with Backtest for {variable}",
                    "Value", zoom_range
                )

                # Create bar charts
                bar_chart_original = create_bar_chart(
                    filtered_data, forecasts_original,
                    f"Original Values Bar Chart for {variable}",
                    "Value", zoom_range
                )
                bar_chart_normalized = create_bar_chart(
                    filtered_data, forecasts_normalized,
                    f"Normalized Values Bar Chart for {variable}",
                    "Value", zoom_range
                )

                # Display line charts
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_original)
                with col2:
                    st.plotly_chart(fig_normalized)

                # Display bar charts
                st.write("Bar Charts:")
                bar_col1, bar_col2 = st.columns(2)
                with bar_col1:
                    st.plotly_chart(bar_chart_original, use_container_width=True)
                with bar_col2:
                    st.plotly_chart(bar_chart_normalized, use_container_width=True)