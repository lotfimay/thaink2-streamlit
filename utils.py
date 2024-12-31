import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
from th2analytics_py.th2analytics.forecasting import ForecastingAPI
import plotly.graph_objects as go

# Load environment variables from a .env file
load_dotenv()

# Database connection parameters loaded from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT"),
}

# Initialize the Forecasting API with the base URL and API token
api = ForecastingAPI(
    base_url=os.getenv("API_URL"),
    api_token=os.getenv("THAINK2_API_TOKEN")
)

def load_data_from_data_base():
    """
    Connects to a PostgreSQL database and retrieves data from the `sales_economics` table.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the `sales_economics` table.
    """
    engine = create_engine(
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    query = "SELECT * FROM sales_economics;"
    df = pd.read_sql_query(query, engine)
    return df

def get_api_forecasts(actuals, fcast_horizon=30, group_target=None, target_var="value", date_var="date", models_list=["xgboost"], save_as_df=True):
    """
    Fetches forecasts from the Forecasting API.

    Args:
        actuals (pd.DataFrame): Historical data used as input for forecasting.
        fcast_horizon (int): The number of time points to forecast.
        group_target (str, optional): Grouping variable, if applicable. Defaults to None.
        target_var (str): The target variable for forecasting.
        date_var (str): The date variable in the input data.
        models_list (list): A list of forecasting models to use.
        save_as_df (bool): Whether to return the result as a DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the forecasted values.
    """
    forecasts = api.th2forecast_api(
        actuals=actuals,
        fcast_horizon=fcast_horizon,
        group_target=group_target,
        target_var=target_var,
        date_var=date_var,
        models_list=models_list
    )
    result = pd.json_normalize(forecasts)
    result = result.rename(columns={'.index': 'date', '.value': target_var})
    result['date'] = pd.to_datetime(result['date'])
    return result

def combine_backtest_forecast(filtered_data, backtest_data, fcast_horizon, target_var, models):
    """
    Combines backtest and forecast data into a single DataFrame.

    Args:
        filtered_data (pd.DataFrame): Filtered historical data.
        backtest_data (pd.DataFrame): Data for backtesting.
        fcast_horizon (int): The forecast horizon.
        target_var (str): The target variable for forecasting.
        models (list): List of models to use for forecasting.

    Returns:
        pd.DataFrame: A combined DataFrame of backtest and forecast data.
    """
    backtest_df = get_api_forecasts(actuals=backtest_data, fcast_horizon=fcast_horizon, target_var=target_var, date_var="date", models_list=models)
    forecast_df = get_api_forecasts(actuals=filtered_data, fcast_horizon=fcast_horizon, target_var=target_var, date_var="date", models_list=models)
    return pd.concat([backtest_df, forecast_df])

def generate_model_dict(selected_models):
    """
    Generates a dictionary mapping model IDs to model names.

    Args:
        selected_models (list): List of selected model names.

    Returns:
        dict: A dictionary where keys are model IDs and values are model names.
    """
    return {index + 1: model for index, model in enumerate(selected_models)}

def split_forecasts_by_model(forecast_df, model_dict):
    """
    Splits forecast data by model using a model dictionary.

    Args:
        forecast_df (pd.DataFrame): The DataFrame containing forecast data.
        model_dict (dict): Dictionary mapping model IDs to model names.

    Returns:
        dict: A dictionary where keys are model names and values are DataFrames of forecast data for each model.
    """
    return {
        model_dict[model_id]: forecast_df[forecast_df['.model_id'] == model_id]
        for model_id in model_dict
    }

def create_line_plot(filtered_data, forecasts, title, yaxis_title, zoom_range):
    """
    Creates a line plot for visualizing actuals and forecast data.

    Args:
        filtered_data (pd.DataFrame): Historical data.
        forecasts (dict): Dictionary of forecast data by model.
        title (str): Title of the plot.
        yaxis_title (str): Label for the y-axis.
        zoom_range (list): Range for the x-axis zoom.

    Returns:
        plotly.graph_objects.Figure: The generated line plot.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['value'],
        mode='lines+markers',
        name='Actuals',
        marker=dict(color='blue', size=6),
        line=dict(color='blue', width=2)
    ))
    for model, model_data in forecasts.items():
        fig.add_trace(go.Scatter(
            x=model_data['date'],
            y=model_data['value'],
            mode='lines+markers',
            name=f'{model}',
            marker=dict(size=5),
            line=dict(width=2)
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        template="plotly_white",
        showlegend=True,
        xaxis=dict(range=zoom_range)
    )
    return fig

def create_bar_chart(filtered_data, forecasts, title, yaxis_title, zoom_range):
    """
    Creates a bar chart for visualizing actuals and forecast data.

    Args:
        filtered_data (pd.DataFrame): Historical data.
        forecasts (dict): Dictionary of forecast data by model.
        title (str): Title of the bar chart.
        yaxis_title (str): Label for the y-axis.
        zoom_range (list): Range for the x-axis zoom.

    Returns:
        plotly.graph_objects.Figure: The generated bar chart.
    """
    bar_chart = go.Figure()
    bar_chart.add_trace(go.Bar(
        x=filtered_data['date'],
        y=filtered_data['value'],
        name='Actuals',
        marker=dict(color='blue')
    ))
    for model, model_data in forecasts.items():
        bar_chart.add_trace(go.Bar(
            x=model_data['date'],
            y=model_data['value'],
            name=f'{model}',
            marker=dict()
        ))
    bar_chart.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        template="plotly_white",
        barmode='group',
        showlegend=True,
        xaxis=dict(range=zoom_range)
    )
    return bar_chart
