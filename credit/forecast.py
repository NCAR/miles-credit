from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def load_forecasts(conf):
    if "type" in conf["predict"]["forecasts"]:
        forecast_type = conf["predict"]["forecasts"]["type"]
        start_date = datetime(
            conf["predict"]["forecasts"]["start_year"],
            conf["predict"]["forecasts"]["start_month"],
            conf["predict"]["forecasts"]["start_day"],
        )
        start_hours = conf["predict"]["forecasts"].get("start_hours", [0])  # Default to [0] if not provided
        if forecast_type == "10day_year":
            return generate_forecasts(start_date, days=10, duration=365, start_hours=start_hours)
        elif forecast_type == "custom":
            days = conf["predict"]["forecasts"].get("days", 10)
            lead_time_periods = conf["data"].get("lead_time_periods", 1)
            duration = conf["predict"]["forecasts"].get("duration", 365)
            ic_interval_days = conf["predict"]["forecasts"].get("ic_interval_days", 1)
            return generate_forecasts(
                start_date,
                lead_time_periods,
                days=days,
                duration=duration,
                start_hours=start_hours,
                ic_interval_days=ic_interval_days,
            )
        else:
            logger.warning(f"Forecast type '{forecast_type}' not supported")
            raise ValueError(f"Forecast type '{forecast_type}' not supported")
    else:
        return conf["predict"]["forecasts"]


# Function to generate forecasts for specified duration
def generate_forecasts(start_date, lead_time_periods=1, days=10, duration=365, start_hours=(0), ic_interval_days=1):
    """
    lead_time_periods = 1 for hourly forecast; =6 for 6 hourly forecast
    """
    forecasts = []
    current_date = start_date  # Use the provided start_date directly

    # Generate forecast for each day
    for _ in range(duration // ic_interval_days):
        for hour in start_hours:
            start_datetime = current_date + timedelta(hours=hour)
            end_datetime = start_datetime + timedelta(days=days - 1, hours=24 - lead_time_periods)
            forecasts.append(
                [
                    start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )
        current_date += timedelta(days=ic_interval_days)

    if len(forecasts) == 0:
        raise RuntimeError("No forecasts generated")

    logger.info(f"Generated {len(forecasts)} unique forecast periods ")

    return forecasts


if __name__ == "__main__":
    # Example usage
    forecast_details = {
        "type": "custom",
        "start_year": 2018,
        "start_month": 6,
        "start_day": 1,
        "days": 10,
        "start_hours": [0, 12],
        "duration": 4,
    }

    conf = {"predict": {"forecasts": forecast_details}}
    forecasts = load_forecasts(conf)

    # Print example forecasts
    for forecast in forecasts:
        print(forecast)
