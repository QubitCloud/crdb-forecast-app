from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from prophet import Prophet
from datetime import datetime

app = FastAPI(title="CRDB Forecast Dashboard")
templates = Jinja2Templates(directory="templates")

def prepare_data():
    df = pd.read_csv('CRDB_Share_Prices.csv')
    df['ds'] = pd.to_datetime(df['Month'], format='%b-%y')
    df['y'] = df['Average CRDB Share Price (TSh)']
    return df[['ds', 'y']]

def forecast_from(month: str, year: int):
    df = prepare_data()
    model = Prophet()
    model.fit(df)

    try:
        start_date = datetime.strptime(f"{month} {year}", "%b %Y")
    except ValueError:
        raise Exception("Invalid format, try again.")

    future = model.make_future_dataframe(periods=60, freq='MS')
    future = future[future['ds'] >= start_date]

    forecast = model.predict(future)
    forecast_result = forecast[forecast['ds'] == forecast['ds'].min()]
    predicted_value = round(forecast_result['yhat'].values[0], 2) if not forecast_result.empty else "N/A"

    return f"The estimated price in {month} {year} is TSh {predicted_value}"

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "forecast": None})

@app.post("/", response_class=HTMLResponse)
def get_forecast(request: Request, month: str = Form(...), year: int = Form(...)):
    try:
        forecast_result = forecast_from(month, year)
    except Exception:
        forecast_result = "Invalid format, try again."
    return templates.TemplateResponse("index.html", {"request": request, "forecast": forecast_result})