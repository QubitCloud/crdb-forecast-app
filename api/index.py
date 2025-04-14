from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from prophet import Prophet
from datetime import datetime
import os

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../templates"))

def prepare_data():
    df = pd.read_csv('CRDB_Share_Prices.csv')
    df['ds'] = pd.to_datetime(df['Month'], format='%b-%y')
    df['y'] = df['Average CRDB Share Price (TSh)']
    return df[['ds', 'y']]

def get_forecast_price(month: str, year: int):
    df = prepare_data()
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=60, freq='MS')
    forecast = model.predict(future)

    target_date = datetime.strptime(f"{month} {year}", "%b %Y")
    forecast_row = forecast[forecast['ds'] == target_date]

    if forecast_row.empty:
        return None

    return round(forecast_row.iloc[0]['yhat'], 2)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "forecast_result": None})

@app.post("/", response_class=HTMLResponse)
def post_forecast(request: Request, month: str = Form(...), year: int = Form(...)):
    try:
        price = get_forecast_price(month, year)
        if price is None:
            result = f"No forecast data available for {month} {year}."
        else:
            result = f"Estimated CRDB share price for {month} {year} is TSh {price}."
    except:
        result = "Invalid format, try again."
    return templates.TemplateResponse("index.html", {"request": request, "forecast_result": result})
