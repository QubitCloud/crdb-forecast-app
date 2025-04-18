from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from prophet import Prophet
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def prepare_data():
    try:
        df = pd.read_csv("CRDB_Share_Prices.csv")
        df['ds'] = pd.to_datetime(df['Month'], format='%b-%y')  # No Year column, just Month like "Jan-24"
        df['y'] = df['Average CRDB Share Price (TSh)']
        return df[['ds', 'y']]
    except Exception as e:
        print("Data preparation failed:", e)
        raise

def get_forecast_price(month: str, year: int):
    df = prepare_data()
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=60, freq='MS')
    forecast = model.predict(future)

    try:
        target_date = datetime.strptime(f"{month} {year}", "%b %Y")
        forecast_row = forecast[forecast['ds'] == target_date]
        if forecast_row.empty:
            return None
        return round(forecast_row.iloc[0]['yhat'], 2)
    except Exception as e:
        print("Forecast error:", e)
        return None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "forecast_result": None})

@app.post("/", response_class=HTMLResponse)
def post_forecast(request: Request, month: str = Form(...), year: int = Form(...)):
    try:
        price = get_forecast_price(month, year)
        if price is None:
            result = f"No forecast available for {month} {year}."
        else:
            result = f"📈 Estimated CRDB share price for <b>{month} {year}</b>: <span style='color:green;'>TSh {price}</span>"
    except Exception as e:
        result = f"⚠️ Invalid input. Please try again."
        print("User input error:", e)

    return templates.TemplateResponse("index.html", {"request": request, "forecast_result": result})
