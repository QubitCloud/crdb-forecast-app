from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from prophet import Prophet
from datetime import datetime
import traceback

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def prepare_data():
    try:
        df = pd.read_csv("CRDB_Share_Prices.csv")
        df['ds'] = pd.to_datetime(df['Month'] + " " + df['Year'].astype(str), format='%b-%y')
        df['y'] = df['Average CRDB Share Price (TSh)']
        return df[['ds', 'y']]
    except Exception as e:
        print("Data preparation failed:", e)
        traceback.print_exc()
        return pd.DataFrame(columns=['ds', 'y'])  # return empty DataFrame as fallback

def get_forecast_price(month: str, year: int):
    df = prepare_data()
    if df.empty:
        return None

    try:
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=60, freq='MS')
        forecast = model.predict(future)

        target_date = datetime.strptime(f"{month} {year}", "%b %Y")
        forecast_row = forecast[forecast['ds'] == target_date]

        if forecast_row.empty:
            return None

        return round(forecast_row.iloc[0]['yhat'], 2)
    except Exception as e:
        print(f"Forecasting failed for {month} {year}: {e}")
        traceback.print_exc()
        return None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "forecast_result": None})

@app.post("/", response_class=HTMLResponse)
def post_forecast(request: Request, month: str = Form(...), year: int = Form(...)):
    try:
        print(f"User input received - Month: {month}, Year: {year}")
        price = get_forecast_price(month, year)
        if price is None:
            result = f"❌ No forecast available for <b>{month} {year}</b>."
        else:
            result = f"📈 Estimated CRDB share price for <b>{month} {year}</b>: <span style='color:green;'>TSh {price}</span>"
    except Exception as e:
        print("Error during form submission:", e)
        traceback.print_exc()
        result = "⚠️ Invalid input or internal error. Please try again."

    return templates.TemplateResponse("index.html", {"request": request, "forecast_result": result})
