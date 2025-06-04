from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras import backend as K

# Определение пользовательской метрики
def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

app = FastAPI()

# Настройка путей
BASE_DIR = Path(__file__).parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

# Создаем директории, если их нет
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# Проверка и создание шаблона
index_path = templates_dir / "index.html"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Загрузка модели с правильным путем
try:
    model_path = BASE_DIR / "my_model.keras"
    print(f"Attempting to load model from: {model_path}")
    
    if model_path.exists():
        model = tf.keras.models.load_model(
            str(model_path),  # Явное преобразование Path в строку
            custom_objects={'r_squared': r_squared}
        )
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model file not found at: {model_path}")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "show_result": False
    })

@app.post("/predict")
async def predict_price(
    request: Request,
    rcmnd_cruise_knots: float = Form(...),
    stall_knots_dirty: float = Form(...),
    fuel_gal_lbs: float = Form(...),
    eng_out_rate_of_climb: float = Form(...),
    takeoff_over_50ft: float = Form(...)
):
    try:
        if model is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Model not loaded. Please check server logs."
            })
        
        input_data = pd.DataFrame([[rcmnd_cruise_knots, stall_knots_dirty, fuel_gal_lbs, 
                                 eng_out_rate_of_climb, takeoff_over_50ft]],
                               columns=['Rcmnd cruise Knots', 'Stall Knots dirty', 
                                       'Fuel gal/lbs', 'Eng out rate of climb',
                                       'Takeoff over 50ft'])
        
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0][0])
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "show_result": True,
            "predicted_price": f"{predicted_price:,.2f}"
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Prediction error: {str(e)}"
        })


from fastapi.responses import JSONResponse



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
