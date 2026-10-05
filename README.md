# Car Valuation System

This project predicts the market value of used cars using a trained machine-learning model. It combines a FastAPI backend for prediction logic and a Streamlit frontend for user interaction.

## Project structure

- [AI Project/backend/main.py](./AI%20Project/backend/main.py) — FastAPI service that loads the trained model and exposes the `/predict` endpoint.
- [AI Project/frontend/app.py](./AI%20Project/frontend/app.py) — Streamlit UI where users choose make, model, year, fuel, transmission, and mileage.
- [AI Project/models/pipeline.pkl](./AI%20Project/models/pipeline.pkl) — trained prediction pipeline used by the backend.
- [AI Project/notebook/Car_Valuation.ipynb](./AI%20Project/notebook/Car_Valuation.ipynb) — notebook used for model development and analysis.

## Features

- Car brand and model selection
- Year, mileage, fuel, assembly, transmission, and document inputs
- Vehicle age calculation from the selected year
- Price estimation through a trained model
- Backend API for prediction requests

## Setup

1. Open a terminal in the project root.
2. Create and activate a virtual environment if needed.
3. Install the project dependencies:

   ```bash
   pip install -r "AI Project/requirements.txt"
   ```

## Run the system

Start the backend:

```bash
cd "AI Project/backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend in a second terminal:

```bash
cd "AI Project/frontend"
streamlit run app.py
```

Open the Streamlit URL shown in the terminal, usually `http://localhost:8501`.

## Example request

The backend expects a JSON payload like this:

```json
{
  "Make": "Toyota",
  "Model": "Corolla XLI",
  "Year": 2019,
  "KM's driven": 20000,
  "Fuel": "Petrol",
  "Car documents": "Original",
  "Assembly": "Local",
  "Transmission": "Manual",
  "Car Age": 5
}
```

## Notes

- The backend expects the trained model file to exist at `AI Project/models/pipeline.pkl`.
- The frontend sends prediction requests to `http://127.0.0.1:8000/predict` by default.
- The model is built for the specific feature set used by the project form and notebook.
