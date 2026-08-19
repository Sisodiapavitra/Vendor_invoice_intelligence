# Vendor Invoice Intelligence

A Streamlit application for vendor invoice analysis. It supports freight cost prediction and invoice risk flagging using trained machine-learning models.

## Features

- Predict freight cost from invoice dollar amounts.
- Flag invoices for review using invoice and purchase information.
- Make single-record predictions through Streamlit forms.
- Upload CSV files for batch predictions.
- Download prediction results as CSV.

## Quick Start

Open PowerShell in the project directory:

```powershell
cd "C:\Users\Pavitra Sisodia\Vendor-invoice-prediction"
```

Install the dependencies:

```powershell
py -m pip install -r requirements.txt
```

Start the application:

```powershell
py -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```text
app.py                              Streamlit application
data/                               Database and sample CSV files
freight_cost_prediction/            Freight model training and evaluation
invoice_flagging/                   Invoice risk model training and evaluation
inference/                          Standalone prediction scripts
models/                             Trained model artifacts
notebooks/                          Exploratory notebooks
requirements.txt                    Python dependencies
```

## Using the Application

### Freight Cost Prediction

Use the **Freight cost** tab to enter one invoice dollar amount or upload a CSV for batch prediction.

The CSV must contain a numeric `Dollars` column:

```csv
Dollars
200
3000
9000
18500
25000
```

The application adds a `Predicted_Freight` column to the results.

Sample file: `data/sample_freight_prediction.csv`

### Invoice Flagging

Use the **Invoice flagging** tab to enter one invoice or upload multiple invoices in a CSV file.

The CSV must contain these numeric columns:

```csv
invoice_quantity,invoice_dollars,Freight,total_item_quantity,total_item_dollars
10,1000,50,10,1000
8,875,40,8,800
```

The application adds:

- `Predicted_flag`: `1` means flagged and `0` means clear.
- `Risk`: `Flagged` or `Clear`.

Sample file: `data/sample_invoice_flagging.csv`

Column names are case-sensitive.

## Model Artifacts

The application loads the following files from the root `models/` directory:

```text
models/predict_freight_model.pkl
models/predict_flag_invoice.pkl
models/scaler.pkl
```

The model files are ignored by Git. Make sure they are available when running the application from a fresh clone, or retrain the models locally.

## Retraining Models

The training scripts expect the project data to be available in `data/`.

Train the freight model:

```powershell
cd freight_cost_prediction
py train.py
```

Train the invoice flagging model:

```powershell
cd ..\invoice_flagging
py train.py
```

Return to the project root and start Streamlit:

```powershell
cd ..
py -m streamlit run app.py
```

## Standalone Inference

Run the freight prediction script directly from the project root:

```powershell
py inference\predict_freight.py
```

## Notes

- Uploaded CSV values must be numeric.
- The invoice model uses `scaler.pkl` before prediction.
- Do not commit secrets, local databases, virtual environments, or generated model artifacts.
