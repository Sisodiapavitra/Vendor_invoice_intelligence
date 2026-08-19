from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
FREIGHT_MODEL_PATH = PROJECT_ROOT / "models" / "predict_freight_model.pkl"
INVOICE_MODEL_PATH = PROJECT_ROOT / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

INVOICE_FEATURES = [
	"invoice_quantity",
	"invoice_dollars",
	"Freight",
	"total_item_quantity",
	"total_item_dollars",
]


@st.cache_resource
def load_artifacts():
	"""Load the trained models once per Streamlit process."""
	freight_model = joblib.load(FREIGHT_MODEL_PATH)
	invoice_model = joblib.load(INVOICE_MODEL_PATH)
	scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
	return freight_model, invoice_model, scaler


def validate_columns(dataframe, required_columns):
	missing_columns = [column for column in required_columns if column not in dataframe.columns]
	if missing_columns:
		return f"Missing required columns: {', '.join(missing_columns)}"
	return None


def predict_freight(dataframe, model):
	values = dataframe[["Dollars"]].apply(pd.to_numeric, errors="raise")
	result = dataframe.copy()
	result["Predicted_Freight"] = model.predict(values).round(2)
	return result


def predict_invoice_flag(dataframe, model, scaler):
	values = dataframe[INVOICE_FEATURES].apply(pd.to_numeric, errors="raise")
	model_values = scaler.transform(values) if scaler is not None else values
	result = dataframe.copy()
	result["Predicted_flag"] = model.predict(model_values).astype(int)
	result["Risk"] = result["Predicted_flag"].map({1: "Flagged", 0: "Clear"})
	return result


def show_batch_upload(required_columns, prediction_function, uploader_key):
	uploaded_file = st.file_uploader(
		"Upload a CSV for batch scoring",
		type="csv",
		key=uploader_key,
	)
	if uploaded_file is None:
		return

	dataframe = pd.read_csv(uploaded_file)
	error = validate_columns(dataframe, required_columns)
	if error:
		st.error(error)
		st.info(f"Expected columns: {', '.join(required_columns)}")
		return

	try:
		predictions = prediction_function(dataframe)
	except (TypeError, ValueError) as error:
		st.error(f"Could not score the uploaded data: {error}")
		return

	st.dataframe(predictions, use_container_width=True, hide_index=True)
	st.download_button(
		"Download predictions",
		predictions.to_csv(index=False).encode("utf-8"),
		file_name="predictions.csv",
		mime="text/csv",
		key=f"download_button_{uploader_key}"
	)


def freight_tab(freight_model):
	st.subheader("Freight cost prediction")
	st.write("Estimate freight cost from the invoice dollar amount.")

	with st.form("freight_form"):
		dollars = st.number_input("Invoice dollars", min_value=0.0, value=18500.0, step=100.0)
		submitted = st.form_submit_button("Predict freight cost", type="primary")

	if submitted:
		prediction = predict_freight(pd.DataFrame({"Dollars": [dollars]}), freight_model)
		st.metric("Predicted freight", f"${prediction.at[0, 'Predicted_Freight']:,.2f}")

	st.divider()
	show_batch_upload(
		["Dollars"],
		lambda dataframe: predict_freight(dataframe, freight_model),
		"freight_csv",
	)


def invoice_tab(invoice_model, scaler):
	st.subheader("Invoice risk flagging")
	st.write("Classify an invoice using its quantity, dollar, freight, and purchase totals.")

	with st.form("invoice_form"):
		values = {
			"invoice_quantity": st.number_input("Invoice quantity", min_value=0.0, value=10.0, step=1.0),
			"invoice_dollars": st.number_input("Invoice dollars", min_value=0.0, value=1000.0, step=50.0),
			"Freight": st.number_input("Freight", min_value=0.0, value=50.0, step=10.0),
			"total_item_quantity": st.number_input("Purchase quantity", min_value=0.0, value=10.0, step=1.0),
			"total_item_dollars": st.number_input("Purchase dollars", min_value=0.0, value=1000.0, step=50.0),
		}
		submitted = st.form_submit_button("Check invoice", type="primary")

	if submitted:
		prediction = predict_invoice_flag(pd.DataFrame([values]), invoice_model, scaler)
		flagged = prediction.at[0, "Predicted_flag"] == 1
		if flagged:
			st.error("Invoice flagged for review")
		else:
			st.success("Invoice appears clear")

	st.divider()
	if scaler is None:
		st.warning("The saved scaler was not found. Predictions will use the model without scaling.")
	show_batch_upload(
		INVOICE_FEATURES,
		lambda dataframe: predict_invoice_flag(dataframe, invoice_model, scaler),
		"invoice_csv",
	)


def main():
	st.set_page_config(page_title="Vendor Invoice Intelligence", page_icon="📊", layout="wide")
	st.title("Vendor Invoice Intelligence")
	st.caption("Predict freight costs and identify invoices that may need review.")

	try:
		freight_model, invoice_model, scaler = load_artifacts()
	except FileNotFoundError as error:
		st.error(f"A trained model artifact is missing: {error.filename}")
		st.info("Run the project training scripts first, then restart Streamlit.")
		st.stop()

	freight_page, invoice_page = st.tabs(["Freight cost", "Invoice flagging"])
	with freight_page:
		freight_tab(freight_model)
	with invoice_page:
		invoice_tab(invoice_model, scaler)


if __name__ == "__main__":
	main()
