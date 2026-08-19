import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "predict_flag_invoice.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.
    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_flag'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    # Example usage
    sample_input = {
        "feature1": [0.5, 1.2],
        "feature2": [3.4, 2.1],
        "feature3": [1.0, 0.8]
    }
    predictions = predict_invoice_flag(sample_input)
    print(predictions)