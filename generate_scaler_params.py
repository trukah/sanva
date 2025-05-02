import os
import sys
import json
import joblib
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def generate_scaler_params():
    """
    Extract min/max values from scaler and save as JSON for Cloudflare Worker
    """
    # Load the scaler
    scaler_path = '../models/scaler_xau.pkl'
    scaler = joblib.load(scaler_path)
    
    # Get the feature names
    feature_columns = ['Open', 'High', 'Low', 'Close']
    
    # Extract min and max values
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    
    # Create parameter dictionary
    scaler_params = {
        'min': dict(zip(feature_columns, data_min.tolist())),
        'max': dict(zip(feature_columns, data_max.tolist()))
    }
    
    # Save as JSON
    output_path = '../models/scaler_params.json'
    with open(output_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    print(f"Scaler parameters saved to {output_path}")
    print(f"Min values: {scaler_params['min']}")
    print(f"Max values: {scaler_params['max']}")

if __name__ == "__main__":
    generate_scaler_params()
