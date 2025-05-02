# Blog-Based Predictive AI System for Market Prices (XAU/USD)

This project implements a lightweight, deployable AI model that predicts XAU/USD (Gold) prices and displays them via a Blogspot UI frontend using Cloudflare Pages as the backend interface.

## Project Architecture Overview

```text
+------------------+        +---------------------------+        +--------------------------+
|  Blogspot (UI)   | <===>  |  Cloudflare Pages (API)   | <===>  | GitHub (Model & Script)  |
| HTML + JS Fetch  |        | Worker/Functions (predict)|        | - model.tflite (TinyML)  |
+------------------+        +---------------------------+        +--------------------------+
                                                            |
                                                            | Deploy via Termux (once)
                                                            v
                                                       +----------+
                                                       | Termux   |
                                                       | Python   |
                                                       | Zip Push |
                                                       +----------+
```

## Directory Structure

```text
pricePredictor/
├── data/
│   └── XAU_1h_data.csv               # Historical price data
├── models/
│   ├── lstm_model_xau.h5            # Trained Keras model
│   ├── scaler_xau.pkl               # Scaler object
│   ├── scaler_params.json           # Scaler parameters for JS
│   └── model.tflite                 # Converted TFLite model (for Cloudflare)
├── scripts/
│   ├── train_model.py               # Training pipeline
│   ├── predict_price.py             # Local inference
│   ├── evaluate_model.py            # Evaluation script
│   └── generate_scaler_params.py    # Generate scaler params for JS
├── src/
│   ├── preprocess.py                # Load + scale + window data
│   ├── model.py                     # Build FFN/LSTM model
│   ├── train.py                     # Model training logic
│   ├── predict.py                   # Inference logic
│   └── utils.py                     # Plotting + metrics
├── cloudflare_worker/
│   └── predict.js                   # Worker function for prediction
├── public/
│   ├── index.html                   # UI for Blogspot embedding
│   └── style.css                    # Optional styling
├── README.md
└── requirements.txt
```

## Implementation Steps

### Step 1: Setup and Data Preparation

1. **Create Project Structure**: Set up the directory structure as shown above.

2. **Install Dependencies**: Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare Training Data**: Place your XAU/USD historical data in the `data/` directory. The data should have at least these columns: Time, Open, High, Low, Close.

   You can obtain historical XAU/USD data from:
   - Financial data providers like Alpha Vantage, Quandl, or Yahoo Finance
   - Forex/trading platforms that offer historical data export

### Step 2: Model Training

1. **Run the Training Script**:
   ```
   cd scripts
   python train_model.py
   ```

   This will:
   - Load an
