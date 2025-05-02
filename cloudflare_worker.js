/**
 * Cloudflare Worker script for XAU/USD prediction
 * This worker loads the TFLite model and provides predictions
 * 
 * Deploy via Cloudflare Pages or Cloudflare Workers
 */
import * as tf from '@tensorflow/tfjs';

// Configuration
const CONFIG = {
  MODEL_URL: 'https://yourgithubusername.github.io/pricePredictor/models/model.tflite',
  SCALER_URL: 'https://yourgithubusername.github.io/pricePredictor/models/scaler_params.json',
  SEQUENCE_LENGTH: 24,
  FEATURES: ['Open', 'High', 'Low', 'Close']
};

// Cors headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

/**
 * Handle OPTIONS request for CORS preflight
 */
function handleOptions(request) {
  return new Response(null, {
    status: 204,
    headers: corsHeaders
  });
}

/**
 * Normalize input data using the scaler parameters
 */
function normalizeData(data, scalerParams) {
  const normalized = [];
  
  for (let i = 0; i < data.length; i++) {
    const feature = CONFIG.FEATURES[i];
    const value = data[i];
    const min = scalerParams.min[feature];
    const max = scalerParams.max[feature];
    normalized.push((value - min) / (max - min));
  }
  
  return normalized;
}

/**
 * Denormalize predicted value
 */
function denormalizeValue(value, scalerParams) {
  const min = scalerParams.min.Close;
  const max = scalerParams.max.Close;
  return value * (max - min) + min;
}

/**
 * Fetch the TFLite model and set up the interpreter
 */
async function loadModel() {
  const modelResponse = await fetch(CONFIG.MODEL_URL);
  const modelBuffer = await modelResponse.arrayBuffer();
  
  const tfLiteModel = await tf.tflite.loadTFLiteModel(modelBuffer);
  return tfLiteModel;
}

/**
 * Fetch the scaler parameters
 */
async function loadScalerParams() {
  const scalerResponse = await fetch(CONFIG.SCALER_URL);
  return await scalerResponse.json();
}

/**
 * Make a prediction using the TFLite model
 */
async function predict(model, inputData, scalerParams) {
  // Normalize input data
  const normalizedData = [];
  for (let i = 0; i < inputData.length; i++) {
    normalizedData.push(normalizeData(inputData[i], scalerParams));
  }
  
  // Prepare input tensor (shape: [1, sequence_length, features])
  const inputTensor = tf.tensor3d([normalizedData], [1, inputData.length, CONFIG.FEATURES.length]);
  
  // Run inference
  const outputTensor = model.predict(inputTensor);
  const prediction = outputTensor.dataSync()[0];
  
  // Denormalize the prediction
  return denormalizeValue(prediction, scalerParams);
}

/**
 * Generate predictions for multiple steps ahead
 */
async function generateForecast(model, inputData, scalerParams, steps) {
  let currentSequence = [...inputData];
  const forecast = [];
  
  for (let step = 0; step < steps; step++) {
    // Make prediction for next step
    const normalizedData = [];
    for (let i = 0; i < currentSequence.length; i++) {
      normalizedData.push(normalizeData(currentSequence[i], scalerParams));
    }
    
    const inputTensor = tf.tensor3d([normalizedData], [1, currentSequence.length, CONFIG.FEATURES.length]);
    const outputTensor = model.predict(inputTensor);
    const predictedValue = outputTensor.dataSync()[0];
    
    // Denormalize the prediction
    const predictedPrice = denormalizeValue(predictedValue, scalerParams);
    forecast.push(predictedPrice);
    
    // Update sequence for next prediction (drop oldest, add newest)
    currentSequence.shift();
    // Use the same value for all features as placeholder
    currentSequence.push([predictedPrice, predictedPrice, predictedPrice, predictedPrice]);
  }
  
  return forecast;
}

/**
 * Main handler for the Cloudflare Worker
 */
async function handleRequest(request) {
  try {
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return handleOptions(request);
    }
    
    // Parse request data
    let requestData;
    if (request.method === 'POST') {
      requestData = await request.json();
    } else {
      requestData = {
        data: null,
        steps: 24 // Default forecast steps
      };
    }
    
    // Load model and scaler parameters
    const [model, scalerParams] = await Promise.all([
      loadModel(),
      loadScalerParams()
    ]);
    
    // Generate timestamps for forecast
    const now = new Date();
    const timestamps = [];
    for (let i = 0; i < requestData.steps; i++) {
      const timestamp = new Date(now);
      timestamp.setHours(now.getHours() + i + 1);
      timestamps.push(timestamp.toISOString());
    }
    
    // Generate forecast
    const forecast = await generateForecast(
      model, 
      requestData.data || Array(CONFIG.SEQUENCE_LENGTH).fill([0, 0, 0, 0]), // Default data if none provided
      scalerParams,
      requestData.steps || 24
    );
    
    // Return forecast data
    const response = {
      success: true,
      timestamp: new Date().toISOString(),
      forecast: timestamps.map((time, i) => ({
        time,
        price: forecast[i].toFixed(2)
      }))
    };
    
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    });
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: error.message || 'An error occurred during prediction'
    }), {
      status: 500,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    });
  }
}

// Export the request handler for the Cloudflare Worker
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});
