from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# -------------------------------
# Load model and scaler safely
# -------------------------------
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("✅ Model loaded.")
except Exception as e:
    print("❌ Model not found or invalid:", e)

try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("✅ Scaler loaded.")
except Exception as e:
    scaler = None
    print("⚠️ Scaler not found:", e)

# -------------------------------
# Home route
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch values from HTML form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Combine into array
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        print("📥 Input data:", input_data)

        # Scale if scaler exists
        if scaler is not None:
            input_data = scaler.transform(input_data)
            print("📊 Scaled data:", input_data)

        # Predict
        prediction = model.predict(input_data)
        print("🔮 Raw model output:", prediction)

        # Crop dictionary mapping
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidneybeans", 5: "Pigeonpeas",
            6: "Mothbeans", 7: "Mungbean", 8: "Blackgram", 9: "Lentil",
            10: "Pomegranate", 11: "Banana", 12: "Mango", 13: "Grapes", 14: "Watermelon",
            15: "Muskmelon", 16: "Apple", 17: "Orange", 18: "Papaya", 19: "Coconut",
            20: "Cotton", 21: "Jute", 22: "Coffee"
        }

        # Decode crop name
        crop = crop_dict.get(int(prediction[0]), "Unknown Crop")
        result = f"{crop} is the best crop to be cultivated right there."
        print("🌾 Result:", result)

        # Send back to frontend
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        print("❌ Error:", e)
        return render_template('index.html', prediction_text="Error during prediction. Check model/scaler or inputs.")

# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
