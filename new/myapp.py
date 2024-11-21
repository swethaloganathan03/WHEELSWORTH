from flask import Flask, render_template, request
import os
from models.car_model import predict_car_price
from models.bike_model import predict_bike_price

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def main_page():
    return render_template('main.html')

# Car Price Prediction
@app.route('/car', methods=['GET', 'POST'])
def car_index():
    predicted_price = None
    user_input = {}
    uploaded_image_url = None
    exchange_rate = 82.0  # Example exchange rate from USD to INR; adjust as needed

    if request.method == 'POST':
        user_input = {
            'brand': request.form['brand'],
            'model': request.form['model'],
            'model_year': int(request.form['model_year']),
            'fuel_type': request.form['fuel_type'],
            'engine': request.form['engine'],
            'transmission': request.form['transmission'],
            'ext_col': request.form['ext_col'],
            'accident': request.form['accident'],
            'clean_title': request.form['clean_title']
        }

        if 'vehicle_image' in request.files:
            vehicle_image = request.files['vehicle_image']
            if vehicle_image.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], vehicle_image.filename)
                vehicle_image.save(image_path)
                uploaded_image_url = f"/{image_path}"

        predicted_price_usd = predict_car_price(**user_input)
        predicted_price = round(predicted_price_usd * exchange_rate)  # Convert to INR

    return render_template('car_index.html', predicted_price=predicted_price, user_input=user_input, uploaded_image_url=uploaded_image_url)

# Bike Price Prediction
@app.route('/bike', methods=['GET', 'POST'])
def bike_index():
    predicted_price = None
    user_input = {}
    uploaded_image_url = None

    if request.method == 'POST':
        user_input = {
            'bike_name': request.form['bike_name'],
            'kms_driven': int(request.form['kms_driven']),
            'age': int(request.form['age']),
            'power': float(request.form['power']),
            'brand': request.form['brand']
        }

        if 'vehicle_image' in request.files:
            vehicle_image = request.files['vehicle_image']
            if vehicle_image.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], vehicle_image.filename)
                vehicle_image.save(image_path)
                uploaded_image_url = f"/{image_path}"

        predicted_price = round(predict_bike_price(**user_input))

    return render_template('bike_index.html', predicted_price=predicted_price, user_input=user_input, uploaded_image_url=uploaded_image_url)

if __name__ == '__main__':
    app.run(debug=True)
