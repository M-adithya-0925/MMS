from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Blueprint, request, jsonify, current_app

import os
import numpy as np


app = Flask(__name__,
            template_folder='app/templates',
            static_folder='app/static')
app.config['UPLOAD_FOLDER'] = 'app/static/uploads' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost:3306/medicinal_quality_monitoring'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --------------------------
# Database Models
# --------------------------

class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100))
    manufacturer = db.Column(db.String(100))
    expiry_date = db.Column(db.Date)
    current_stock = db.Column(db.Integer, default=0)


class QualityInspection(db.Model):
    __tablename__ = 'quality_inspections'
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    inspection_time = db.Column(db.DateTime, default=datetime.utcnow)
    quality_status = db.Column(db.Enum('Pass', 'Fail', name='quality_status_enum'), nullable=False)
    confidence_score = db.Column(db.Float)
    image_path = db.Column(db.String(255))


class Sale(db.Model):
    __tablename__ = 'sales'
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity_sold = db.Column(db.Integer, nullable=False)
    sale_date = db.Column(db.Date, nullable=False)


class Alert(db.Model):
    __tablename__ = 'alerts'
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class DemandPrediction(db.Model):
    __tablename__ = 'demand_predictions'
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    predicted_date = db.Column(db.Date, nullable=False)
    predicted_demand = db.Column(db.Integer, nullable=False)
    model_used = db.Column(db.String(100))
    prediction_time = db.Column(db.DateTime, default=datetime.utcnow)


# --------------------------
# Routes for Frontend Display
# --------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/products')
def get_products():
    products = Product.query.all()
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'category': p.category,
        'manufacturer': p.manufacturer,
        'expiry_date': str(p.expiry_date),
        'current_stock': p.current_stock
    } for p in products])


@app.route('/inspections')
def get_inspections():
    inspections = QualityInspection.query.all()
    return jsonify([{
        'product_id': i.product_id,
        'product_name': Product.query.get(i.product_id).name,
        'quality_status': i.quality_status,
        'confidence_score': i.confidence_score,
        'image_path': i.image_path,
        'inspection_time': str(i.inspection_time)
    } for i in inspections])



@app.route('/alerts')
def get_alerts():
    # Example: generate a low stock alert for any product < 10 units
    low_stock_products = Product.query.filter(Product.current_stock < 10).all()
    for product in low_stock_products:
        exists = Alert.query.filter_by(product_id=product.id, alert_type='LowStock').first()
        if not exists:
            alert = Alert(
                product_id=product.id,
                alert_type='LowStock',
                message=f'Stock for {product.name} dropped below safe level.'
            )
            db.session.add(alert)

    db.session.commit()

    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    return jsonify([{
        'alert_type': a.alert_type,
        'message': a.message,
        'timestamp': str(a.timestamp)
    } for a in alerts])



@app.route('/predictions')
def get_predictions():
    today = datetime.utcnow().date()

    products = Product.query.all()
    for product in products:
        exists = DemandPrediction.query.filter_by(product_id=product.id, predicted_date=today).first()
        if not exists:
            predicted_demand = np.random.randint(5, 50)  # Simulate
            prediction = DemandPrediction(
                product_id=product.id,
                predicted_date=today,
                predicted_demand=predicted_demand,
                model_used='SimulatedModel'
            )
            db.session.add(prediction)

    db.session.commit()

    predictions = DemandPrediction.query.order_by(DemandPrediction.predicted_date).all()
    return jsonify([{
        'predicted_date': str(p.predicted_date),
        'predicted_demand': p.predicted_demand,
        'model_used': p.model_used,
        'prediction_time': str(p.prediction_time)
    } for p in predictions])



@app.route('/sales')
def get_sales():
    today = datetime.utcnow().date()

    for product in Product.query.all():
        exists = Sale.query.filter_by(product_id=product.id, sale_date=today).first()
        if not exists:
            quantity = np.random.randint(1, 10)
            sale = Sale(product_id=product.id, quantity_sold=quantity, sale_date=today)
            product.current_stock = max(product.current_stock - quantity, 0)
            db.session.add(sale)

    db.session.commit()

    sales = Sale.query.order_by(Sale.sale_date.desc()).all()
    return jsonify([{
        'product_id': s.product_id,
        'quantity_sold': s.quantity_sold,
        'sale_date': str(s.sale_date)
    } for s in sales])

@app.route('/quality_upload')
def upload_page():
    return render_template('quality_upload.html')

model = load_model('app/models/quality_classifier.h5')

UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_quality', methods=['POST'])
def upload_quality():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    name = request.form.get('name')
    category = request.form.get('category')
    manufacturer = request.form.get('manufacturer')
    expiry_date = request.form.get('expiry_date')
    stock = request.form.get('stock')

    if not all([name, category, manufacturer, expiry_date, stock]):
        return jsonify({'error': 'Missing product details'}), 400

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict quality
        prediction = model.predict(img_array)[0][0]
        quality_status = 'Pass' if prediction >= 0.5 else 'Fail'
        confidence = round(float(prediction if prediction >= 0.5 else 1 - prediction), 2)

        # Check if product already exists
        product = Product.query.filter_by(
            name=name,
            category=category,
            manufacturer=manufacturer,
            expiry_date=expiry_date
        ).first()

        if not product:
            product = Product(
                name=name,
                category=category,
                manufacturer=manufacturer,
                expiry_date=expiry_date,
                current_stock=int(stock)
            )
            db.session.add(product)
            db.session.commit()
        else:
            # Update stock
            product.current_stock += int(stock)
            db.session.commit()

        # Save relative image path
        image_path = os.path.join('uploads', filename)

        # Save inspection
        inspection = QualityInspection(
            product_id=product.id,
            quality_status=quality_status,
            confidence_score=confidence,
            image_path=image_path
        )
        db.session.add(inspection)

        # Add alert if quality fails
        if quality_status == 'Fail':
            alert = Alert(
                product_id=product.id,
                alert_type='Quality Fail',
                message=f'{product.name} failed quality inspection with {confidence * 100:.1f}% confidence.'
            )
            db.session.add(alert)

        db.session.commit()

        return render_template('index.html')

    return jsonify({'error': 'Invalid file type'}), 400


# --------------------------
# App Runner
# --------------------------

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
