from flask import Flask, render_template, request
from jinja2.exceptions import TemplateNotFound
import requests
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)

# Replace 'YOUR_MAPBOX_API_KEY' with your actual Mapbox API key
MAPBOX_API_KEY = 'pk.eyJ1IjoicHJpc2tpbGEtMjgiLCJhIjoiY2xwMDI3N2t6MDM4MzJocXU5ZmJ2Y2x2dyJ9.i2R8XDFuKihKtU4WdQcjMA'
MAPBOX_API_KEY = 'pk.eyJ1IjoicHJpc2tpbGEtMjgiLCJhIjoiY2xwMDI3N2t6MDM4MzJocXU5ZmJ2Y2x2dyJ9.i2R8XDFuKihKtU4WdQcjMA'
# Dictionary to store product locations (latitude, longitude) and shipping costs
product_locations = {}

def get_coordinates(address):
    # Use Mapbox Geocoding API to get latitude and longitude for Chennai
    url = f'https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={MAPBOX_API_KEY}&bbox=80.18,12.78,80.27,13.17'
    response = requests.get(url)
    data = response.json()

    if data['features']:
        location = data['features'][0]['geometry']['coordinates']
        return location[1], location[0]  # Mapbox returns [lon, lat]
    else:
        return None

def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula to calculate distance between two points on Earth
    R = 6371  # Radius of the Earth in kilometers

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


@app.route('/')
def index():
    try:
        return render_template('index.html')  # Updated to render index.html
    except TemplateNotFound:
        return "Template not found. Please make sure the 'templates' folder is in the correct location.", 500

@app.route('/add_product', methods=['POST'])
def add_product():
    product_id = request.form['product_id']
    product_address = request.form['product_address']
    cost_per_km = request.form.get('cost_per_km', type=float)

    # Get product coordinates
    product_coordinates = get_coordinates(product_address)

    if product_coordinates and cost_per_km is not None:
        product_locations[product_id] = {'coordinates': product_coordinates, 'cost_per_km': cost_per_km}
        return render_template('index_input.html', product_locations=product_locations, message='Product added successfully!')
    else:
        return render_template('index_input.html', product_locations=product_locations, error='Invalid product information. Please try again.')

@app.route('/calculate_distance_cost', methods=['POST'])
def calculate_distance_cost_route():
    user_address = request.form['user_address']
    product_id = request.form['product_id']

    # Get user coordinates
    user_coordinates = get_coordinates(user_address)

    if user_coordinates:
        # Check if the product ID is valid
        if product_id in product_locations:
            product_coordinates = product_locations[product_id]['coordinates']
            cost_per_km = product_locations[product_id]['cost_per_km']
        else:
            return render_template('index_input.html', product_locations=product_locations, error='Invalid product ID. Please try again.')

        # Calculate distance
        distance = calculate_distance(user_coordinates[0], user_coordinates[1],
                                      product_coordinates[0], product_coordinates[1])

        # Calculate cost
        cost = distance * cost_per_km

        return render_template('index_input.html', product_locations=product_locations, distance=distance, cost=cost)
    else:
        return render_template('index_input.html', product_locations=product_locations, error='Invalid user address. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)
    