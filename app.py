from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Read the CSV file and preprocess data
df = pd.read_csv('chain_restaurants.csv')
scaler = MinMaxScaler()
features = df[['cost', 'healthiness', 'speed']]
scaler.fit(features)
model = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(features)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle recommendation request
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    cost = float(data['cost'])
    healthiness = float(data['healthiness'])
    speed = float(data['speed'])
    
    input_features = scaler.transform([[cost, healthiness, speed]])
    distances, indices = model.kneighbors(input_features)
    recommended_restaurants = df.iloc[indices[0]].to_dict(orient='records')
    
    return jsonify(recommended_restaurants)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(port=5001)
