import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

st.set_page_config(page_title="Food Delivery Time Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
 .main-header {font-size:2.5rem;font-weight:800;background:linear-gradient(135deg,#2563EB,#7C3AED);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1rem 0;}
 .prediction-box {background:linear-gradient(135deg,#059669,#10B981);border-radius:20px;padding:2rem;text-align:center;color:white;font-size:1.5rem;margin:1rem 0;box-shadow:0 8px 25px rgba(5,150,105,0.3);}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
 """Load trained model and preprocessing artifacts"""
 model = joblib.load("food_delivery_model.pkl")
 scaler = joblib.load("scaler.pkl")
 feature_names = joblib.load("feature_names.pkl")
 encoding_maps = joblib.load("encoding_maps.pkl")
 return model, scaler, feature_names, encoding_maps

try:
 model, scaler, feature_names, encoding_maps = load_artifacts()
 model_loaded = True
except Exception as e:
 model_loaded = False
 st.error(f"Error loading model: {e}")

def haversine(lat1, lon1, lat2, lon2):
 """Calculate great-circle distance between two GPS coordinates"""
 R = 6371
 lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
 dlat = lat2 - lat1; dlon = lon2 - lon1
 a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
 return R * 2 * math.asin(math.sqrt(a))

def compute_bearing(lat1, lon1, lat2, lon2):
 """Calculate compass bearing from origin to destination"""
 lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
 dlon = lon2 - lon1
 x = math.sin(dlon) * math.cos(lat2)
 y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
 return (math.degrees(math.atan2(x, y)) + 360) % 360

st.markdown('<h1 class="main-header">Food Delivery Time Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#94a3b8;font-size:1.1rem;">AI-powered ETA prediction for Indian food delivery services</p>', unsafe_allow_html=True)
st.markdown("---")

if model_loaded:
 col1, col2, col3 = st.columns(3)
 with col1:
  st.subheader("Location Details")
  rest_lat = st.number_input("Restaurant Latitude", value=22.745, format="%.6f")
  rest_lon = st.number_input("Restaurant Longitude", value=75.892, format="%.6f")
  del_lat = st.number_input("Delivery Latitude", value=22.765, format="%.6f")
  del_lon = st.number_input("Delivery Longitude", value=75.912, format="%.6f")
 with col2:
  st.subheader("Conditions")
  weather = st.selectbox("Weather", ["Sunny","Cloudy","Windy","Fog","Sandstorms","Stormy"])
  traffic = st.selectbox("Traffic Density", ["Low","Medium","High","Jam"])
  city = st.selectbox("City Type", ["Urban","Semi-Urban","Metropolitian"])
  festival = st.selectbox("Festival Day", ["No","Yes"])
 with col3:
  st.subheader("Delivery Person")
  age = st.slider("Age", 18, 45, 30)
  rating = st.slider("Rating", 1.0, 5.0, 4.5, 0.1)
  vehicle_type = st.selectbox("Vehicle", ["motorcycle","scooter","electric_scooter","bicycle"])
  vehicle_cond = st.slider("Vehicle Condition", 0, 3, 1)
  multi_del = st.selectbox("Multiple Deliveries", [0,1,2,3])
  order_type = st.selectbox("Order Type", ["Meal","Snack","Drinks","Buffet"])
 st.markdown("---")
 col_t1, col_t2 = st.columns(2)
 with col_t1: order_hour = st.slider("Order Hour", 0, 23, 14)
 with col_t2: prep_time = st.slider("Prep Time (min)", 5, 30, 15)

 if st.button("Predict Delivery Time", use_container_width=True, type="primary"):
  # Calculate geospatial features
  distance = haversine(rest_lat, rest_lon, del_lat, del_lon)
  bearing = compute_bearing(rest_lat, rest_lon, del_lat, del_lon)
  
  # Encode categorical variables
  traffic_map = {"Low":0,"Medium":1,"High":2,"Jam":3}
  weather_map = {"Sunny":0,"Cloudy":1,"Windy":2,"Fog":3,"Sandstorms":4,"Stormy":5}
  traffic_enc = traffic_map[traffic]
  weather_enc = weather_map[weather]
  festival_enc = 1 if festival=="Yes" else 0
  
  # Calculate efficiency score
  rating_norm = (rating - 1.0) / 4.0
  age_inv = 1 - (age - 18) / 27.0
  veh_norm = vehicle_cond / 3.0
  load_inv = 1 - multi_del / 3.0
  efficiency = 0.4*rating_norm + 0.2*age_inv + 0.15*veh_norm + 0.25*load_inv
  
  # Temporal features
  is_peak = 1 if (12<=order_hour<=14 or 19<=order_hour<=22) else 0
  day_of_week=2; month=3; is_weekend=0
  age_b = 0 if age<=25 else (1 if age<=30 else (2 if age<=35 else 3))
  
  # Build feature dictionary
  feature_dict = {"Delivery_person_Age":age,"Delivery_person_Ratings":rating,"Vehicle_condition":vehicle_cond,"multiple_deliveries":multi_del,"efficiency_score":efficiency,"distance_km":distance,"bearing_angle":bearing,"order_hour":order_hour,"order_day_of_week":day_of_week,"order_month":month,"is_peak_hour":is_peak,"is_weekend":is_weekend,"order_prep_time":prep_time,"traffic_encoded":traffic_enc,"weather_encoded":weather_enc,"festival_encoded":festival_enc,"distance_x_traffic":distance*traffic_enc,"weather_x_distance":weather_enc*distance,"age_bin":age_b}
  
  # One-hot encode categorical features
  for c in ["Metropolitian","Semi-Urban","Urban"]: feature_dict[f"city_{c}"] = 1 if city==c else 0
  for v in ["bicycle","electric_scooter","motorcycle","scooter"]: feature_dict[f"vehicle_{v}"] = 1 if vehicle_type==v else 0
  for o in ["Buffet","Drinks","Meal","Snack"]: feature_dict[f"order_{o}"] = 1 if order_type==o else 0
  
  # Prepare input and predict
  feature_values = [feature_dict.get(f, 0) for f in feature_names]
  X_input = np.array(feature_values).reshape(1, -1)
  X_input_scaled = scaler.transform(X_input)
  prediction = model.predict(X_input_scaled)[0]
  
  # Display results
  margin = 4
  st.markdown(f'<div class="prediction-box"><h2>Estimated Delivery Time</h2><h1 style="font-size:3.5rem;margin:0.5rem 0;">{prediction:.0f} minutes</h1><p style="font-size:1.2rem;">90%% CI: {max(0,prediction-margin):.0f} \u2013 {prediction+margin:.0f} minutes</p></div>', unsafe_allow_html=True)
  c1,c2,c3 = st.columns(3)
  with c1: st.metric("Distance", f"{distance:.2f} km")
  with c2: st.metric("Bearing", f"{bearing:.1f}\u00b0")
  with c3: st.metric("Efficiency", f"{efficiency:.3f}")
