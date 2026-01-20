## End to End Machine learning project on Airline's passengers satisfaction

### Airline Passenger Satisfaction — Prediction API (Flask) + Azure Deployment
Predict whether an airline passenger will be Satisfied vs Neutral or Dissatisfied from post-flight survey and operational signals (service ratings + delays), then expose the model through a Flask web app / REST API for demos and integration.

### Business context (airline)
Airlines can use this model in two ways:
- Quality analytics: identify the strongest levers of satisfaction (e.g., digital journey vs cabin service) to prioritize improvements.​
- Service recovery: flag high-risk passengers for proactive recovery actions (voucher, apology, priority handling) based on predicted probability

### Dataset
This project is built on the “Airline Passenger Satisfaction” dataset from Kaggle (binary target: satisfaction).

### Target definition
- satisfaction = "satisfied" → 1
- satisfaction = "neutral or dissatisfied" → 0

### Model inputs
The inference pipeline expects the same feature names used during training (important: exact spelling/casing).

Typical inputs include:
- Passenger profile: Gender, Customer Type, Age, Type of Travel, Class
- Flight ops: Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes
- Service ratings (0–5): e.g., Inflight wifi service, Online boarding, Seat comfort, Cleanliness, etc.

