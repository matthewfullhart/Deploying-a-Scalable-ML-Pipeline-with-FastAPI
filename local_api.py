import json

import requests

URL_def = "http://127.0.0.1:8000/"

# Send a GET using the URL http://127.0.0.1:8000
r = requests.get(URL_def)

# Print the status code
print(f"Status Code: {r.status_code}")
# Print the welcome message
print(f"Get request reply: {r.json()}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST using the data above
r = requests.post((URL_def + "/data/"), json = data)

# Print the status code
print(f"Status Code: {r.status_code}")
# Print the result
print(f"Result: {r.json()}")
