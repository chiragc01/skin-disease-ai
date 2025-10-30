import requests

url = "http://127.0.0.1:5000/predict"
file_path = "test_image.jpg"  # path to any image on your system

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
