import requests

r = requests.get("http://127.0.0.1:5000/report/csv")
print("Status:", r.status_code)
print("Server header:", r.headers.get("Server"))
print("Text:", r.text)
