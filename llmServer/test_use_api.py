import requests
import json

url = 'http://127.0.0.1:8000/chat/'

params = {"query": "你好", "history": [], }
html = requests.post(url, json.dumps(params))
response = json.loads(html.text)
print(response)

params = {"query": "睡不着", "history": [], }
html = requests.post(url, json.dumps(params))
response = json.loads(html.text)
print(response)

params = {"query": "早上好", "history": [], }
html = requests.post(url, json.dumps(params))
response = json.loads(html.text)
print(response)
