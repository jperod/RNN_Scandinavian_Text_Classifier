import requests
from flask import jsonify

sentence = 'Hej, jeg hedder Pedro og jeg elsker at drikke øl!'
url_2 = '/'+sentence
resp = requests.get('http://localhost:5000/predict'+url_2)
