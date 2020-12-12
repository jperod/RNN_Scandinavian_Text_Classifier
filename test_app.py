import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})