import requests

url = "https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv"
output_path = "data/raw/raw.csv"

response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

with open(output_path, "wb") as file:
    file.write(response.content)

print(f"File saved to {output_path}")
