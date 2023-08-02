import requests

# response = requests.get("https://string-db.org/api/tsv/homology?identifiers=[your_identifiers]")
# https://string-db.org/api/tsv/homology?identifiers=CDK1%0dCDK2

response = requests.get("https://string-db.org/api/tsv/homology?identifiers=A0A009IHW8")
print(response)