import json
import csv

try:
    # Load JSON data from file
    with open('MadridPlaces.json', 'r') as json_file:
        data = json.load(json_file)

    if not isinstance(data, dict) or not data:
        raise ValueError("JSON data is not in the expected format or is empty.")

    # Extract column headers from JSON keys
    headers = list(data.keys())

    # Write data to CSV
    with open('MadridPlaces.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerow(data)

except Exception as e:
    print(f"An error occurred: {e}")
