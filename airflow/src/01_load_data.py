# libraries to import
import requests
import json
import os
from datetime import datetime, UTC
import logging


def fetch_data():
    # input: API key and cities list
    api_key = ""
    cities = ["recife", "salvador", "natal"]

    weather_data = []
    timestamp = None

    # Loop over cities and fetch weather data
    for city in cities:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        )
        response = requests.get(url)

        if response.status_code == 200:
            city_data = response.json()
            # Add city data to weather_data
            weather_data.append(city_data)

            # Set the timestamp to the most recent city data timestamp
            if timestamp is None:
                timestamp = city_data["dt"]

            print(f"Data for {city} received!")
            print(f"Timestamp: {timestamp}")
        else:
            print(
                f"Error fetching data for {city}: {response.status_code}, {response.text}"
            )

    # Test directory setup
    output_dir = "./raw_files/"
    os.makedirs(output_dir, exist_ok=True)

    # Convert timestamp to formatted string
    if timestamp:
        formatted_time = datetime.fromtimestamp(timestamp, UTC).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Create filename
        filename = os.path.join(output_dir, f"{formatted_time}.json")

        # Write data to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(weather_data, f, ensure_ascii=False, indent=4)

        print(f"The data of the three cities has been saved to {filename}! :-)")
    else:
        print("No valid data retrieved. File was not created :-(.")


fetch_data()
