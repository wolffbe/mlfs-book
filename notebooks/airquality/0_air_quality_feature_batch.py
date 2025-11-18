import csv
import papermill as pm
import argparse

parser = argparse.ArgumentParser(description="Run a notebook with parameters from a CSV file.")
parser.add_argument("--notebook", required=True, help="Path to the input notebook")
parser.add_argument("--csvfile", required=True, help="Path to the CSV file containing parameters")
args = parser.parse_args()

with open(args.csvfile) as f:
    reader = csv.DictReader(f)
    for row in reader:
        city = row["city"]
        borough = row["borough"]
        country = row["country"]
        aqicn_url = row["aqicn_url"]
        csv_path = row["csv"]

        print(f"Running notebook with: city={city}, borough={borough}, country={country}, aqicn_url={aqicn_url}, csv={csv_path}")

        out_name = f"out_{city}_{borough}.ipynb"

        pm.execute_notebook(
            args.notebook,
            out_name,
            parameters={
                "city": city,
                "borough": borough,
                "country": country,
                "aqicn_url": aqicn_url,
                "csv": csv_path
            }
        )
