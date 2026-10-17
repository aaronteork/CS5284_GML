import zipfile

import pandas as pd
import torch


def read_data(drive_path, local_path):
    # Extract zip file
    # Move to SSD for faster read and write
    # The original downloaded MD file does not have a subfolder so this is a re-zip
    zip_files = sorted(drive_path.glob("*.zip"))
    print(f"Found {len(zip_files)} zip file(s): {[f.name for f in zip_files]}")

    for zip_path in zip_files:
        print(f"Extracting {zip_path.name} to {local_path}...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(local_path)
        print(f"Finished extracting {zip_path.name}!")

    # Read data
    data = {
        "DE": {"filepath": local_path / "DE"},
        "MD": {"filepath": local_path / "MD"},
    }

    for state, info in data.items():
        info["adj"] = torch.load(
            info["filepath"] / "adj_matrix.pt", map_location="cpu", weights_only=False
        ).coalesce()
        diff = (info["adj"] - info["adj"].t()).coalesce()
        is_symmetric = torch.all(diff.values() == 0).item()
        print(f"{state}: Symmetric = {is_symmetric}")

        info["edges"] = torch.load(
            info["filepath"] / "Edges" / "edge_features.pt",
            map_location="cpu",
            weights_only=False,
        )

        info["edge_features_traffic"] = {}
        for traffic_file in sorted(
            (info["filepath"] / "Edges").glob("edge_features_traffic_*.pt")
        ):
            # Extract the year from the filename (e.g., 'edge_features_traffic_2019.pt' -> 2019)
            year = int(traffic_file.stem.split("_")[-1])
            info["edge_features_traffic"][year] = torch.load(
                traffic_file, map_location="cpu", weights_only=False
            )

        info["accidents"] = pd.read_csv(info["filepath"] / "accidents_monthly.csv")

    return data
