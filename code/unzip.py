import zipfile
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("shortlist.csv")
fp = "/home/rnanawa1/GeoLifeLearn/output/SatellitePatches/PO-Train-SatellitePatches-RGB.zip"

# extract only the files from df[img_path]
with zipfile.ZipFile(fp, "r") as z:
    for img_path in tqdm(df["img_path"]):
        z.extract(
            img_path,
            path="/home/rnanawa1/GeoLifeLearn/data/",
        )
