import pandas as pd
from pathlib import Path
from lim_code.lim_logger import logger


def main(path="results"):
    for f in Path(path).glob("**/results.csv"):
        df = pd.read_csv(f)
        fps = sum(df["membership_fp"] > 0)
        if fps > 0:
            logger.info(f"There are {fps} false positives in the membership inference attack")

        tps = sum(df["membership_fp"] > 0)
        if tps > 0:
            logger.info(f"There are {tps} true positives in the membership inference attack")
