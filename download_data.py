import os
import subprocess
from pathlib import Path

from src.config.settings import settings


def main():

    TARGET_DIR = settings.NLB_DATA_RAW
    print(f"Downloading Neural Latents Benchmark datasets into: {TARGET_DIR}")
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    os.chdir(TARGET_DIR)
    print()

    # Note: with the --existing=refresh option, dandi will check if an existing file is
    # the newest version. If it is, it will skip downloading it. If not, it will download
    # the new version.

    print("Downloading MC_Maze [#000128] via dandi (1/7)")
    subprocess.run(['dandi', 'download', '--existing=refresh', 'https://dandiarchive.org/dandiset/000128/draft'])
    print()

    print("Downloading MC_RTT [#000129] via dandi (2/7)")
    subprocess.run(["dandi", "download", "--existing=refresh", "https://dandiarchive.org/dandiset/000129/draft"])
    print()

    print("Downloading Area2_Bump [#000127] via dandi (3/7)")
    subprocess.run(["dandi", "download", "--existing=refresh", "https://dandiarchive.org/dandiset/000127/draft"])
    print()

    print("Downloading DMFC_RSG [#000130] via dandi (4/7)")
    subprocess.run(["dandi", "download", "--existing=refresh", "https://dandiarchive.org/dandiset/000130/draft"])
    print()

    print("Downloading MC_Maze_Large [#000138] via dandi (5/7)")
    subprocess.run(["dandi", "download", "--existing=refresh", "https://dandiarchive.org/dandiset/000138/draft"])
    print()

    print("Downloading MC_Maze_Medium [#000139] via dandi (6/7)")
    subprocess.run(["dandi", "download", "--existing=refresh", "https://dandiarchive.org/dandiset/000139/draft"])
    print()

    print("Downloading MC_Maze_Small [#000140] via dandi (7/7)")
    subprocess.run(["dandi", "download", "--existing=refresh", "https://dandiarchive.org/dandiset/000140/draft"])
    print()

    print("Success!")


if __name__=="__main__":
    main()