from pathlib import Path
import csv
import rarfile
import wget
from quixstreams.sources import CSVSource

HistoricalNewsDataSource = CSVSource


def download_and_extract_rar_file(url_rar_file: str) -> str:
    """
    Downloads a RAR file from a URL, extracts it, and returns the path to the first CSV file found.

    Args:
        url_rar_file (str): URL of the RAR file to download

    Returns:
        str: Path to the extracted CSV file

    Raises:
        Exception: If no CSV file is found in the RAR archive
    """
    # Create a temporary directory for downloads if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    # Download the RAR file using wget
    rar_path = temp_dir / "download.rar"
    wget.download(url_rar_file, str(rar_path))

    # Extract the RAR file
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(temp_dir)

        # Find the first CSV file in the extracted contents
        for file in rf.namelist():
            if file.lower().endswith(".csv"):
                return str(temp_dir / file)

    raise Exception("No CSV file found in the RAR archive")


def get_historical_data_source() -> CSVSource:
    """
    Creates a CSVSource for historical news data.
    Uses a pre-downloaded CSV file from the news-signal service.
    """
    path_to_csv_file = Path("../news-signal/data/cryptopanic_news.csv")

    # Verify the file exists
    if not path_to_csv_file.exists():
        raise FileNotFoundError(f"CSV file not found at {path_to_csv_file}")

    # Verify we can read the file
    try:
        with open(path_to_csv_file, "r", encoding="utf-8") as f:
            csv.reader(f)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")

    return CSVSource(path=str(path_to_csv_file), name="historical_news")
