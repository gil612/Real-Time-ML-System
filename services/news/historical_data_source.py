import os
import tempfile
import requests
from pathlib import Path
import rarfile
from typing import Optional

def dowload_and_extract_rar_file(url_rar_file: str) -> Optional[str]:
    """
    Downloads a RAR file from a URL, extracts it, and returns the path to the first file inside.
    
    Args:
        url_rar_file: URL of the RAR file to download
        
    Returns:
        str: Path to the first extracted file, or None if operation fails
    """
    try:
        # Create a temporary directory for our files
        temp_dir = tempfile.mkdtemp()
        rar_path = os.path.join(temp_dir, "downloaded.rar")
        
        # Download the file
        response = requests.get(url_rar_file, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the RAR file
        with open(rar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the RAR file
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(temp_dir)
            # Get the first file from the archive
            first_file = rf.namelist()[0]
            extracted_path = os.path.join(temp_dir, first_file)
            
            # If the extracted item is a directory, find the first file in it
            if os.path.isdir(extracted_path):
                for root, _, files in os.walk(extracted_path):
                    if files:  # Return the first file found
                        return os.path.join(root, files[0])
            return extracted_path
            
    except Exception as e:
        print(f"Error processing RAR file: {str(e)}")
        return None

# 
from quixstreams.sources import CSVSource

def get_historical_data_source(
    url_rar_file: str,
) -> CSVSource:
    # Download and extract the file
    path_to_csv_file = dowload_and_extract_rar_file(url_rar_file)
    if not path_to_csv_file:
        raise ValueError("Failed to download or extract the RAR file")
        
    return CSVSource(
        name="historical_crypto_news",  # A descriptive name for the source
        path=path_to_csv_file
    )

if __name__ == "__main__":
    # Use the raw content URL instead of the GitHub web page
    url_rar_file = "https://raw.githubusercontent.com/soheilrahsaz/cryptoNewsDataset/main/CryptoNewsDataset_csvOutput.rar"
    get_historical_data_source(url_rar_file)
