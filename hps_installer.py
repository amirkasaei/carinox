import os
import requests
import shutil

# Function to download and move the missing file
def download_bpe_vocab_file():
    url = 'https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz'
    
    # Dynamically get the package directory
    try:
        import hpsv2
        package_path = os.path.join(os.path.dirname(hpsv2.__file__), "src", "open_clip")
    except ImportError:
        print("hpsv2 package is not installed or cannot be imported. Please check your environment.")
        return

    filename = 'bpe_simple_vocab_16e6.txt.gz'
    destination_path = os.path.join(package_path, filename)

    # Check if the file already exists
    if not os.path.exists(destination_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the file temporarily
            temp_file = os.path.join(os.getcwd(), filename)
            with open(temp_file, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            print(f"Downloaded {filename} successfully.")

            # Create destination directory if it doesn't exist
            if not os.path.exists(package_path):
                os.makedirs(package_path)

            # Move the file to the correct path
            shutil.move(temp_file, destination_path)
            print(f"Moved {filename} to {package_path}.")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    else:
        print(f"{filename} already exists in {package_path}.")

# Call the function to ensure the file is available before model creation
download_bpe_vocab_file()
