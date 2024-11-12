import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# URL of the webpage to scrape
url = "https://www.ljcmedspa.com/gallery/injectable-services/botox-cosmetic/"

# Directory to save images
save_dir = "webp_images"
os.makedirs(save_dir, exist_ok=True)

# Send a GET request to the webpage
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all image tags
    img_tags = soup.find_all('img')
    for img in img_tags:
        img_url = img.get('src')
        if img_url and img_url.endswith('.webp'):
            # Construct the full URL if necessary
            full_img_url = urljoin(url, img_url)
            # Download the image
            img_data = requests.get(full_img_url).content
            img_name = os.path.join(save_dir, os.path.basename(full_img_url))
            with open(img_name, 'wb') as f:
                f.write(img_data)
            print(f"Downloaded {img_name}")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
