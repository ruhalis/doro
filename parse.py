import requests
from bs4 import BeautifulSoup
import os
import time

# Base URL for gallery
base_url = "https://www.ljcmedspa.com/gallery/injectable-services/botox-cosmetic/"
session = requests.Session()

# Make a request to the gallery page
response = session.get(base_url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    # Directory to save images
    os.makedirs("botox_webp_images", exist_ok=True)

    # Find all patient links (adjust the selector based on actual HTML)
    for link in soup.find_all('a', text="Read More"):
        patient_url = link.get('href')
        if not patient_url.startswith("http"):
            patient_url = "https://www.ljcmedspa.com" + patient_url

        # Go to each patient's page
        patient_response = session.get(patient_url)
        if patient_response.status_code == 200:
            patient_soup = BeautifulSoup(patient_response.text, 'html.parser')

            # Find all .webp images and download them
            for img_tag in patient_soup.find_all('img', src=True):
                img_url = img_tag['src']
                if img_url.endswith('.webp'):
                    if not img_url.startswith("http"):
                        img_url = "https://www.ljcmedspa.com" + img_url

                    # Download the image
                    img_data = session.get(img_url).content
                    img_name = os.path.join("botox_webp_images", os.path.basename(img_url))
                    with open(img_name, 'wb') as img_file:
                        img_file.write(img_data)
                    print(f"Downloaded {img_name}")

            # Be polite and avoid spamming the server
            time.sleep(1)
else:
    print("Failed to access the gallery page.")
