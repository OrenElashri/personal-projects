import qrcode
from PIL import Image
import os

print("\n=== QR Code Generator ===\n")

# User input
data = input("Paste website/link/text: ")

transparent = input("Transparent background? (y/n): ").lower()

box_size = input("Choose QR size (recommended 5-20): ")

# Default size if empty
if box_size.strip() == "":
    box_size = 10

box_size = int(box_size)

# Create QR
qr = qrcode.QRCode(
    version=None,  # automatic smallest version
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=box_size,
    border=1,
)

qr.add_data(data)
qr.make(fit=True)

# Create image
img = qr.make_image(
    fill_color="black",
    back_color="white"
).convert("RGBA")

# Make background transparent if chosen
if transparent == "y":

    datas = img.getdata()
    new_data = []

    for item in datas:
        # White -> transparent
        if item[:3] == (255, 255, 255):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)

# Save in SAME directory as script
script_dir = os.path.dirname(os.path.abspath(__file__))

filename = "qrcode.png"

save_path = os.path.join(script_dir, filename)

img.save(save_path)

print(f"\nQR code saved successfully:")
print(save_path)