import qrcode

# Site URL/Txt
data = "https://orenelashri.carrd.co/"

# QR creation
qr = qrcode.QRCode(
    version=1,  # size
    error_correction=qrcode.constants.ERROR_CORRECT_L,  
    box_size=10,  # Square size
    border=4,  # White border size
)

# Adding the data to the QR
qr.add_data(data)
qr.make(fit=True)

# Photo creation
img = qr.make_image(fill_color="black", back_color="white")
img.save("qrcode.png")

print("QR code named 'qrcode.png' saved")
