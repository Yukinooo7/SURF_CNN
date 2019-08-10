# import the Python Image processing Library

from PIL import Image

# Create an Image object from an Image

imageObject = Image.open("E:\\Study\\2019-Summer\\SURF\\CNN\\data\\kaggle\\train\\cat.0.jpg")

imageObject.show()

# Crop the iceberg portion

cropped = imageObject.crop((100, 30, 400, 330))  # The box is a 4-tuple defining the left, upper, right, and lower
# Display the cropped portion

cropped.show()
