import os 
from PIL import Image

image_names = [name for name in os.listdir('.') if os.path.splitext(name)[1] == ".png"]

to_image = Image.new('RGB', (640, 550))

for y in range(1, 3):
    for x in range(1, 3):
        from_image = Image.open(image_names[2*(y-1)+x-1]).resize((320,275),Image.ANTIALIAS)
        to_image.paste(from_image, ((x-1)*320, (y-1)*275))
        to_image.save('res.jpg')
