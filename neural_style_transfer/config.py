from PIL import Image
import numpy as np
import tensorflow as tf

# Define function to make global variables img_height, img_width
content = Image.open('pipoca.jpg')
style = Image.open('style_pavao.jpg')

def get_img_size(content_image):
	original_width, original_height = content_image.size
	img_height = 400
	img_width = round(original_width * img_height / original_height)
	return img_height, img_width
	
	
def preprocess_image(image_loaded, target_size):
	img = image_loaded.resize(target_size)
	img = np.array(img)
	img = np.expand_dims(img, axis=0)
	img = tf.keras.applications.vgg19.preprocess_input(img)
	return img

img_height, img_width = get_img_size(content)
# Preprocess the images
base_image_proc = preprocess_image(content, target_size=(img_width, img_height))
style_reference_image_proc = preprocess_image(style, target_size=(img_width, img_height))