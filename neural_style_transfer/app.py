if __name__=="__main__":
	import streamlit as st
	from PIL import Image
	import numpy as np
	import tensorflow as tf
	from stt import StyleTransfer
	
	#1. Create a streamlit title widget, this will be shown first
	st.title("Neural Style Transfer")
			
	upload1 = st.file_uploader('Insert Content image', type=['png','jpg'])
	upload2 = st.file_uploader('Insert Style image', type=['png','jpg'])

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

	c1, c2= st.columns(2)
	if (upload1 and upload2) is not None:
		# Open the image with PIL
		content = Image.open(upload1)
		style = Image.open(upload2)

		# Get the size of the images
		img_height, img_width = get_img_size(content)
		# Preprocess the images
		base_image_proc, style_reference_image_proc = \
			preprocess_image(content, target_size=(img_width, img_height)), \
			preprocess_image(style, target_size=(img_width, img_height))
		
		with st.spinner('Wait for it... It can take a while!'):
			style_transfer = StyleTransfer()
			img = style_transfer.minimize(img_height, img_width, base_image_proc, \
								style_reference_image_proc, iterations=30)
		st.success("Done!")


	
		c1.header('Content Image')
		c1.image(content)

		c1.header('Style Image')
		c1.image(style)
			
		c2.header('Processed Image')
		c2.image(img)
	
