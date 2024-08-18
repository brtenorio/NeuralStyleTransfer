import tensorflow as tf
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from keras.applications import VGG16
from config import base_image_proc, style_reference_image_proc, img_height, img_width

# Define the functions to preprocess and deprocess the images --------------------------------------------

# def preprocess_image(image_path, target_size=(img_height, img_width, 3)):
#     img = tf.keras.utils.load_img(
#         image_path, target_size=target_size)
#     img = tf.keras.utils.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = tf.keras.applications.vgg19.preprocess_input(img)
#     return img

def deprocess_image(img, target_size=(img_height, img_width, 3)):
    # revert the preprocessing by VGG19
    # remove zero-center by mean pixel value of ImageNet.
    img = img.reshape(target_size)
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # Also, convert back from 'BGR' to 'RGB'.
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def load_image(base_image_proc, style_reference_imag_proc):
    base_image = K.constant(base_image_proc)
    style_reference_image = K.constant(style_reference_imag_proc)
    combination_image = K.variable(base_image)
    return base_image, style_reference_image, combination_image


def input_tensor_concat(base_image, style_reference_image, combination_image):
    """ This is a helper function to concatenate the base_image, style_reference_image and
        combination_image into a single tensor"""
    
    return K.concatenate(
       [base_image, style_reference_image, combination_image], axis=0
    )

# Load the images --------------------------------------------
base_image, style_reference_image, combination_image = load_image(base_image_proc, style_reference_image_proc)


def load_model(target_size=(img_height, img_width, 3)):
    # Load the VGG19 model and set it to non-trainable
    submodel = VGG16(input_shape=target_size, weights='imagenet', include_top=False)
    
    outputs_dict = dict([(layer.name, layer.output) for layer in submodel.layers])
    model = tf.keras.Model(inputs=submodel.inputs, outputs=outputs_dict)
    return model

model = load_model()


# Define the loss functions --------------------------------------------

def content_loss(base, generated):
    """ content loss will make sure that the top layer of VGG16 has a
        similar view of the base image
        
        base: the activations of the intermediate layer of the base image
        generated: the activations of the intermediate layer of the generated image
    """
    
    return K.sum(K.square(generated - base))


def gram_matrix(x):
    """ the gram matrix of an image tensor (feature map) is the dot product of the
        reshaped tensor with its transpose"""
    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, generated, img_height=img_height, img_width=img_width):
    """ style loss is the sum of the squared differences between the gram matrices of the
        style image and the generated image"""
    
    S = gram_matrix(style)
    G = gram_matrix(generated)
    channels = 3
    size = img_height * img_width
    
    return K.sum(K.square(S - G)) / (4.0 * (channels ** 2) * (size ** 2))


def total_variation_loss(x, img_height=img_height, img_width=img_width):
    """ total variation loss is the sum of the squared differences between the pixels of the
        generated image and the pixels of the generated image shifted by one pixel in both
        the horizontal and vertical directions.
        This loss is used to smooth the generated image"""
    
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))


# Define the combined loss function --------------------------------------------

def compute_loss(base_image, style_reference_image, combination_image):
    """ compute the total loss of the combination image.

        combination_image : the image to be optimized.
        base_image : the image that will be used to compute the content loss.
        style_reference_image : the image that will be used to compute the style loss.
        returns the total loss as a scalar tensor.
    """
    # Define the names of the layers that will be used to compute the loss
    style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
    ]
    content_layer_name = "block5_conv3"

    # Define the weights in the weighted average for the loss components
    # You can play around with these weights to get different results
    total_variation_weight = 1e-6
    style_weight = 1e-4
    content_weight = 1e-8

    input_tensor = input_tensor_concat(base_image, style_reference_image, combination_image)

    print("BRN TEST", input_tensor.shape)
    print(model.summary())

    # features is a dictionary to capture the activations of the intermediate layers
    features = model(input_tensor) 
    loss =  tf.zeros(shape=()) # loss is a scalar tensor

    # get the activations of the content layer
    layer_features = features[content_layer_name] 
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    # The content loss
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    
    # The style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_loss_value = style_loss(
          style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * style_loss_value

    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

# Define the optimizer --------------------------------------------

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=10.,
    decay_steps=10,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule)


# Define the training step --------------------------------------------

@tf.function
def train_step(base_image, style_reference_image, combination_image):
    """ function to be used with tf.GradientTape to compute the gradients of the loss
        with respect to the combination image.
    """

    with tf.GradientTape() as tape:
        loss = compute_loss(base_image, style_reference_image, combination_image)
    
    # compute the gradients of the loss with respect to the combination_image
    grads = tape.gradient(loss, combination_image)
    #apply the gradients to the combination_image
    optimizer.apply_gradients([(grads, combination_image)])
    return loss, grads


# Define the function to minimize the loss --------------------------------------------

def minimize(n=40):
    """ function to minimize the loss by updating the combination image"""

    loss_values = []
    grad_values = []
    iterations = n
    for i in range(1, iterations+1):
        loss, grads = train_step(base_image, style_reference_image, combination_image)
        loss_values.append(loss)
        grad_values.append(grads)
    
        if i % 10 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))

    # deprocess the image, save it, and plot it
    img = deprocess_image(combination_image.numpy())

    return img


if __name__ == "__main__":

    time = datetime.now()

    img = minimize(20)

    print("Time taken: ", datetime.now() - time)

    plt.imshow(img)
    plt.axis("off")
    plt.show()