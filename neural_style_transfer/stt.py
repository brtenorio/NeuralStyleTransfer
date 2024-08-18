import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.applications import VGG16

class StyleTransfer:
    def __init__(self):
        self.model = None
        self.optimizer = None

    def deprocess_image(self, img, target_size):
        """Revert the preprocessing by VGG19."""
        img = img.reshape(target_size)
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype("uint8")
        return img

    def load_image(self, base_image_proc, style_reference_image_proc):
        base_image = K.constant(base_image_proc)
        style_reference_image = K.constant(style_reference_image_proc)
        combination_image = K.variable(base_image)
        return base_image, style_reference_image, combination_image

    def input_tensor_concat(self, base_image, style_reference_image, combination_image):
        """Helper function to concatenate images into a single tensor."""
        return K.concatenate([base_image, style_reference_image, combination_image], axis=0)

    def load_model(self, target_size):
        """Load the VGG19 model."""
        submodel = VGG16(input_shape=target_size, weights='imagenet', include_top=False)
        outputs_dict = dict([(layer.name, layer.output) for layer in submodel.layers])
        self.model = tf.keras.Model(inputs=submodel.inputs, outputs=outputs_dict)

    def content_loss(self, base, generated):
        """Calculate the content loss."""
        return K.sum(K.square(generated - base))

    def gram_matrix(self, x):
        """Calculate the Gram matrix."""
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    def style_loss(self, style, generated, img_height, img_width):
        """Calculate the style loss."""
        S = self.gram_matrix(style)
        G = self.gram_matrix(generated)
        channels = 3
        size = img_height * img_width
        return K.sum(K.square(S - G)) / (4.0 * (channels ** 2) * (size ** 2))

    def total_variation_loss(self, x, img_height, img_width):
        """Calculate the total variation loss."""
        a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
        b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    def compute_loss(self, base_image, style_reference_image, combination_image, img_height, img_width):
        """Compute the total loss of the combination image."""
        style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        content_layer_name = "block5_conv3"
        total_variation_weight = 1e-6
        style_weight = 1e-4
        content_weight = 1e-8

        input_tensor = self.input_tensor_concat(base_image, style_reference_image, combination_image)
        
        features = self.model(input_tensor)
        loss = tf.zeros(shape=())

        layer_features = features[content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += content_weight * self.content_loss(base_image_features, combination_features)

        for layer_name in style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            loss += (style_weight / len(style_layer_names)) * self.style_loss(
                style_reference_features, combination_features, img_height, img_width
            )

        loss += total_variation_weight * self.total_variation_loss(combination_image, img_height, img_width)
        return loss

    @tf.function
    def train_step(self, base_image, style_reference_image, combination_image, img_height, img_width):
        """Compute the gradients of the loss with respect to the combination image."""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(base_image, style_reference_image, combination_image, img_height, img_width)
        grads = tape.gradient(loss, combination_image)
        self.optimizer.apply_gradients([(grads, combination_image)])
        return loss, grads

    def minimize(self, img_height, img_width, base_image_proc, style_reference_image_proc, iterations=40):
        """Minimize the loss by updating the combination image.
        img_height: the height of the image
        img_width: the width of the image
        base_image_proc: the preprocessed base image
        style_reference_image_proc: the preprocessed style reference image
        iterations: the number of iterations to run the optimization
        """

        self.load_model((img_height, img_width, 3))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=10.0, decay_steps=10, decay_rate=0.9
        )
        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule)

        base_image, style_reference_image, combination_image = self.load_image(base_image_proc, style_reference_image_proc)

        for i in range(1, iterations + 1):
            loss, grads = self.train_step(base_image, style_reference_image, combination_image, img_height, img_width)
            if i % 10 == 0:
                print("Iteration %d: loss=%.2f" % (i, loss))

        img = self.deprocess_image(combination_image.numpy(), (img_height, img_width, 3))
        return img

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime
    from config import base_image_proc, style_reference_image_proc, img_height, img_width

    style_transfer = StyleTransfer()

    time = datetime.now()

    img = style_transfer.minimize(img_height, img_width, base_image_proc, \
                                  style_reference_image_proc, iterations=20)

    print("Time taken: ", datetime.now() - time)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
