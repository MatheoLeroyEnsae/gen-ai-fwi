import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import display, Markdown
import os


class DCGAN_tuned(keras.Model):
    """
    A DCGAN model with tuned call method to handle prediction properly
    """

    version = "1.1"

    def __init__(self, discriminator=None, generator=None, latent_dim=100, **kwargs):
        """
        DCGAN_tuned instantiation with a given discriminator and generator
        args :
            discriminator : discriminator model
            generator : generator model
            latent_dim : latent space dimension
        return:
            None
        """
        super(DCGAN_tuned, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        print(f"Fidle DCGAN_tuned is ready :-)  latent dim = {latent_dim}")

    def call(self, inputs):
        """
        Implementation of the model forward pass
        args:
            inputs : vectors from latent space (can be a single tensor or a tuple for CGAN)
        return:
            output : Output of the generator
        """
        if isinstance(inputs, (list, tuple)):
            latent_vectors = inputs[0]
        else:
            latent_vectors = inputs

        outputs = self.generator(latent_vectors)
        return outputs

    def compile(
        self,
        discriminator_optimizer=keras.optimizers.Adam(),
        generator_optimizer=keras.optimizers.Adam(),
        loss_function=keras.losses.BinaryCrossentropy(),
    ):
        """
        Compile the model
        args:
            discriminator_optimizer : Discriminator optimizer (Adam)
            generator_optimizer : Generator optimizer (Adam)
            loss_function : Loss function
        """
        super(DCGAN_tuned, self).compile()
        self.discriminator.compile(
            optimizer=discriminator_optimizer, loss=loss_function
        )
        self.generator.compile(optimizer=generator_optimizer, loss=loss_function)

        self.d_optimizer = discriminator_optimizer
        self.g_optimizer = generator_optimizer
        self.loss_fn = loss_function
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, inputs):
        """
        Implementation of the training update.
        Receive some real images.
        This will compute loss, get gradients and update weights for generator and discriminator
        Return metrics.
        args:
            real_images : real images
        return:
            d_loss  : discriminator loss
            g_loss  : generator loss
        """

        if isinstance(inputs, tuple):
            real_images = inputs[0]
        else:
            real_images = inputs

        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        generated_images = self.generator(random_latent_vectors)

        combined_images = tf.concat([generated_images, real_images], axis=0)

        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(fake_images)
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    def save(self, filename):
        """Save model in 2 part"""
        save_dir = os.path.dirname(filename)
        filename, _extension = os.path.splitext(filename)
        os.makedirs(save_dir, mode=0o750, exist_ok=True)
        self.discriminator.save(f"{filename}-discriminator.h5")
        self.generator.save(f"{filename}-generator.h5")

    def reload(self, filename):
        """Reload a 2 part saved model.
        Note : to train it, you need to .compile() it..."""
        filename, extension = os.path.splitext(filename)
        self.discriminator = keras.models.load_model(
            f"{filename}-discriminator.h5", compile=False
        )
        self.generator = keras.models.load_model(
            f"{filename}-generator.h5", compile=False
        )
        print("Reloaded.")

    @classmethod
    def about(cls):
        """Basic whoami method"""
        display(Markdown("<br>**FIDLE 2022 - DCGAN_tuned**"))
        print("Version              :", cls.version)
        print("TensorFlow version   :", tf.__version__)
        print("Keras version        :", tf.keras.__version__)
