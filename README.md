# Generative Adversarial Networks (GANs)

![GANs](https://miro.medium.com/v2/resize:fit:1200/1*Wz5W33-C7vzBsoRq2ir_qg.png)

## Introduction

Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. GANs consist of two neural networks, a generator and a discriminator, which are trained simultaneously through adversarial processes.

## How GANs Work

1. **Generator**: This network generates new data instances that resemble the training data.
2. **Discriminator**: This network evaluates the authenticity of the data, distinguishing between real data (from the training set) and fake data (from the generator).

The training process involves:
- The generator trying to produce data that fools the discriminator.
- The discriminator trying to become better at distinguishing real data from fake data.

The goal is to reach a point where the discriminator can no longer distinguish between real and generated data.

## Architecture

### Generator

The generator takes a random noise vector as input and transforms it into a data sample (e.g., an image).

### Discriminator

The discriminator receives both real and generated data samples and outputs a probability indicating whether the sample is real or fake.

### Training Process

1. Sample random noise and generate fake data.
2. Train the discriminator with both real and fake data.
3. Train the generator via the feedback received from the discriminator.

This adversarial process is akin to a game where both players (generator and discriminator) improve over time.

## Applications

- **Image generation**: Creating high-resolution images.
- **Data augmentation**: Generating additional training data.
- **Art creation**: Producing new pieces of art.
- **Super-resolution**: Enhancing image resolution.

## Code Example

Here's a simple implementation using TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers

# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=100))
    model.add(layers.Dense(784, activation='sigmoid'))
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Create GAN
def build_gan(generator, discriminator):
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Instantiate models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Compile models
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Display model summaries
generator.summary()
discriminator.summary()
gan.summary()
