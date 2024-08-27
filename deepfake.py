import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Conv2D, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from PIL import Image

# 이미지 전처리 함수
def preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return None
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

# 생성자 모델 정의
def build_generator(latent_dim):
    model = Sequential([
        Input(shape=(latent_dim,)),
        Dense(128 * 16 * 16),
        Reshape((16, 16, 128)),
        Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', activation='tanh')
    ])
    return model

def build_discriminator(image_shape=(64, 64, 3)):
    model = Sequential([
        Input(shape=image_shape),
        Conv2D(64, (3,3), strides=(2,2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.4),
        Conv2D(64, (3,3), strides=(2,2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.4),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model



# GAN 모델 구축
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

# GAN 트레이닝 함수
def train_gan(gan, generator, discriminator, latent_dim, images, n_epochs=10000, n_batch=128):
    half_batch = int(n_batch / 2)
    for epoch in range(n_epochs):
        idx = np.random.randint(0, images.shape[0], half_batch)
        real_images = images[idx]
        real_labels = np.ones((half_batch, 1))
        discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))
        discriminator_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        gan_labels = np.ones((half_batch, 1))
        gan_loss = gan.train_on_batch(noise, gan_labels)

# 딥페이크 이미지 생성
def generate_fake_images(generator, latent_dim, n_images):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    fake_images = generator.predict(noise)
    return fake_images

# 이미지 저장 함수
def save_image(image_array, filename):
    image = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(filename)

# 이미지 데이터 로드 및 전처리
image_directory = "no"
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('png', 'jpg', 'jpeg'))]

images = []
for image_file in image_files:
    image = preprocess_image(image_file, target_size=(64, 64))
    if image is not None:
        images.append(image)

images = np.array(images)

# 모델 초기화 및 트레이닝
latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator((64, 64, 3))
gan = build_gan(generator, discriminator)
train_gan(gan, generator, discriminator, latent_dim, images)

# 딥페이크 이미지 생성 및 저장 예시
fake_images = generate_fake_images(generator, latent_dim, 10)
for i, img in enumerate(fake_images):
    save_image(img, f"fake_image_{i}.png")
