# ee298-deep_learning
Submissions for EE 298 - Deep Learning course

## Autoencoding beyond pixels
An implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf) using Keras. 
John Rufino Macasaet & Martin Roy Nabus (CoE 197-Z/EE 298)

## Architecture
### Encoder
![Encoder](https://s3-ap-southeast-1.amazonaws.com/celebadataset/Original/vae_cnn_encoder.png)
### Decoder
![Decoder](https://i.imgur.com/TD3yVEo.png)
### Discriminator
![Discriminator](https://i.imgur.com/GF9GiVF.png)
### Generator Model
![Generator](https://i.imgur.com/L0SnRCm.png)
### Encoder and Decoder Trainer (VAE)
![Encoder and Decoder Trainer (VAE)](https://i.imgur.com/zjtVQvQ.png)

## Dataset
The dataset used for the training was the cropped and aligned version of the [Celeb-A Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The dataset features 202,599 images with 10,177 unique identities and annotations for 5 landmark locations and 40 binary attributes per image.
![celeba](http://mmlab.ie.cuhk.edu.hk/projects/celeba/intro.png)

## Using the code
### Dataset Folder
The expected location of the dataset is at `./img/`
### Training the model
To start the training:
```
python3 vae_gan.py
```
To resume training from a previous training instance:
```
python3 vae_gan.py continue
```

### Testing the VAE
To test the functionality of the VAE:
```
python3 vae_gan.py test
```

### Generation
To generate images from noise:
```
python3 vae_gan.py generate
```

## Results
Following the recommendations of the paper regarding the architecture of the encoder, decoder/generator, and discriminator, some of the images obtained from the trained models are shown below.
### Results of the VAE (encoder+decoder models)

### Results of the GAN (decoder+discriminator models)

Unfortunately, the combined model did not generate face images. Regardless, a copy of the code can be found in the folder `vae_gan_original` for your perusal.
To address the non-functionality of the combined model, the discriminator was modified:
1. LeakyReLU was used instead of ReLU, and Dropout was added
2. This modified discriminator uses Conv-LeakyReLU-DOut-BNorm instead of Conv-BNorm-ReLU
3. The Lth layer was entirely removed from the model; instead of relying from the Lth layer information to train the encoder and decoder, the actual image info was used instead.

### Results of the modified VAE-GAN

### Results of the modified Conditional VAE-GAN

### Results of the modified Query-to-image VAE-GAN
## Recommendations and Pitfalls
1. While training on the deep learning machines provided, we noticed that the machines were not configured to use `tensorflow-gpu` (i.e. it cannot access NVIDIA cuDNN). This hampered our capability to perform more tests.
2. Some configurations in the model's architecture were not mentioned on the paper; one of the most crucial examples is how the encoder's reparameterization network is defined. This may have been one of the reasons why our original implementation did not work.
3. Using LeakyReLU seems to be more effective than using plain ReLU, as shown by the results in our experiments and in other groups' experiments.
