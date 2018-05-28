# ee298-deep_learning
Submissions for EE 298 - Deep Learning course

Documentation for the final project is written below.

## Autoencoding beyond pixels
An implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf) using Keras.
John Rufino Macasaet & Martin Roy Nabus (CoE 197-Z/EE 298)

## Architecture
### Encoder
![Encoder](https://s3-ap-southeast-1.amazonaws.com/celebadataset/Original/vae_cnn_encoder.png)
### Decoder
![Decoder](https://i.imgur.com/TD3yVEo.png)
### Discriminator
![Discriminator](https://raw.githubusercontent.com/mrnabus/ee298-deep_learning/master/pics/vae_gan_disc_orig.png)

## Dataset
The dataset used for the training was the cropped and aligned version of the [Celeb-A Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The dataset features 202,599 images with 10,177 unique identities and annotations for 5 landmark locations and 40 binary attributes per image.
![celeba](http://mmlab.ie.cuhk.edu.hk/projects/celeba/intro.png)

## Using the code
### Dataset Folder
The expected location of the dataset is at `../img/`, i.e. the folder containing the code should be in the same directory as the folder containing the images.
### Training the model
To start the training:
```
python3 vae_gan.py
```
To resume training from a previous training instance:
```
python3 vae_gan.py continue
```
The model weights files (encoder, decoder, discriminator) must be on the same directory as their corresponding Python code for the continuation of the training to work. For example, the default names for the weights files of `vae_gan.py` are `vae_gan_enc.h5`, `vae_gan_dec.h5`, and `vae_gan_disc.h5`. Run details such as number of epochs and checkpointing are handled by editing code variables manually.
### Testing the VAE-GAN
To test the model generated by the VAE-GAN:
```
python3 vae_gan.py test
```
The encoder & decoder weights files must be on the same directory as the Python code for this to work
### Generation
To generate images from noise:
```
python3 vae_gan.py generate
```
For this case, only the corresponding decoder weights file is needed.
## Results
Following the recommendations of the paper regarding the architecture of the encoder, decoder/generator, and discriminator, some of the images obtained from the trained models are shown below.
### Results of the VAE (encoder+decoder models)
![VAE results](https://raw.githubusercontent.com/mrnabus/ee298-deep_learning/master/pics/results_vae.jpg)

### Results of the GAN (decoder+discriminator models)
Unfortunately, the models following the recommended GAN architecture did not yield face images. While models with a few epochs of training did show varying generated images, models with extended training (2000+ epochs) eventually ended up generating a single image over and over again. The code for training the model is in the folder `gan_cnn`, alongside two saved models. To test the generated models, run `gan_test.py name_of_model`; replace `name_of_model` with whichever model you wish to use to generate images.

The failure of the GAN to work on its own may have caused the combined model to fail. Regardless, a copy of the code can be found in the folder `vae_gan_original` for your perusal.

To address the non-functionality of the combined model, the discriminator was modified:
1. LeakyReLU was used instead of ReLU, and Dropout was added
2. This modified discriminator uses Conv-LeakyReLU-Dropout-BNorm instead of Conv-BNorm-ReLU
3. The Lth layer was entirely removed from the model; instead of relying from the Lth layer information to train the encoder and decoder, the actual image info was used instead.

An image of the modified discriminator is shown below.
![Modified Discriminator](https://raw.githubusercontent.com/mrnabus/ee298-deep_learning/master/pics/vae_gan_discriminator.png)

Using the modified discriminator, three models were created: a VAE-GAN, a Conditional VAE-GAN based on the attribute vectors on `list_attr_celeba.txt`, and a query-to-image VAE-GAN that accepts an attribute vector and outputs an image corresponding to that vector. Take note that for codes that use attribute vector data, `list_attr_celeba.txt` should be in the same directory as the `img` folder (not inside it).

### Results of the modified VAE-GAN
![VAE-GAN results](https://raw.githubusercontent.com/mrnabus/ee298-deep_learning/master/pics/results_vaegan.png)

### Results of the modified Conditional VAE-GAN
The images below are results of slightly altering the images' corresponding attribute vectors; for example, under the "Bald" category, Bald is set to 1 while all other hair attributes are set to -1.
![CVG results](https://raw.githubusercontent.com/mrnabus/ee298-deep_learning/master/pics/results_cvg.jpg)

### Results of the modified Query-to-image VAE-GAN
![QVG results](https://raw.githubusercontent.com/mrnabus/ee298-deep_learning/master/pics/results_qvg.png)

Overall, the results were not as good as hoped, but it is evident that some features still carry over the reconstructions; this can also be seen with the labels affecting the resulting images.

## Recommendations and Pitfalls
1. While training on the deep learning machines provided, we noticed that the machines were not configured to use `tensorflow-gpu` (i.e. it cannot access NVIDIA cuDNN). This hampered our capability to perform more tests.
2. Some configurations in the model's architecture were not mentioned on the paper; one of the most crucial examples is how the encoder's reparameterization network is defined. This may have affected our original implementation.
3. Using LeakyReLU seems to be more effective than using plain ReLU, as shown by the results in our experiments and in other groups' experiments.
