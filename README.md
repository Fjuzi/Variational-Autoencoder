# Variational-Autoencoder

### Part 1

![Image Generation](https://github.com/Fjuzi/Variational-Autoencoder/blob/master/images/subtask31.png)

Generation of 25 images from the generative model. This is done by drawing z from the prior, to then generate x, using the condition distribution p_theta(x|z)

### Part 2

Generation of 10 image reconstructions using the reconstruction model and then the generative model. The reconsturctions are obtained by generating z using q(z|x) and then generating x again using p(x|z).

![Image Generation](https://github.com/Fjuzi/Variational-Autoencoder/blob/master/images/subtask32.png)

### Part 3

Generate interpolations in the latent space from one image to an another. It is obtained by finding the latent representation of each image. 

![Image Generation](https://github.com/Fjuzi/Variational-Autoencoder/blob/master/images/subtask33_0with_1.png)
