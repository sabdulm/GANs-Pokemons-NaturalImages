# GANs-Pokemons-NaturalImages

## Introduction

The focus of this project is the understanding and the uses of Generative Adversarial Networks (GANs). The reason for choosing this topic is firstly, we did not cover such networks in our Machine Learning class and secondly these are a new class of neural networks that have emerged with numerous applications which for us seems particularly interesting.

GANs are a class of unsupervised networks that can generate data based on some intially labelled data using two sub models, a generator and a discriminator. As the name suggests, the generator learns patters from the input data and  generates new data which should resemble the original data. This output is then passed onto the discriminator which tries to classify the data as fake or real. Both models are trained continously until the generator is able to produce plausible outputs that can fool the discriminator to classify the generated data as real. GANs can be used in a variety of applications such as Text-to-Text, Image-to-Image, Image-to-Text translation etc.


For our experiments we have written a basic implementation of GAN that does Image-to-Image translation i.e. it generates new images based off the original images.

## Methods

The novel implementation of a GAN can be found in the paper "Generative Adverserial Networks" by Ian Goodfellow. Our GAN implementation is much simpler and is mostly based off code provided in https://towardsdatascience.com/image-generation-in-10-minutes-with-generative-adversarial-networks-c2afc56bfa3b. 

We have made certain changes to the above code to allow us to build upon our experiments. The modifications include a GAN model class that we can use to train, save and display the generator output images after each epoch. Furthermore we have tested our experiments against 3 different datasets and the findings for each data is listed below.

1. MNIST dataset
    * The dataset was imported from keras.datasets in the tensorflow library. It contains labelled images of  handwritten digits/numbers.
2. CFAR10 dataset
    * This dataset was also imported from keras.datasets in the tensorflow library. This dataset contains images from 10 different classes that are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. https://www.cs.toronto.edu/~kriz/cifar.html
3. Pokemon dataset
    * The pokemon dataset was downloaded from https://www.kaggle.com/kvpratama/pokemon-images-dataset. For this experiment we only use the jpg format images. The directory for this dataset is "pokemon/" with only the "pokemon_jpgs" as the only sub-directory.

    
An example run for each dataset can be found in [Mannan-Report.ipynb](https://github.com/sabdulm/GANs-Pokemons-NaturalImages/blob/main/Mannan-Report.ipynb). 
    
To test the GAN model we test out different number of epochs to train our model and determine what the optimal number of epochs for each training dataset. 

* Note: Before running any of the examples make sure there is a directory called `training_output/`

## Results

The results of all the experiments ran using the different datasets can be found in [Mannan-Report.ipynb](https://github.com/sabdulm/GANs-Pokemons-NaturalImages/blob/main/Mannan-Report.ipynb) again. Below are some of findings/insights gained from our experiments:

### Experiment results

Given the above outputs, it is clear we have some clear and interesting insights.

* Firstly, the MNIST dataset does show some images of numbers that can be passed off as a handwritten number, but for some images it does seem that the GAN model mixes one or more of the features from different numbers together. The same thing can be said for the CFAR dataset. It was able to generate new images that did show some interesting patterns. If we look at the top-right most image at epoch 500, it seems to be combination of car and horse in the same image while the second image in the second row seems to resemble that of a bird. 

* A few things to notice is how long it takes for our GAN model to generate some passable images.
    * for MNIST, it is apparent that MNIST can produce such results somewhere between 50-100 epochs. Any more than this, the changes in the individual images is insignificant. However, for the CFAR dataset it takes more number of epochs for the model to generate some output that starts to resemble some of the training data. And even after 350 epochs the output varies in each of the generated image. In the generated gif, we can also see that after every epoch some of the images generate a different class altogether, e.g. the image would be a car in one epoch and a train in the next. 
    
* The pokemon dataset was not able to produce any reasonable result. The generated outputs seem to either contain outlines of some form, random colors in each of the image or just a combination of the two. There were no reasonable patterns within each of the image that one could find to confidently say that the GAN model was able to generate new pokemon images. Even training for 500 epochs could produce a reasonable output.

* Also looking at the loss graphs printed, in the Methods section for the final epoch for each dataset we can conclude the following:
    * For the MNIST dataset, the deviations between the loss values for each batch is not very high. There is a certain pattern in the graph as well, when the discriminator loss spikes the generator loss takes a dip. This could mean that for batches with high discriminator loss the generator was able to create images that were able to fool the discriminator into thinking it was a real image. The viceversa hold true for this notion as well.
    * The same conclusions can also be reached for the CFAR dataset. However the deviations in the loss is much more eratic than that of MNIST. This could be due to the third dimension in the image data i.e. "color" as the generator and discriminator also have to factor in color in generating/discriminiting images.
    * For the pokemon dataset, the generator loss is much higher than the discriminator loss, and the generator loss when compared to the other models trained using the other datasets is much higher as well. This indicates that the generator is not able to learn any patterns/features from the data whereas the discriminator is easily able to tell which image is real. This could also be a factor in determining how well the model performed which coincedentally correlates to the results of the experiments on the pokemon data.

The reason that the pokemon dataset didnot perform as well could be two-fold:
* The dataset itself is not a good dataset. What we mean is that the MNIST and CFAR datasets have numerous examples for each class of image they have, whereas the pokemon dataset contains about 800 images with each image representing a unique pokemon. Therefore the number of training samples for our GANs model is highly lacking. 
* Our implementation of the GANs model is much simpler. It contains Convulational layers with LeakyRelu as the activation functions. Creating more sophisticated generator or discriminator models may improve the performance of our GAN model for the pokemon dataset. 

## Conclusions

Our implementation of GANs does show promise for certain datasets while for others it does not perform acceptably. Implementing a GAN model using tensorflow or pytorch is not a difficult task, all it depends is the kind of translation one is looking to perform. Some of the issues we faced and possible improvements we could perform are as follows:

* For example, for Image-to-Image translation our input noise data needs to be changed to a certain output size by the generator to produce a particular image. Each output size would require either slightly changing the input 1-dimensional data or change the convolutional layers that transpose the input.
* To improve on our implementation we could try out new configurations of our generator or discriminator models to see how different the output would be to our current results. Or something we could also try is to somehow customize the configuration of the layers depending on what dataset we are using.
* Lastly, because of not being able to complete part 2 of our proposal, we were not able to create a custom dataset that could generate images based on our input text. This in return hindered our ability to fully understand how the StackGAN framework works.

## References

1. "Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014.
2. https://towardsdatascience.com/image-generation-in-10-minutes-with-generative-adversarial-networks-c2afc56bfa3b
3. https://stackoverflow.com/a/45280846
4. https://www.kaggle.com/kvpratama/pokemon-images-dataset
5.  "StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks" , https://arxiv.org/abs/1612.03242
