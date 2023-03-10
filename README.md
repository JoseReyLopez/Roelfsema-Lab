# Roelfsema-Lab

# Under Construction

---------------------------------


## Preoject aim:

In this project our aim is to write a mathematical models of the visual cortex previously proposed and then train them using as inputs of an ImageNet pretrained CNN (VGG-19 and InceptionV1), as outputs they will have the MUA activity from a region of hte visual cortex (V1, V4, IT) measured from a monkey  with several blackrock Utah arrays. (Images and activations were already collected, this part does not require any animal interaction).


## Part 1: Writing the models in pytorch

My supervisor wrote the model proposed by Bashivan (to be cited) and I wrote the model proposed by Cadena [Deep convolutional models improve predictions of macaque V1 responses to natural images](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006897)

After wiriting it we need to find a way to train it as the model by itself fails to converge and needs regularization. The regularization proposed in the paper seems to be working fine but the parameters tuning it need to be found for this particular case.

The time to convergence is also very high and a training strategy had to be found, I opter for starting with a high learning rate what will initially decrease L1 from the model, and decrese the lr after 70 batches, the best lr for tuning fine details is around lr = 1e-6 and lrs lower than that are too small to perform substancial changes. This technique seems to be working well and can fit the model in around 2 epochs in the best case.

Once the model has been written and trained, we need to generalize it to all combinations between the implant sites and the layers in the pretrained CNN.


## Part 2: Adapting all possible combinations

After havinb both model (Bashivan and Cadena) I adapted to both networks (VGG-19 and Inceptionv1) and trained all possible combinations. I made a heat map showing the lowest evaluation loss for each of the 4 combinations.

Using a single Nvidia RTX 3060 12GB this step can take slightly over a week, being VGG-19/Cadena the most expensive to compute, taking up to one hour per combination of layer and implant position. And the other three taking around 3 days combined.

For what we can see, implants in deeper parts of the visual cortex have the best fit in deeper layers of the pretrained network, this is very clear for VGG-19, and the trend continues with InceptionV1 but the trend is not as strong.

<b>Image are too big to be shown here, there will be a link for them all at the end of the document.</b>


## Part 3: Image generation

Once all combinations have been trained, it is time to generate the MEIs (Most Exciting Images). For that purpose we use [lucent](https://github.com/JoseReyLopez/lucent) a pytorch adaptatin from the lucid library. When provided with a CNN it can be used to select a layer and a neuron, neurons or the whole layer and create the image that excites it the most.

Further details about the internal of lucent/lucid can be found on the [original posts from the lucid authors](https://github.com/tensorflow/lucid#notebooks).

For each combination layer/implant an image was generated, this part has reproducibility issues because lucent allows for us to set a seed but it doesn't generate the same image every time.

There are too many things to comment on the results for make an extensive discussion here. But it is possible to provide a few bullet points about them.
* Receptive Field of the measured neurons seems to be present on the reconstruction
* The RF slowly fades away as we move into deeper areas, as expected (deeper areas are most excited by whole objects rather than simple features)
* From shallow areas we are able to recover Gabor-like features and simple but intrincate textured
* From deeper areas (IT) we can see how monkey-like figures and faces start to arise, which is an important sanity check, as IT tends to be most excited about faces.

The fittings and the most exciting iamges can be checked here: 


## Part 4: TO DO...

This project is expected to be finished be July '23, some of the thins that we expect to implement are:

* Using pretrained networks on ecoset dataset, to avoid the overwhelming presence of dog-like features presence on the reconstruction due to ImageNet bias about including so many dogs on its training dataset.
* Using [other ways](https://arxiv.org/abs/1605.09304) to find the most exciting images 
* Redoing the project with the data from a trained monkey and then presenting the images to it in order to measure its neural activity to see the goodness of the fitting.

