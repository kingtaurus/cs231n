Stanford CS231n Convolutional Neural Networks for Visual Recognition Assignments
================================================================================

##Assignment 1
From [Assignment 1 Webpage](http://cs231n.github.io/assignments2016/assignment1/):

In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier.
The goals of this assignment are as follows:
* understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
* understand the train/val/test splits and the use of validation data for hyperparameter tuning.
* develop proficiency in writing efficient vectorized code with numpy
* implement and apply a k-Nearest Neighbor (kNN) classifier
* implement and apply a Multiclass Support Vector Machine (SVM) classifier
* implement and apply a Softmax classifier
* implement and apply a Two layer neural network classifier
* understand the differences and tradeoffs between these classifiers
* get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

##Assignment 2 
From [Assignment 2 Webpage](http://cs231n.github.io/assignments2016/assignment2/):

In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

* understand Neural Networks and how they are arranged in layered architectures
* understand and be able to implement (vectorized) backpropagation
* implement the core parameter update loop of mini-batch gradient descent
* effectively cross-validate and find the best hyperparameters for Neural Network architecture
* understand the architecture of Convolutional Neural Networks and train gain experience with training these models on data

##Assignment 3 
From [Assignment 1 Webpage](http://cs231n.github.io/assignments2016/assignment3/):

In the previous assignment, you implemented and trained your own ConvNets. In this assignment, we will explore many of the ideas we have discussed in lectures. Specifically, you will:

* Reduce overfitting using dropout and data augmentation
* Combine models into ensembles to boost performance
* Use transfer learning to adapt a pretrained model to a new dataset
* Use data gradients to visualize saliency maps and create fooling images



###Setup
You can work on the assignment in one of two ways: locally on your own machine, or on a virtual machine through Terminal.com. I will only detail how to run in locally.

####Working locally
Get the initial code as a zip file [here](http://vision.stanford.edu/teaching/cs231n/winter1516_assignment1.zip).

**[Use Anaconda]** The preferred approach for installing all the assignment dependencies is to use [Anaconda](https://www.continuum.io/downloads), which is a Python distribution that includes many of the most popular Python packages for science, math, engineering and data analysis. Once you install it you can skip all mentions of requirements and you're ready to go directly to working on the assignment.

**[Virtual Environment (through pip)]**
If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run the following (assuming you already have pip installed):

```
cd assignment1
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
```

Alternatively you can use `virtualenvwrapper` to manage your virtual environments. **Note** there are few other points:
1. python packages can be installed through package managers. Sometimes one should to use the switch `--system-site-packages` during virtualenv creation.
2. do not use `sudo` when calling `pip`.
3. `conda env` can be used to install/upgrade packages in a similar to virtual environments ([documentation](http://conda.pydata.org/docs/using/envs.html)). **USE IT**

####Download data:
Once you have the starter code, you will need to download the CIFAR-10 dataset. Run the following from the assignment1 directory:
```
cd cs231n/datasets
./get_datasets.sh
```

####Start IPython:
After you have the **CIFAR-10** data, you should start the IPython notebook server from the `assignment1` directory. If you are unfamiliar with IPython, you should read our [IPython tutorial](http://cs231n.github.io/ipython-tutorial).

**NOTE:** 
If you are working in a virtual environment on OSX, you may encounter errors with matplotlib due to the [issues described here](http://matplotlib.org/faq/virtualenv_faq.html). You can work around this issue by starting the IPython server using the `start_ipython_osx.sh` script from the `assignment1` directory; the script assumes that your virtual environment is named `.env`.
