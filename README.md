# Indian Sign Language Recognition

Sign Languages are a set of languages that use predefined actions and movements to convey a message. These languages are primarily developed to aid deaf and other verbally challenged people. They use a simultaneous and precise combination of movement of hands, orientation of hands, hand shapes etc. Different regions have different sign languages like American Sign Language, Indian Sign Language etc. We focus on Indian Sign language in this project.

Indian Sign Language (ISL) is a sign language that is predominantly used in South Asian countries. It is sometimes referred to as Indo-Pakistani Sign Language (IPSL). There are many special features present in ISL that distinguish it from other Sign Languages. Features like Number Signs, Family Relationship, use of space etc. are crucial features of ISL. Also, ISL does not have any temporal inflection.

In this project, we aim towards analyzing and recognizing various alphabets from a database of sign images. Database consists of various images with each image clicked in different light condition with different hand orientation. With such a divergent data set, we are able to train our system to good levels and thus obtain good results.

We investigate different machine learning techniques like:
- [K-Nearest-Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## Getting Started
### Prerequisites
Before running this project, make sure you have following dependencies - 
* [Dataset](https://drive.google.com/folderview?id=0Bw239KLrN7zofmxvSmtsVHlrbkFRY1NwMjh2NFJGX1ZtY0lKOTR0REJnQnBUdVgyVDlMMkk&usp=sharing) (Download the images from this link)
* [Python 3.6](https://www.python.org/downloads/)
* [pip](https://pypi.python.org/pypi/pip)
* [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)

Now, using ```pip install``` command, include following dependencies 
+ Numpy 
+ Pandas
+ Sklearn
+ Scipy
+ Opencv
+ Tensorflow

### Running
To run the project, perform following steps -

 1. Put all the training and testing images in a directory and update their paths in the config file *`common/config.py`*.
 2. Generate image-vs-label mapping for all the training images - `generate_images_labels.py train`.
 3. Apply the image-transformation algorithms to the training images - `transform_images.py`.
 4. Train the model(KNN & SVM) - `train_model.py <model-name>`. Note that the repo already includes pre-trained models for some algorithms serialized at *`data/generated/output/<model-name>/model-serialized-<model-name>.pkl`*.
 5. Generate image-vs-label mapping for all the test images - `generate_images_labels.py test`.
 6. Test the model - `predict_from_file.py <model-name>`.
 7. To obtain Better Results, train the model using Convolutional Neural Network which can be done by running the cnn.py file after activating Tesorflow.
 
 #### Accuracy without CNN
The best accuracy was achieved by CNN using Momentum Optimizer and a learning rate of 0.5.The Plot for the same is depicted bleow.

<p align="center">
  <br>
  <img align="center" src="https://github.com/sanghaisubham/Indian-Sign-Language/blob/master/momentum.png">
        <br>  
  </p>
  
  **To-Do:**
 - Improve the accuracy if possible by collecting more data and applying various CNN architectures like VGG16,Le-Net5 etc.
  

