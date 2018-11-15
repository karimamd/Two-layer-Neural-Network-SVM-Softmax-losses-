Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Fall 2018.

#	Computer Vision Assignment #2 @ FOE Alexandria University

 Zyad Shokry Abozaid        3517
 Kareem Ahmed Abdelsalam    3356

## Contents
 1-Implementing and applying a **k-Nearest Neighbor (kNN)** classifier
 
 2-Implementing and applying a **Multiclass Support Vector Machine (SVM)** classifier
 
 3-Implementing and applying a **Softmax** classifier
 
 4-Implementing and applying a **Two Layer Neural Network** classifier
 
 5-Exploring performance improvements from using **higher-level representations** than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

##  - kNN


##### Assumptions
1- All training and test images data have the exact same dimentions and number of channels (RGB)

2-Neglecting any features from adjacent pixels and pixel combinations

3-Assuming distance between 2 images (which show disimmilarity) is gained from the sum of difference of values of pixel intensities of corresponding pixels

##### Test Cases
The algorithm is trained on 5000 images of Cifar dataset and tested on 500 images from it giving accuracies that do not exceed 30% in all cases using k-folds cross validation with k values = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
and best k=10 with 28.2% accuracy.

##### Key Takeaways
1-Using pixel distances as features to classify images is not practical in most cases

2-Writing vectorized code gives significant speed which is required for more complex tasks
```
Two loop version took 24.860667 seconds
One loop version took 68.162820 seconds
No loop version took 0.403292 seconds

```

3-kNN is memory expensive because it needs to store all training data to classify a single test sample and computationally expensive because it needs to process test sample with all training data before class is captured


##  - SVM Loss

##### Assumptions
 1-Assuming SVM loss function is differentiable so that we can manually take the derivative to compute the analytic gradient
 
 2-Using raw pixel values as features of images
 
 3-Assuming no needed preprocessing of the dataset (Cifar)
 
##### Test Cases
Obtained 35.7% on 1000 samples testing set from Cifar dataset after applying gridsearch to find best learning rate and regularization strength (38.9% on validation set which is in range of required accuracy as about 40% was expected)
##### Key Takeaways
1-Analytic gradient is exact,fast but error-prone due to existence of hardly differentiated functions and non-differentiables while Numerical Gradient is easy to code and compute but it is approximate and slow

2-Analytical gradient is commoly used in practice and numerical gradient is used to validate the analytic gradient to reduce that error-prone disadvantage

3-Learning rate is a critical factor that needs to be carefully tuned in Stochastic Gradient Descent and other optimization functions because if too low then reaching minimum will need very long time and iterations and algorithm may get stuck in local minima
and if too large then will always overshoot minimum and diverge.

4-Regulatrization strength is another important parameter to tune along with a good choice of regularization method according to problem
it allows decreasing complexity of the model to keep away from overfitting
and also makes features have equal importance in generating the final score.
Also,including the L2 penalty leads to the appealing max margin property in SVMs


##  - Softmax Loss

##### Assumptions

##### Test Cases

##### Key Takeaways


##  - Two-Layer Neural Network

##### Assumptions

##### Test Cases

##### Key Takeaways


##  - Features

##### Assumptions

##### Test Cases



##### Key Takeaways



##  - Methodologies
All written code and Jupyter notebooks for each implementation can be found [here](https://github.com/karimamd/Two-layer-Neural-Network-SVM-Softmax-losses-) with full explanation of steps in the notebook.
