Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Fall 2018.


#	Computer Vision Assignment #2 @ FOE Alexandria University
 Zyad Shokry Abozaid        3517
 
 Kareem Ahmed Abdelsalam    3356

## Contents

1.  **k-Nearest Neighbor (kNN)** classifier

2.   **Multiclass Support Vector Machine (SVM)** classifier

3.   **Softmax** classifier

4.   **Two Layer Neural Network** classifier
5.  Exploring performance improvements from using **higher-level representations** than raw pixels 
(e.g. color histograms, Histogram of Gradient (HOG) features)

6.  Methodologies

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

5-Softmax uses hinge loss and determining margin helps choose how sharp we want our classification to be (of course this implies more complexity in the model and thats why the margin and regression strength are inversly proportional)


##  - Softmax Loss

##### Assumptions
Same as SVM assumptions
##### Test Cases
Obtained 35.6% test accuracy and 36.7% Validation accuracy with same conditions as SVM.
##### Key Takeaways
1. Softmax is a probabilistic classifier that output the probability of each class for a point and chooses the point with the highest score and it can be said that SVM is a special case of Softmax.
2. Softmax is highly affected by outliers unlike SVM loss.
3. Softmax uses Cross-entropy loss.

##  - Two-Layer Neural Network

##### Assumptions

##### Test Cases

Trained the network on 49k datapoints and tested on 1k datapoints and 1k validation set.
Initially optained a validation accuracy of 28.7% then after hyperparameter tuning (Learning rate and hidden layer size only) obtained validation accuracy of 48.7% and test accuracy of 49.3%.

##### Key Takeaways
1. Choosing learning rate is the highest priority when training neural networks due to its high acontribution to the accuracy.

2. Neural Networks are highly prone to overfitting so some tricks need to be done to prevent that from happening.

3. visualizing loss history per epoch and classification accuracy are useful tools for debugging.





##  - Higher Level Representations

##### Assumptions
1. Assuming the Histogram of Oriented Gradients and color histogram implementations by cs231n is implemented correctly
2. Assuming that concatenating features of both HOG which gives texture information about the image and color histogram which gives color information alone without texture will be a give a better results than any one of them alone

##### Test Cases
Best Validation accuracy of SVM is 47.5% and best testing accuracy is 46.5% which is about 10% higher accuracy than numbers mentioned earlier in SVM section using raw pixels as features.

As for neural network : the obtained accuracy was 20% which is wrong but it was done after about 2000 different combinations of hyperparameters so we missed getting this part to the wanted accuracy.


##### Key Takeaways
1. Using HOG and color histograms will give better features than that of raw pixels



##  - Methodologies
All written code and Jupyter notebooks for each implementation can be found [here](https://github.com/karimamd/Two-layer-Neural-Network-SVM-Softmax-losses-) with full explanation of steps in the notebook.
