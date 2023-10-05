# Module 21 Report Template

## Files and Folders
* AlphabetSoupCharity_Optimizer_HD5 Folder
  * Contains all the HD5 files if you want to import them 
* Image Loss and Accuracy:
  * Contains all the graphs for each model and only used to write this report
* Remaining Notebook Files
  * Each Notebook represents each Machinne Model. The 4th notebook had the best validation accuracy 

  
## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include: 

* Explain the purpose of the analysis.<br/>
*The purpose of the analysis is to evaluate how accuarte the neural network is at predicting whether an applicant is successful if funded by the company.*
* Data Processing.<br/>
  * What variable(s) are the target(s) for your model?<br/>
  *The target variable is IS_SUCCSSESFUL*<br/>
  * What variable(s) are the features for your model?<br/>
  *The variables that are the features were:*
    * APPLICATION_TYPE
    * AFFILIATION
    * CLASSIFICATION
    * USE_CASE
    * ORGANIZATION STATUS
    * STATUS
    * INCOME_AMT
    * SPECIAL_CONSIDERATIONS	
    * ASK_AMT
  * What variable(s) should be removed from the input data because they are neither targets nor features?<br/>
  *The variables that were removed from the input data were EIN and NAME as they have no relevance to the model in predicting the success of applicants*


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1-1 :
  * The structure of the Neural Network is as follow: 
    * **Input Layer:** 43 neurons 
    * **Hidden Layer #1:** 86 neurons, relu
    * **Hidden Layer #2:** 30 neurons, relu
    * **Output Layer:** 1 neuron, sigmoid 
    * **Epoch:** 100
 
  * Machine Learning Model 1-2:
    * **Input Layer:** 43 neurons 
    * **Hidden Layer #1:** 10 neurons, relu
    * **Hidden Layer #2:** 15 neurons, relu
    * **Output Layer:** 1 neuron, sigmoid 
    * **Epoch:** 100
   ![Machine Learning Model 1](https://github.com/Dav5T/deep-learning-challenge/assets/130593953/cfe8005e-e1f2-4eba-9e59-5cfe83055837)
   **Figure 1:** Epoch accuracy and Loss for Machine Learning Model 1-1, 1-2

  * The results of Model 1-1 loss was 56.16% and the accuracy is 72.80% for the test data.Training accuracy mainly hung around 74% towards the end. This indicates that overfitting is not taking place. In addition to running the first model, I also ran a 2nd model in the same notebook to see if fewer neurons would have a significant effect on accuracy and it turns out it doesn't. There was slight drop resulting in a 72.52% validation accuracy. However, it did take a little longer for the training accuracy to acheive roughly a 73.6% accuracy. I chose Relu as the activation function since it is supposed to be a faster learner than tanh. 

  * Machine Learning Model 2-1:
    * **Input Layer:** 43 neurons 
    * **Hidden Layer #1:** 129 neurons, relu
    * **Hidden Layer #2:** 86 neurons, relu
    * **Hidden Layer #3:** 30 neurons, relu
    * **Output Layer:** 1 neuron, sigmoid 
    * **Epoch:** 150

  * Machine Learning Model 2-2:
    * **Input Layer:** 43 neurons 
    * **Hidden Layer #1:** 129 neurons, relu
    * **Hidden Layer #2:** 86 neurons, relu
    * **Hidden Layer #3:** 30 neurons, relu
    * **Hidden Layer #4:** 10 neurons, relu
    * **Output Layer:** 1 neuron, sigmoid 
    * **Epoch:** 100
  ![Machine Learning Model 2](https://github.com/Dav5T/deep-learning-challenge/assets/130593953/918beabd-f0ae-4e6c-9953-08ece9d1a1a5)
  **Figure 2:** Epoch accuracy and Loss for Machine Learning Model 2-1, 2-2

  * The results of Model 2-1 validation loss was 60.04% and the validation accuracy is 72.86% for the test data.Training accuracy mainly hung around 74.1% towards the end. For Model 2-2, validation accuracy was 72.82% and a loss of 57.37%. In this case, the training accuracy also lingered around 74%. Overall, increasing the number of hidden layer did not add much value to the accuracy. Increasing the number of neurons and number of hidden layers to 3 did help increase validation accuracy by 0.6%. After looking at the graph, it seems like 100 epoch is sufficient as we can start to seem some huge spikes in loss as we approach 120 epoch. 


  * Machine Learning Model 3-1:
    * **Input Layer:** 43 neurons 
    * **Hidden Layer #1:** 86 neurons, relu, 
    * **Hidden Layer #2:** 30 neurons, relu, GaussianNoise 0.1
    * **Hidden Layer #3:** 30 neurons, relu, GaussianNoise 0.1
    * **Output Layer:** 1 neuron, sigmoid 
    * **Epoch:** 100

  * Machine Learning Model 3-2:
    * **Input Layer:** 43 neurons 
    * **Hidden Layer #1:** 129 neurons, relu
    * **Hidden Layer #2:** 110 neurons, relu
    * **Hidden Layer #3:** 100 neurons, relu
    * **Output Layer:** 1 neuron, sigmoid 
    * **Epoch:** 100
      ![Machine Learning Model 3](https://github.com/Dav5T/deep-learning-challenge/assets/130593953/63aaa733-8b31-4316-bf8b-0a2d3e74eece)
       **Figure 3:** Epoch accuracy and Loss for Machine Learning Model 3-1, 3-2
      
  * The results for MOdel 3-1 has the highest validation accuracy so far of 72.93% and the lowest loss of 55.8%. I decided to add some noise to the 2nd and 3rd hidden layer because I noticed thatt as the validation accuracy increased, so did the validation loss. Based on the graph and comparing the accuracy of the test and trainig data, there isn't a strong indication of overfitting. However, adding noise to the data almost guarentees that the model will not overfit. Model 3-2, validation accuary decreased by a tiny bit to 72.84%. Therefore, adding more neurons doesn't make a significant difference. 


  * Machine Learning Model 4-1:
    * **Input Layer:** 43 neurons 
    * **Hidden Layer #1:** 80 neurons, relu, 
    * **Hidden Layer #2:** 30 neurons, relu, GaussianNoise 0.2
    * **Hidden Layer #3:** 20 neurons, relu, GaussianNoise 0.2
    * **Output Layer:** 1 neuron, sigmoid 
    * **Epoch:** 100

  * Machine Learning Model 4-2:
    * **Input Layer:** 43 neurons 
    * **Hidden Layer #1:** 86 neurons, relu, 
    * **Hidden Layer #2:** 30 neurons, tanh, GaussianNoise 0.1
    * **Hidden Layer #3:** 30 neurons, tahn, GaussianNoise 0.1
    * **Output Layer:** 1 neuron, sigmoid 
    * **Epoch:** 100
      ![Machine Learning Model 4](https://github.com/Dav5T/deep-learning-challenge/assets/130593953/b7ec9910-7eb6-4da2-aed4-04dac6a5c037)
      **Figure 4:** Epoch accuracy and Loss for Machine Learning Model 4-1, 4-2

  * The results for Model 4-1 validation accuracy was 72.98%, however, with a higher loss of 56.1%. This is so far the best accuracy that I have been able to achieve. For Model 4-2, accuracy was worse than Model 3-1. It had a validation accuracy of 72.75%. I wanted to see if tahn would perform better than using relu since Model 3-1 had given the best result before running all of the models above. 


## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:<br/>
* Overall the best model was Machine Learning Model 4-1. It has the best validation accuracy of 72.98%. I chose relu over tahnh as the activation function for majority of the models because relu tends to learn faster than tanh. I did use tanh for Model 4-2 to compare how it Model 3-1, it resulted in a worse validation accuracy. It probably could have performed better with more epoch. Following some of the reccommendations that we learned, adding more hidden layers, adding more neurons, adding more epochs, and trying different activation function, the best strategy I found was 3 hidden layers with less neurons, and adding GaussianNoise gave the best result. With the help of the graphs from TensorBoard, I was able to see how the model was performing based on the graphs. Reducing the epochs did not make sense and going over 100 epochs caused random spikes in the training loss. 

* Recomeendation:<br/>
*Since we are limited by the resources of the computer, such as RAM and processing speed, it makes it more difficult to try out more combinations. An approach we could have taken was creating a Hyperparameter Tuner to help determine the best combination to create a neural network in order to acheive the most the highest validation accuracy. Another option is going back to the data and trying to re-tweak the categorical encoding. In addition, I would consider the possibility of eliminating a feature, but that would be the last resort at this point as it seems like having less data would not increase the accuracy of the models. We could also chage the split of the training and testing data or keep training the data and retrain it using less epoch. 

## Resources
1. How is it possible that validation loss is increasing while validation accuracy is increasing as well: https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy <br/>
2. How to Improve Deep Learning Model Robustness by Adding Noise: https://machinelearningmastery.com/how-to-improve-deep-learning-model-robustness-by-adding-noise/ <br/>
3. Pandas: Get values from column that appear more than X times: https://stackoverflow.com/questions/22320356/pandas-get-values-from-column-that-appear-more-than-x-times
