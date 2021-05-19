# Project 2: Classification benchmarks
This project was developed as a solution to assignment 4 set by our teacher, Ross Deans Kristensens-McLachlan, during the course. A Github repository which contains all of the code in relation to my solution to the assignment can be found here: 
https://github.com/frillecode/VisualAnalytics2021/tree/main/src/project2

## Project description 
### Classifier benchmarks using Logistic Regression and a Neural Network 
You'll use your new knowledge and skills to create two command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models.  

You should create two Python scripts. One takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. The other should take the full MNIST dataset, train a neural network classifier, and print the evaluation metrics to the terminal.


__Bonus challenges__  
- Have the scripts save the classifier reports in a folder called out, as well as printing them to screen. Add the user should be able to define the file name as a command line argument (easier)
- Allow users to define the number and size of the hidden layers using command line arguments (intermediate)
- Allow the user to define Logistic Regression parameters using command line arguments (intermediate)
- Add an additional step where you import some unseen image, process it, and use the trained model to predict it's value - like we did in session 6 (intermediate)
- Add a new method to the Neural Network class which will allow you to save your trained model for future use (advanced)


## Methods
For this assignment, I created two scripts which both perform a classification task on the MNIST data set in order to classify the handwritten numbers. The first script trains a multinomial Logistic Regression classifier on the data set using _scikit-learn_. The second script trains a multilayered feedforward neural network classifier on the data set using _scikit-learn_ and a NeuralNetwork() class we used in the course. The scripts will print the classifier reports to the terminal as well as save them in the 'out'-folder under a filename which can be specified through the command-line. We can then use the evaluation metrics to compare the performance of the two models.   

The trained neural network is saved into a .npy-file for future use. 

The results I interpret here is from running the scripts with the default values from the parsed arguments (i.e. without parsing new arguments in the command-line) as these were the values that I found to yield the best results. As can be seen in the scripts, I used a 0.8/0.2 train/test split for both models. For the neural network, I used one hidden layer consisting of 32 neurons, and the model was trained with 1000 epochs.  

## Usage
The structure of the files belonging to this assignment is as follows:  
  - Data: inbuilt 'mnist'-dataset from ```sklearn```  
  - Code: _lr-mnist.py_  , _nn-mnist.py_  
  - Results: _out/_

### Cloning repo and installing dependencies 
To run the script, I recommend cloning this repository and installing relevant dependencies in a virtual ennvironment:

```bash
$ git clone https://github.com/frillecode/VisualAnalytics2021
$ cd VisualAnalytics2021
$ bash ./create_venv.sh #use create_venv_win.sh for windows
```

If you run into issues with some libraries/modules not being installed correctly when creating the virtual environment, install these manually by running the following:  
```bash
$ cd VisualAnalytics2021
$ source cds-vis/bin/activate
$ pip install {module_name}
$ deactivate
```

### Running scripts
After updating the repo (see above), you can run the .py-files from the command-line by writing the following:
``` bash
$ cd VisualAnalytics2021
$ source cds-vis/bin/activate
$ cd src/project2
$ python3 lr-mnist.py
$ python3 nn-mnist.py
```

The scripts take different optional arguments that can be specified in the command-line. For the neural network script, 'nn-mnist.py', you can specify:
- Filename for saving the classifier reports
- Number and size of hidden layers 
- Number of epochs
- Proportional size of test set  

For the Logistic Regression script, 'lr-mnist.py', you can specify:
- Filename for saving the classifier reports
- Value for random state of model
- Proportional size of test set  

For example, to run the 'lr-mnist.py'-script with a random state of 1, a test-size of 0.3, and save the classifier report under the filename "classifier_report", you can run: 
```bash
$ python3 lr-mnist.py -rs 1 -ts 0.3 -fn "classifier_report"
```

You can get more information on the arguments that can be parsed by running:
``` bash
$ python3 nn-mnist.py --help
$ python3 lr-mnist.py --help
```


## Discussion of results  
The resulting output-files from running the scripts can be found in 'out/'.  
Both models performed quite well on prediciting the outcome classes (0-9) of the MNIST data set. The Logistic Regression classifier had a weighted average accuracy of 95%, whereas the neural network yielded a weighted average accuracy of 97%. For both of the models, the f1-scores between the different classes varied a bit ranging between 92%-100% for the Logistic Regression and 94%-100% for the neural network model. Interestingly, it was not the same outcome classes that the two models were best at predicting. However, overall the results of the two models can be interpreted as performing quite successfully. 