# CS 725 (Autumn 2022): Assignment

This assignment is due by **11.59 PM on October 23, 2022**.

## General Instructions
Please read the following important instructions before getting started on the assignment.
1. This assignment should be completed individually.
2. This assignment is entirely programming-based. A correponding Kaggle task is also hosted [here](https://www.kaggle.com/competitions/cs725-2022-assignment-regression/overview). Please signup on Kaggle using your IITB LDAP email accounts, with Kaggle `Display Name = <your_roll_number>` . Instructions to join the Kaggle competition will be shared on Moodle.
3. Your final submission should be a .tar.gz bundle of a directory organized exactly as described in the [Submission Instructions](#submission-instructions). Submissions that do not strictly adhere to this structure will be penalized.
4. Successful completion of the assignment would include: (A) Submitting <your-roll-number>.tar.gz on Moodle and (B) Having your roll number appear on the Kaggle leaderboard.

## Implement a Feedforward Neural Network using NumPy
This assignment will familiarize you with training and evaluating feedforward neural networks. You will work on a regression task where you will have to predict the release year of a song from a set of timbre-based audio features extracted from the song. This consists of a year range between 1922 to 2011. (More details about this corpus are available [here](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd).) Click [here](https://www.kaggle.com/competitions/cs725-2022-assignment-regression/data) to download the training, development and test sets from Kaggle.

## Dataset Information
- The dataset contains three files `train.csv`, `dev.csv`, and `test.csv`
- Each row in the `__.csv` file contains timbre-based audio features extracted from a song.
- The dataset has 90 features: 12 timbre average values and 78 timbre covariance values. Each column denotes a feature.
- `train.csv` and `dev.csv` contains following columns:
```
1. label - Year of release of the song in the range [1922, 2011]
2. TimbreAvg1
3. TimbreAvg2
.
.
13. TimbreAvg12
14. TimbreCovariance1
15. TimbreCovariance2
.
.
91. TimbreCovariance78
```
- `test.csv` contains same features except `label`

## Part 1
In Part 1, you will implement the neural network, train it using train data and report its performance on dev data.

### Part 1.A (25 Points)
Implement the functions definitions given in [nn.py](nn.py) to create and train a neural network. Run mini-batch gradient descent on Mean Squared Error (MSE) loss function.

For both Part 1.A and Part 1.B, use fixed settings:
```
- Seed for numpy: 42
- Use ReLU activation function for all HIDDEN layers.
```

#### Initialization of Weights and Biases (for both Part 1 and Part 2)
Initialize Weights and Biases using uniform distribution in the range \[-1, 1\].

#### What to submit in Part 1.A?
For Part 1.A, only code needs to be submitted in the file `nn_1.py`.

### Part 1.B (15 Points)
Plot a graph showing the train and dev set loss after each epoch on each of the batch sizes: 32 and 128. Do it for the first 100 epochs. 
If the neural network starts overfitting in the first 100 epochs, report the epoch after which the model overfits. No restrictions on rest of the hyper-parameters.

#### What to submit in Part 1.B?
Image files:
- `train_32.png`: Plot for train set loss for batch size 32
- `dev_32.png`: Plot for dev set loss for batch size 32
- `train_64.png`: Plot for train set loss for batch size 64
- `dev_64.png`: Plot for dev set loss for batch size 64


## Part 2 (10 points)
In Part 2, you will evaluate your network's performance on test data given in `test.csv`.

In this part, there is no restriction on any hyper-parameter values. You are also allowed to explore various hyper-parameter tuning and cross-validation techniques.

You are also free to create any wrapper functions over given functinos in [nn.py](nn.py)

Submit your predictions on test data on [Kaggle competition](https://www.kaggle.com/competitions/cs725-2022-assignment-regression/overview) in a `<roll_number>.csv` file in the following format:
```
Id,Predictions
1,2000
2,1976
3,2002
.
.
5100,1943
```

<!---Report the hyper-parameter configurations you used and score obtained on test data in the leaderboard.--->
<!---Clearly specify the hyper-parameters you used and the score obtained on leaderboard using those hyper-parameter configurations.--->
In a CSV file (`part_2.csv`), write the name of the hyper-parameter and the value you used.

#### What to submit in Part 2?
Create a two-column csv file `part_2.csv` and write the name of hyper-parameter in first column and value in the second column.
Also submit your code for part 2 in `nn_2.py`. This code may contain all the enhancements you did for part 2.

For example:
```
Name,Value
learning_rate,0.001
batch_size,30
dropout,0.10
```


## Extra credit assignment
You will have to work on a classification task where you will predict label among ("Very Old", "Old", "New" and "Recent") based on when the song was released. Click [here](https://www.kaggle.com/competitions/cs-725-autumn-2022-assignment-classification/data) to download the training, development and test sets from Kaggle. The corresponding kaggle task is hosted [here](https://www.kaggle.com/competitions/cs-725-autumn-2022-assignment-classification/overview).


### Dataset Information
- The dataset contains three files `train.csv`, `dev.csv`, and `test.csv`
- Each row in the `__.csv` file contains timbre-based audio features extracted from a song.
- The dataset has 90 features: 12 timbre average values and 78 timbre covariance values. Each column denotes a feature.
- `train.csv` and `dev.csv` contains following columns:
```
1. label - "Very Old", "Old", "New" and "Recent", based on when it was released
2. TimbreAvg1
3. TimbreAvg2
.
.
13. TimbreAvg12
14. TimbreCovariance1
15. TimbreCovariance2
.
.
91. TimbreCovariance78
```
- `test.csv` contains same features except `label`


### Task
You need to modify the `nn.py` used in the regression task to support the classification task and evaluate your network's performance on test data given in `test.csv`.

In this task, there is no restriction on any hyper-parameter values. You are also allowed to explore various hyper-parameter tuning and cross-validation techniques.

You are also free to create any wrapper functions over given functinos in [nn.py](nn.py)

Submit your predictions on test data on [Kaggle competition](https://www.kaggle.com/competitions/cs-725-autumn-2022-assignment-classification/overview) in a `<roll_number>.csv` file in the following format:
```
Id,Predictions
1,Very Old
2,Old
3,New
.
.
5100,Recent
```

In a CSV file (`params.csv`), write the name of the hyper-parameter and the value you used.

#### What to submit?
Create a two-column csv file `params.csv` and write the name of hyper-parameter in first column and value in the second column.
Also submit your code for part 2 in `nn_classification.py`. This code may contain all the enhancements you did for part 2.

For example:
```
Name,Value
learning_rate,0.001
batch_size,30
dropout,0.10
```


#### Tips to improve your rank on leaderboard
You can explore following techniques to get better generalization performance
- [Feature Scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- [Feature Selection](https://en.wikipedia.org/wiki/Feature_scaling)
- [Dropout](https://youtu.be/qfsacbIe9AI?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- [Batch Normalization](https://youtu.be/1XMjfhEFbFA?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- [Early Stopping](https://youtu.be/zm5cqvfKO-o?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)


## Submission Instructions
- Your submission directory should be: 
```
<your_roll_number>
├── nn_1.py
├── nn_2.py
├── part_2.csv
├── plots_1b
│   ├── dev_32.png
│   ├── dev_64.png
│   ├── train_32.png
│   └── train_64.png
├── classification `This submission directory is optional`
│   ├── params.csv
│   └── nn_classification.py
└── readme.txt
```

- Use `Readme.txt` to describe any other information needed to run your code successfully.
- Add these files to directory `<your_roll_number>`.
- Compress the directory `<your_roll_number>` in .tgz format using following command:
 
  ```tar -czf <your_roll_number>.tar.gz <your_roll_number>```
  
- Submit the `<your_roll_number>.tar.gz` file.