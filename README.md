# LMU_ML_26_CA

Spam Email Detection using Naive Bayes
This project is a machine learning assignment (CA02) that implements a Spam Email Classifier using the Gaussian Naive Bayes algorithm. The model is trained on a dataset of emails to distinguish between "Spam" (junk mail) and "Non-Spam" (ham).

Project Overview
The objective of this project is to build a supervised machine learning model that takes raw email text files as input and predicts whether they are spam. The project involves:

Data Cleaning: Processing raw text to remove stop words and non-alphabetic characters.

Dictionary Creation: Identifying the 3,000 most common words in the training set.

Feature Extraction: Converting emails into a numerical feature matrix based on word frequency.

Model Training: Using Scikit-Learn's GaussianNB to train the model.

Evaluation: Testing the model against unseen data and calculating accuracy.

Technologies Used
Python 3(Google Colab)

Libraries:

scikit-learn (for the Naive Bayes model and metrics)

numpy (for matrix operations)

os (for file handling)

collections (for counting word frequencies)

Folder Structure & Data

⚠️ IMPORTANT: To run this code successfully, the dataset must be organized in specific folders relative to the .ipynb file. The code uses relative paths (./train-mails and ./test-mails) to ensure portability.

The directory structure should look like this:

Plaintext
├── CA02_NB_assignment.ipynb   # The main Jupyter Notebook
├── README.md                  # This documentation file
├── train-mails/               # Folder containing 702 training emails
│   ├── 3-1msg1.txt
│   ├── spmsgc10.txt
│   └── ...
└── test-mails/                # Folder containing 260 testing emails
    ├── 8-809msg1.txt
    ├── spmsgc122.txt
    └── ...
train-mails: Contains 702 emails (mixed spam and non-spam) used to train the model.

test-mails: Contains 260 emails used to verify the model's accuracy.

How to Run
Clone the Repository:

Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
Prepare the Data: Ensure train-mails and test-mails folders are in the same directory as the notebook. If you are using Google Colab, you may need to upload the Data.zip file and unzip it using the provided cell:

Python
!unzip Data.zip
Execute the Notebook: Open CA02_NB_assignment.ipynb in Jupyter Notebook or Google Colab and run all cells sequentially.

Code Logic & Methodology
1. Dictionary Creation (make_Dictionary)

The program reads all emails in the training set to build a vocabulary.

It ignores non-alphabetic characters.

It removes single-letter words.

It selects the 3,000 most frequent words to act as the features for the model.

2. Feature Extraction (extract_features)

This function converts the raw text files into a numerical matrix that the machine learning algorithm can understand.

Rows: Represent individual emails.

Columns: Represent the 3,000 words from the dictionary.

Values: The frequency (count) of each word in that specific email.

Labels: It looks at the filename; if it starts with spmsg, it labels the email as Spam (1), otherwise, it is Non-Spam (0).

3. Training and Prediction

The model is initialized using GaussianNB().

It is trained (fit) using the matrix generated from the train-mails directory.

It predicts labels for the test-mails directory.

Finally, accuracy_score is used to compare the predicted labels against the actual labels.

Performance
The model achieves an accuracy of approximately 96.15% on the test dataset.

Python
Completed classification of the Test Data .... now printing Accuracy Score:
0.9615384615384616
👤 Author
Name: Christopher, Jasmine

This assignment is part of the BSAN 6070 (or relevant course code) curriculum.
