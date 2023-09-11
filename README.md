# deep_learning_challenge

# CharityML Deep Learning Model

This README provides documentation for the CharityML Deep Learning Model code. The code presented here aims to create and train a deep learning model for predicting the success of charity applications based on various input features. Below, you'll find a step-by-step explanation of the code's functionality.

## Table of Contents
- [Dependencies](#dependencies)
- [Data Source](#data-source)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Saving](#model-saving)

---

### Dependencies <a name="dependencies"></a>

The code relies on the following Python libraries:
- `sklearn` for data preprocessing and splitting
- `pandas` for data manipulation
- `tensorflow` for creating and training the deep learning model

Please ensure that you have these libraries installed in your Python environment before running the code.

### Data Source <a name="data-source"></a>

The data used for this project is loaded from the following URL: https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv

his dataset contains information related to various charity applications.

### Data Preprocessing <a name="data-preprocessing"></a>

1. **Drop Columns**: The code removes non-beneficial ID columns, namely 'EIN' and 'NAME', as they do not contribute to model training.

2. **Binning Application Types**: Less common 'APPLICATION_TYPE' values are grouped into an 'Other' category for better model performance.

3. **Binning Classifications**: Similarly, less common 'CLASSIFICATION' values are grouped into an 'Other' category.

4. **One-Hot Encoding**: Categorical data is converted into a numeric format using one-hot encoding.

5. **Data Splitting**: The dataset is divided into training and testing subsets using the `train_test_split` function.

6. **Standard Scaling**: Standard scaling is applied to normalize the feature values.

### Model Architecture <a name="model-architecture"></a>

The deep learning model is constructed with the following architecture:
- Input Layer: The number of input features depends on the dataset.
- Hidden Layer 1: Consisting of 80 units with ReLU activation function.
- Hidden Layer 2: Consisting of 30 units with ReLU activation function.
- Output Layer: Comprising 1 unit with sigmoid activation function for binary classification.

### Model Training <a name="model-training"></a>

The model is compiled using binary cross-entropy loss and the Adam optimizer. It is then trained on the training data for 100 epochs.

### Model Evaluation <a name="model-evaluation"></a>

Following training, the model's performance is assessed using the testing data. The evaluation results include the model's loss and accuracy.

### Model Saving <a name="model-saving"></a>

The trained model is saved to an HDF5 file named "AlphabetSoupCharity.h5" for future use or deployment.

---

This README provides a comprehensive overview of the code's functionality. You can execute the provided code within a Python environment to train and evaluate the deep learning model, which predicts the success of charity applications.