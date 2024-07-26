# Using Machine Learning and LSTM for Gold Price Prediction

## Abstract
Gold has long been regarded as a store of value and a hedge against risks in the context of economic instability. Unlike currency, the value of gold is not directly affected by changes in monetary policies or inflation. Therefore, it is often considered a "safe haven" during times of economic and political uncertainty.  

In this project, we focus on applying machine learning models to analyze and predict gold price fluctuations (the end-of-day closing prices). The main objectives of the project include:

*  Data processing
*  Data analysis
*  Applying machine learning models
*  Building and applying an LSTM network model
*  Evaluating performance and improving the model

## Prerequisites
- Python 3.10 or higher
- Required Python packages (listed in `requirements.txt`)

## Config of models
### Random Forest
| **STT** | **Parameters**|  **Setting**  |
| :-- | :---    | :-------|
| 1   | n_estimators  | 100 |
| 2   | min_samples_split | 2 |
| 3   | min_samples_leaf | 1 |
| 4   | max_features | auto |
### SVM
| **STT** | **Parameters**|  **Setting**  |
| :-- | :---    | :-------|
| 1   | kernel  | 'rbf' |
| 2   | C | 1.0 |
| 3   | epsilon | 0.1 |
| 4   | gamma | 'scale' |
### LSTM Custom  
**Propose architecture**  
This repository contains an implementation of a Long Short-Term Memory (LSTM) network designed for sequence prediction tasks. The model architecture is depicted in the diagram below and includes the following components:
![Architecture](https://github.com/namhai03/LSTM_GoldPrice_Prediction/blob/main/images/architecture.png)   
- Input Layer: The initial layer that receives the input data.
- LSTM Layer 1: The first LSTM layer processes the input sequence, capturing temporal dependencies and patterns.
- Dropout Layer: A dropout layer with a rate of 0.2 is applied to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
- LSTM Layer 2: The second LSTM layer further processes the sequence, enhancing the network's ability to model complex temporal relationships.
- Dropout Layer: Another dropout layer with a rate of 0.2 is applied for regularization.
- Dense Layer: The final dense layer converts the LSTM outputs into the desired output format.
***
**During testing I have set the parameters**
| **STT** | **Parameters**|  **Setting**  |
| :-- | :---    | :-------|
| 1   | Units 1  | [8, 16, 32, 64] |
| 2   | Units 2  | [8, 16, 32, 64] |
| 3   | Batch size | [16, 32, 64, 128] |
| 4   | Optimizer  | Adam, SGD, Nadam |

## Results trainning
![Val_loss](https://github.com/namhai03/LSTM_GoldPrice_Prediction/blob/main/images/val_loss_over_epoch_for_different_optimizers.png) 
![Val_loss](https://github.com/namhai03/LSTM_GoldPrice_Prediction/blob/main/images/val_loss_over_epoch_for_different_batchsize.png) 

## Perfomance
![Prediction](https://github.com/namhai03/LSTM_GoldPrice_Prediction/blob/main/images/predict_price.png)   
The following graph shows the performance of our LSTM model compared to other models like Random Forest and SVM in predicting the given data series. The blue line represents the true data, while the other lines represent predictions from different models.
