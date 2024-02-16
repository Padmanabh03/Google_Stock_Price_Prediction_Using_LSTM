# Google_Stock_Price_Prediction_Using_LSTM

## Introduction
In this project, my aim is to predict Google's stock price using a type of Recurrent Neural Network (RNN) known as Long Short-Term Memory (LSTM). LSTM networks are particularly suited for time series prediction due to their ability to capture long-term dependencies in sequential data. Unlike standard feedforward neural networks, LSTMs have a memory cell that can maintain information in memory for long periods of time, which is crucial for understanding the temporal context of stock prices.

## Why LSTM?
The stock market is known for its volatility and complex patterns. Traditional time series models often fall short in capturing the intricacies involved. LSTMs, on the other hand, are designed to recognize patterns over time intervals of varying lengths. This is achieved through their unique architecture that allows for both long-term memory and short-term data through gates that regulate the flow of information.

## LSTM Model
![image](https://github.com/Padmanabh03/Google_Stock_Price_Prediction_Using_LSTM/assets/71133619/c14a1e5c-490b-4053-8e21-69a06910392c)

The LSTM's architecture comprises several components:

- **Input gate**: Decides what new information to include in the cell state.
- **Forget gate**: Determines what information to discard from the cell state.
- **Output gate**: Selects the information from the cell state to output at the current timestep.

This design allows LSTMs to mitigate the vanishing gradient problem common in traditional RNNs, making them highly effective for our task of predicting stock prices, where the understanding of long-term trends is crucial.

## Dataset Overview

The dataset used in this project is split into two CSV files: `Google-Train.csv` for training and `Google-Test.csv` for testing our LSTM model.

- **Training Set**: Contains historical daily stock prices of Google. Each entry includes attributes like the opening price, the highest price of the day, the lowest price, and the closing price.

- **Testing Set**: Comprises more recent stock price data, structured similarly to the training set. This set is used to evaluate the model's performance by comparing its predictions to actual prices.

The data is clean and well-organized, making it ideal for feeding into our LSTM model without requiring extensive preprocessing. We are only going to use the opening price for this prediction.


## Code Summary

The LSTM model for stock price prediction goes through the following steps:

- **Normalization**: Scaled the stock prices to a range of 0 to 1 for better neural network performance.
- **Time Steps**: Created sequences of 40 previous stock prices(you could change it and try differnt timesteps) as input to predict the next stock price.
- **LSTM Layers**: Built a neural network with several LSTM layers to learn from the sequences.
- **Dropout**: Included dropout layers to prevent overfitting.
- **Compilation**: Compiled the model with an optimizer and loss function suitable for regression.
- **Training**: Fitted the model to the training data over 100 epochs with a batch size of 32.
- **Prediction**: Used the model to predict stock prices on new data.
- **Plotting**: Visualized the real vs. predicted stock prices on a graph.

This process aims to leverage LSTM's ability to remember long-term patterns for accurate stock price forecasts.

## Model Performance
![image](https://github.com/Padmanabh03/Google_Stock_Price_Prediction_Using_LSTM/assets/71133619/0ac10aef-5f85-478b-a947-96ffdc516bed)

The graph illustrates the performance of the LSTM model in predicting Google's stock price. The X-axis represents the time period over which the predictions were made, and the Y-axis shows the stock price in USD.

- **Red Line**: The actual stock prices of Google as observed in the market.
- **Green Line**: The stock prices as predicted by our LSTM model.

As seen in the graph, the predicted values follow the trend of the actual stock prices closely, with some discrepancies. This visualization serves as a way to assess how well the model has learned to forecast stock prices over the given time period.





