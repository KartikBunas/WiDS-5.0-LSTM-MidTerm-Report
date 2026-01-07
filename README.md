# WiDS-5.0-LSTM-MidTerm-Report
1. Foundations of Python
My journey began with mastering Python, which serves as the backbone of this project. Unlike other languages, Python's syntax allowed me to focus on logic rather than fighting with complex boilerplate code.


Data Types and Structures: I practiced handling dynamic typing, working with integers, floats, and strings. I spent a lot of time on data structures like lists for storing sequences, dictionaries for key-value mapping, and tuples for immutable data.
+4


Logic and Control: I implemented loops to handle repetitive data tasks and conditional statements to manage decision-making within the code.
+1


Advanced Concepts: Understanding functions and classes was essential, especially since PyTorch requires an object-oriented approach to build neural network layers.

2. The Data Science Library Stack
To process and visualize financial data, I utilized the standard Python ecosystem:

NumPy: This was my primary tool for numerical operations. I used it for creating arrays and performing multi-dimensional slicing to isolate specific price windows.
+1


Pandas: I used Pandas for all data cleaning tasks, including reading CSV and JSON files, and using the locate function to filter through specific dates in the stock history.
+1


Matplotlib: I focused on plotting price trends, creating subplots for multiple indicators, and using histograms and pie charts to understand data distribution.
+2


Sklearn: I used this library to implement baseline linear regression models to compare against my later deep learning results.

3. Introduction to the Stock Market
Using resources from Zerodha Varsity, I developed a fundamental understanding of how markets operate. I learned that the market is a reflection of human psychology and economic value, where indices like the NIFTY 50 track the health of the top companies. This domain knowledge is critical because it helps in understanding why certain features (like volume or opening price) are chosen for the model.

4. Machine Learning Fundamentals
I broke down my learning into Supervised, Unsupervised, and Reinforcement learning.
+2

Regressions: I implemented both linear and logistic regression. I learned that while linear regression predicts a continuous price, logistic regression helps in binary classification, such as predicting if a price will go up or down.
+1


The Overfitting Issue: I studied how models can sometimes memorize noise in training data instead of learning actual trends. I learned to mitigate this by using train-test splits and regularization techniques.
+4

5. Technical Analysis
Again using Zerodha Varsity, I looked into how traders use historical data to find patterns. I focused on candlestick charts, support and resistance levels, and technical indicators. This phase helped me realize that stock prices are not purely random but often follow seasonal or psychological trends that a neural network can potentially pick up.

6. Data Acquisition with YFinance
To get real-world data, I used the YFinance library. This allowed me to automate the process of fetching historical stock data, including daily opening, closing, and high/low prices. This was the first step in creating my own custom dataset for the LSTM model.

7. Neural Networks and How They Learn
I explored the architecture of neural networks, moving from a single neuron to multiple hidden layers. I learned that a network is essentially a series of mathematical weights and biases that get adjusted as the model sees more data.
+2

8. Optimization: Gradient Descent and Backpropagation
I studied how a model actually improves over time.


Gradient Descent: This is the optimization algorithm used to minimize the model's error by finding the local minimum of the loss function.


Backpropagation: I used the example of logistic regression to understand how errors are calculated at the output and sent backward to update each weight in the network.

9. Implementation with PyTorch
PyTorch is the core framework I am using for the final LSTM model.

Tensors and Autograd: I moved from NumPy arrays to Tensors for computation and utilized Autograd for automatic differentiation during training.

Data Handling: I implemented datasets and dataloaders to feed data into the model in batches, which is more memory-efficient.


Modeling: I practiced building basic linear and logistic models in PyTorch, using activation functions like ReLU and Sigmoid, and learning how to save and load these models for future use
