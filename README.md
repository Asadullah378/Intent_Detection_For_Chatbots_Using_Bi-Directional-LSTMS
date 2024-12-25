# Intent Detection for Chatbots Using Bi-Directional LSTMs

This repository contains the code and resources for the research project titled **"Intent Detection for Chatbots Using Bi-Directional LSTMs"**. The project demonstrates the implementation and evaluation of Bi-Directional Long Short-Term Memory (Bi-LSTM) models for detecting user intents in natural language queries, particularly in customer support scenarios.

The research paper can be found at: [Research Paper](https://drive.google.com/file/d/1lqlP_TN2QQdnNjXnqsHqg2LKIW-QzJL-/view)

## Overview

The primary goal of this project is to enhance chatbot capabilities in understanding customer intents accurately using deep learning models. By leveraging the **Bitext Customer Support Dataset**, this research compares various deep learning architectures, including:

- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory Networks (LSTM)
- Bi-Directional LSTM (Bi-LSTM)

The **Bi-LSTM model**, after rigorous hyperparameter tuning, achieved the best performance across key metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

## Features

- **Dataset Preprocessing**: Includes data cleaning, tokenization, stop-word removal, hash encoding, and padding.
- **Deep Learning Models**: Implements multiple neural network architectures.
- **Performance Metrics**: Evaluates models based on accuracy, precision, recall, and F1-score.
- **Hyperparameter Optimization**: Enhances the Bi-LSTM model's performance.

## Dataset

The project uses the **Bitext Customer Support Dataset**, which includes 26,872 question-answer pairs categorized into 27 intents. The dataset is balanced and provides a robust foundation for training intent detection models.

| Metric               | Value   |
|----------------------|---------|
| Total Samples        | 26,872  |
| Training Samples     | 21,497  |
| Test Samples         | 5,375   |
| Total Intents        | 27      |

## Methodology

1. **Data Preprocessing**:
   - Standardizes text by removing unwanted characters, converting to lowercase, and tokenizing.
   - Applies padding for uniform sequence lengths.
   
2. **Model Architectures**:
   - ANN: Basic dense layers with ReLU and Softmax activations.
   - CNN: Convolutional layers with max pooling for feature extraction.
   - LSTM: Captures sequential dependencies in text.
   - Bi-LSTM: Enhances context understanding by processing input in both forward and backward directions.

3. **Training and Evaluation**:
   - Models trained using TensorFlow and Keras libraries.
   - Evaluation metrics include accuracy, precision, recall, and F1-score.

## Results

| Model      | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|------------|--------------|---------------|------------|--------------|
| ANN        | 14           | 15            | 14         | 12           |
| CNN        | 17           | 14            | 17         | 12           |
| LSTM       | 73           | 78            | 73         | 71           |
| Bi-LSTM    | 95           | 95            | 95         | 95           |

After hyperparameter tuning, the Bi-LSTM model achieved:

- **Accuracy**: 98%
- **Precision**: 98%
- **Recall**: 98%
- **F1-Score**: 98%

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
