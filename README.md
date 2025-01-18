# Overview

# Day-ahead Solar Power Forecasting using LightGBM and Self-Attention Based Encoder-Decoder Networks

## Paper Reference
H. N. Hokmabad, O. Husev, and J. Belikov, *"Day-ahead Solar Power Forecasting Using LightGBM and Self-Attention Based Encoder-Decoder Networks,"* IEEE Transactions on Sustainable Energy, DOI: [10.1109/TSTE.2024.3486907](https://doi.org/10.1109/TSTE.2024.3486907)

## Abstract
The integration of renewable energy harvesters, such as solar power, into the grid presents challenges to its stability due to the stochastic and intermittent nature of these sources. Data-driven forecasting methods are critical in addressing these challenges, but they can underperform when sufficient historical data is not available. This paper introduces a novel hybrid forecasting framework for day-ahead photovoltaic (PV) power prediction. The framework integrates a physics-based model with machine learning (ML) techniques, providing enhanced reliability in environments with limited data. The ML tool proposed in the paper features two branches: a set of regressors tailored to specific weather conditions and a self-attention-based encoder-decoder network. The outputs from both branches are fused through a meta-learner, achieving superior forecasting performance compared to benchmark models.

## Repository Overview

This repository contains the code and datasets used in the paper for the development and evaluation of the day-ahead solar power forecasting framework. It includes:
- Code for the hybrid forecasting model integrating LightGBM, self-attention-based encoder-decoder networks, and physics-based models.
- Datasets for training and testing the proposed models.
- Preprocessing and evaluation scripts to assess the performance of the model. (This part is under development and will be updatet soon.)

## Requirements

- Python 3.8+
- LightGBM
- TensorFlow/Keras (for neural network models)
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Other dependencies as specified in `requirements.txt`

## Model Architecture
The hybrid model consists of the following components:

- Weather data classifier: A weather condition classifier based on SYNOPTIC code weather classification standard. 
- LightGBM: A gradient boosting framework used for regression tasks, particularly well-suited for large datasets with categorical features.
- Self-Attention-based Encoder-Decoder Network: A neural network model that uses self-attention mechanisms to capture long-range dependencies in time-series data, suitable for sequential forecasting tasks.
Meta-learner: A model that combines the predictions from multiple branches to improve overall forecasting accuracy.

## Results

The proposed framework outperforms benchmark models in terms of prediction accuracy. Detailed results are included in the paper.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work was supported by the Estonian Research Council grants PRG675 and PRG1463.

## Citation

If you use this repository in your research, please cite the following paper: 

H. N. Hokmabad, O. Husev, and J. Belikov, "Day-ahead Solar Power Forecasting Using LightGBM and Self-Attention Based Encoder-Decoder Networks," *IEEE Transactions on Sustainable Energy*, DOI: [10.1109/TSTE.2024.3486907](https://doi.org/10.1109/TSTE.2024.3486907)



