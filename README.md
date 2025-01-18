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

## Model Architecture
The hybrid model consists of the following components:

- Weather data classifier: A weather condition classifier based on SYNOPTIC code weather classification standard. 
- LightGBM: A gradient boosting framework used for regression tasks, particularly well-suited for large datasets with categorical features.
- Self-Attention-based Encoder-Decoder Network: A neural network model that uses self-attention mechanisms to capture long-range dependencies in time-series data, suitable for sequential forecasting tasks.
Meta-learner: A model that combines the predictions from multiple branches to improve overall forecasting accuracy.

## Figure:
![Image](https://github.com/user-attachments/assets/4334b356-52b0-44fc-9be6-30b8a9d7269d)  <br />
![Image](https://github.com/user-attachments/assets/d70fda75-c782-4684-b276-74aac05a0d09)  <br />
![Image](https://github.com/user-attachments/assets/c9698884-1b68-44e0-957d-c1e152ea4c95)  <br />
![Image](https://github.com/user-attachments/assets/6f874c52-36df-4149-a653-033c4b55d0d3)  <br />
![Image](https://github.com/user-attachments/assets/a09e69e1-56f2-42ba-b562-f9c992f6a8d8)  <br />
![Image](https://github.com/user-attachments/assets/52b04929-4bf7-429e-bb01-ec67cb267331)
![Image](https://github.com/user-attachments/assets/e207481e-0972-4c1c-ba77-de196eaa04bb)
![Image](https://github.com/user-attachments/assets/c489951f-0fe4-4309-a914-6f08febced16)
![Image](https://github.com/user-attachments/assets/210af363-0704-4cc4-b635-0267a07fa359)
![Image](https://github.com/user-attachments/assets/2084a7fa-5683-438d-a77c-34d6ab6539e1)
![Image](https://github.com/user-attachments/assets/dec0a91c-cada-431a-bf18-59933442d4f2)
![Image](https://github.com/user-attachments/assets/8857513f-8f0c-4163-a8a5-af2b676dd7cd)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work was supported by the Estonian Research Council grants PRG675 and PRG1463.

## Citation

If you use this repository in your research, please cite the following paper: 

H. N. Hokmabad, O. Husev, and J. Belikov, "Day-ahead Solar Power Forecasting Using LightGBM and Self-Attention Based Encoder-Decoder Networks," *IEEE Transactions on Sustainable Energy*, DOI: [10.1109/TSTE.2024.3486907](https://doi.org/10.1109/TSTE.2024.3486907)



