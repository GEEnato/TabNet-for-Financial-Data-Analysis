# TabNet for Financial Data Analysis (Credit Card Fraud Detection)

##  Overview
This repository implements **TabNet**, an attention-based deep learning architecture for tabular data, to tackle challenges in **financial data analysis**, focusing on **credit card fraud detection**.  

Fraudulent transactions are rare, imbalanced, and evolve, making traditional machine learning approaches less effective. TabNet leverages **sequential attention** to achieve both **interpretability** and **high performance**.

##  Features
- Implementation of **TabNet** in PyTorch  
- Comparison with baseline models: **Decision Tree** and **Multi-Layer Perceptron (MLP)**  
- Techniques to address **class imbalance**:
  - SMOTE (Synthetic Minority Oversampling Technique)  
  - SVMSMOTE (Support Vector Machine SMOTE)  
- Evaluation with key metrics:
  - Precision, Recall, F1-Score, ROC-AUC  
- Training and testing on the **Kaggle Credit Card Fraud Dataset**  

##  Dataset
The dataset used is the publicly available [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
- **Transactions:** 284,807  
- **Fraudulent:** 492 (0.17%)  
- **Legitimate:** 284,315  

##  Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/GEEnato/TabNet-for-Financial-Data-Analysis.git
cd tabnet-financial-analysis
pip install -r requirements.txt
```


##  Future Work
- Hyperparameter tuning for optimal TabNet performance  
- Comparative analysis with other deep learning models (XGBoost, LightGBM, etc.)  
- Deployment-ready pipeline for real-time fraud detection  

##  References
- Arik, S. Ö., & Pfister, T. (2021). **TabNet: Attentive Interpretable Tabular Learning**. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(8), 6679–6687.  
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). **SMOTE: Synthetic Minority Over-sampling Technique**. *Journal of Artificial Intelligence Research*, 16, 321–357.  
- Nguyen, H. M., Cooper, E. W., & Kamei, K. (2009). **Borderline over-sampling for imbalanced data classification**. *International Journal of Knowledge Engineering and Soft Data Paradigms*, 3(1), 4–21.  

##  Author
Grace Enato  
MSc Financial Modelling & Optimization, University of Edinburgh  
Research focus: **Machine Learning for Financial Data Analysis**
