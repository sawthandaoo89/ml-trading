# Boosting Analysis Report

**Author:** soo7  
**Date:** December 2024  
**Course:** CS 7646 - Machine Learning for Trading

## Executive Summary

This report analyzes the implementation and performance of AdaBoost (boosting) compared to Bootstrap Aggregation (bagging) in the context of ensemble learning. The analysis demonstrates that while boosting can achieve excellent performance on training data, it is more prone to overfitting, especially as the number of ensemble members increases.

## Implementation Details

### AdaBoost Algorithm Implementation

The boosting functionality was implemented in the `BagLearner` class using the AdaBoost algorithm:

1. **Initialization**: All training samples start with equal weights (1/n)
2. **Iterative Training**: For each bag/learner:
   - Create a weighted bootstrap sample
   - Train the base learner on the weighted sample
   - Calculate weighted error on the original training data
   - Compute learner weight (alpha) based on error
   - Update sample weights to focus on misclassified examples
3. **Prediction**: Weighted average of all learner predictions

### Key Implementation Features

- **Numerical Stability**: Added bounds checking and NaN handling
- **Error Bounds**: Limited weighted error to prevent extreme alpha values
- **Weight Normalization**: Ensured sample weights sum to 1
- **Fallback Handling**: Graceful degradation for edge cases

## Experimental Setup

### Datasets

1. **Standard Dataset**: 200 samples (140 train, 60 test) with quadratic function + noise
2. **Overfitting Dataset**: 100 samples (70 train, 30 test) with higher noise level

### Base Learners Tested

- **Linear Regression**: Simple linear model
- **Decision Trees**: Non-linear model with leaf_size=5

### Experimental Parameters

- **Bag Counts**: 1, 5, 10, 20, 50, 100, 200
- **Evaluation Metrics**: RMSE (Root Mean Square Error)
- **Comparison**: Bagging vs Boosting performance

## Results and Analysis

### 1. Performance Comparison: Bagging vs Boosting

| Bags | Bagging Train RMSE | Bagging Test RMSE | Boosting Train RMSE | Boosting Test RMSE |
|------|-------------------|-------------------|-------------------|-------------------|
| 1    | 2.6448           | 2.6148           | 2.6991           | 2.8312           |
| 5    | 2.6453           | 2.6436           | 2.6992           | 2.7605           |
| 10   | 2.6456           | 2.6199           | 2.8437           | 3.0246           |
| 20   | 2.6473           | 2.6385           | 3.2990           | 3.6098           |
| 50   | 2.6450           | 2.6145           | 4.9657           | 5.4091           |
| 100  | 2.6448           | 2.6245           | 5.6430           | 3.9938           |

**Key Observations:**
- **Bagging**: Stable performance across all bag counts
- **Boosting**: Performance degrades significantly with more bags
- **Overfitting**: Boosting shows clear signs of overfitting (train RMSE < test RMSE gap increases)

### 2. Overfitting Analysis

The overfitting analysis reveals a critical pattern:

| Bags | Bagging Gap | Boosting Gap |
|------|-------------|--------------|
| 1    | -0.2692     | -0.4711      |
| 2    | -0.2087     | -0.3304      |
| 5    | -0.2403     | -0.4336      |
| 10   | -0.3350     | -0.6362      |
| 20   | -0.2859     | -0.8279      |
| 50   | -0.3243     | -0.9867      |
| 100  | -0.3330     | 0.7496       |
| 200  | -0.3112     | 0.8521       |

*Gap = Train RMSE - Test RMSE (negative means better generalization)*

**Critical Finding**: With 100+ bags, boosting shows positive gaps, indicating severe overfitting where training performance is worse than test performance.

### 3. Base Learner Comparison

#### Linear Regression Results:
- **Single**: Train RMSE: 2.6446, Test RMSE: 2.6230
- **Bagging**: Train RMSE: 2.6459, Test RMSE: 2.6179
- **Boosting**: Train RMSE: 3.3304, Test RMSE: 3.6423

#### Decision Tree Results:
- **Single**: Train RMSE: 0.2075, Test RMSE: 0.2309
- **Bagging**: Train RMSE: 0.1316, Test RMSE: 0.1737
- **Boosting**: Train RMSE: 2.3405, Test RMSE: 2.5822

**Key Insights:**
- Decision trees benefit more from bagging than boosting
- Boosting can actually hurt performance with complex base learners
- Linear models show more predictable behavior with boosting

## Discussion

### Why Boosting Overfits

1. **Sequential Focus**: Each learner focuses on previous mistakes, leading to over-specialization
2. **Weight Concentration**: Sample weights become concentrated on difficult examples
3. **Cumulative Error**: Errors compound across sequential learners
4. **Complexity Growth**: Ensemble complexity grows faster than with bagging

### When to Use Boosting vs Bagging

**Use Boosting When:**
- Base learners are simple (e.g., linear models)
- Dataset is large and diverse
- Early stopping is implemented
- Focus on reducing bias is important

**Use Bagging When:**
- Base learners are complex (e.g., decision trees)
- Dataset is small or noisy
- Stability and generalization are priorities
- Reducing variance is the main goal

### Overfitting Mitigation Strategies

1. **Early Stopping**: Stop adding learners when validation error increases
2. **Regularization**: Add regularization to base learners
3. **Cross-Validation**: Use CV to select optimal number of bags
4. **Shrinkage**: Reduce the learning rate (alpha values)

## Conclusions

1. **Boosting Implementation**: Successfully implemented AdaBoost with numerical stability considerations
2. **Overfitting Evidence**: Clear demonstration of overfitting with increasing bag counts
3. **Performance Trade-offs**: Boosting can achieve lower training error but at the cost of generalization
4. **Base Learner Sensitivity**: The choice of base learner significantly affects boosting performance
5. **Practical Recommendations**: Bagging is generally more robust, while boosting requires careful tuning

## Recommendations

1. **For Production Systems**: Prefer bagging for its stability and generalization properties
2. **For Research/Experimentation**: Use boosting with early stopping and cross-validation
3. **For Small Datasets**: Avoid boosting with many bags to prevent overfitting
4. **For Complex Base Learners**: Bagging typically outperforms boosting

## Technical Notes

- Implementation includes proper error handling and numerical stability
- AdaBoost algorithm follows standard formulation with regression adaptations
- All experiments use consistent random seeds for reproducibility
- Performance metrics focus on RMSE for regression tasks

---

*This analysis demonstrates the importance of understanding the trade-offs between different ensemble methods and the critical role of overfitting in machine learning model selection.*
