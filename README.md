# Gradient Boosting Algorithms Comparison

A comprehensive comparison of four popular gradient boosting algorithms using the Bank Marketing dataset from UCI Machine Learning Repository.

## 🎯 Objective

Compare the performance of the following gradient boosting algorithms:
- **XGBoost** (Extreme Gradient Boosting)
- **LightGBM** (Light Gradient Boosting Machine)
- **CatBoost** (Categorical Boosting)
- **NGBoost** (Natural Gradient Boosting)

## 📊 Dataset

**Bank Marketing Dataset** from UCI ML Repository
- **Source**: https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Task**: Binary Classification
- **Goal**: Predict if a client will subscribe to a term deposit
- **Instances**: 45,211
- **Features**: 16 (mix of categorical and numerical)
- **Target**: `y` (yes/no - term deposit subscription)

### Dataset Features

**Bank Client Data:**
- `age`: Numeric
- `job`: Categorical (12 categories)
- `marital`: Categorical (married, divorced, single)
- `education`: Categorical (4 levels)
- `default`: Binary (has credit in default?)
- `balance`: Numeric (average yearly balance in euros)
- `housing`: Binary (has housing loan?)
- `loan`: Binary (has personal loan?)

**Last Contact Information:**
- `contact`: Categorical (cellular, telephone)
- `day`: Numeric (last contact day of month)
- `month`: Categorical
- `duration`: Numeric (last contact duration in seconds)

**Campaign Attributes:**
- `campaign`: Numeric (number of contacts during campaign)
- `pdays`: Numeric (days since last contact from previous campaign)
- `previous`: Numeric (number of contacts before this campaign)
- `poutcome`: Categorical (outcome of previous campaign)

## 📈 What the Script Does

1. **Data Loading**: Automatically fetches the Bank Marketing dataset from UCI ML Repository
2. **Preprocessing**: 
   - Encodes categorical variables
   - Handles binary target encoding
   - Splits data into train/test sets (80/20)
   
3. **Model Training**: Trains all four algorithms with similar hyperparameters for fair comparison:
   - n_estimators: 100
   - max_depth: 6
   - learning_rate: 0.1 (0.01 for NGBoost)
   
4. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
   - Training Time
   - Confusion Matrix
   
5. **Visualizations**:
   - Feature importance plots for each model
     - XGBoost, LightGBM, CatBoost: Built-in feature importance
     - NGBoost: Permutation importance (mean decrease in ROC-AUC when feature is shuffled)
   - Performance comparison charts
   - Bar charts for all metrics

6. **Output Files**:
   - `model_comparison_results.csv`: Detailed results table
   - `feature_importance_comparison.png`: Feature importance visualization
   - `performance_comparison.png`: Performance metrics comparison

## 🔍 Gradient Boosting Methods

### X and Y (features & target)

- X: feature matrix = all dataset features (age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome). After preprocessing categorical features are numeric (one-hot/label encoded or CatBoost categorical). Typical type: `pandas.DataFrame` → `numpy.ndarray`.
- y: target vector = column `y` ("yes"/"no") encoded as binary (`yes` → 1, `no` → 0).
- Shapes: full dataset ≈ (45211, 16) for X and (45211,) for y. After 80/20 split: train ≈ (36168, 16), test ≈ (9043, 16).
- Predictions: `model.predict(X_test)` → class labels (0/1); `model.predict_proba(X_test)[:, 1]` → probability of positive class. NGBoost may return predictive distributions (use its `.predict_dist` / API).

### XGBoost
- **Strengths**: 
  - High performance and accuracy
  - Built-in regularization
  - Handles missing values
  - Parallel processing
- **Use Case**: General-purpose gradient boosting, competitions

### LightGBM
- **Strengths**:
  - Very fast training speed
  - Memory efficient
  - Leaf-wise tree growth
  - Good for large datasets
- **Use Case**: Large-scale datasets, production systems

### CatBoost
- **Strengths**:
  - Native categorical feature handling
  - Reduces overfitting
  - Ordered boosting
  - No need for extensive preprocessing
- **Use Case**: Datasets with many categorical features

### NGBoost
- **Strengths**:
  - Probabilistic predictions
  - Uncertainty quantification
  - Natural gradient boosting
  - Flexible distribution support
- **Use Case**: When prediction uncertainty is important
- **Note**: Uses permutation importance instead of built-in feature importance (measures impact on ROC-AUC when features are shuffled)

## Evaluation Criteria
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Correct positive predictions out of all positive predictions.
- **Recall**: Correct positive predictions out of all actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the ROC curve, measures classification performance.
- **Training Time**: Time taken to train the model.

## 📊 Expected Output

The script will print:
- Dataset information and statistics
- Training progress for each model
- Performance metrics for each algorithm
- Summary comparison table
- Best model for each metric

Example output:
```
================================================================================
SUMMARY: Model Comparison
================================================================================

     Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC  Training Time
  XGBoost    0.9012     0.8543   0.7234    0.7833   0.8912          2.34
 LightGBM    0.8998     0.8512   0.7198    0.7801   0.8901          1.23
 CatBoost    0.9034     0.8567   0.7312    0.7876   0.8934          4.56
  NGBoost    0.8876     0.8234   0.6876    0.7498   0.8723         12.45

================================================================================
Best Models by Metric
================================================================================
Accuracy    : CatBoost     (0.9034)
Precision   : CatBoost     (0.8567)
Recall      : CatBoost     (0.7312)
F1-Score    : CatBoost     (0.7876)
ROC-AUC     : CatBoost     (0.8934)
Fastest     : LightGBM     (1.23s)
```

1. **Accuracy**: All models typically achieve 88-91% accuracy
2. **Speed**: LightGBM is usually the fastest, NGBoost the slowest
3. **Categorical Handling**: CatBoost often excels due to native categorical support
4. **Feature Importance**: Duration, balance, and age typically rank high
5. **Imbalanced Data**: The dataset is imbalanced (~11% positive class)

## 📚 References

- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- CatBoost: https://catboost.ai/
- NGBoost: https://github.com/stanfordmlgroup/ngboost
- Bank Marketing Dataset: [Moro et al., 2014]
- Top Gradient Boosting Methods: https://blog.dailydoseofds.com/p/top-gradient-boosting-methods
- Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing
