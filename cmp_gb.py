"""
Gradient Boosting Algorithms Comparison
Dataset: Bank Marketing from UCI ML Repository
Task: Binary Classification - Predict term deposit subscription

Algorithms Compared:
- XGBoost
- LightGBM
- CatBoost
- NGBoost
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Import gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_bank_marketing_data():
    """
    Load Bank Marketing dataset from UCI ML Repository
    """
    print("=" * 80)
    print("Loading Bank Marketing Dataset from UCI ML Repository")
    print("=" * 80)
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset
        bank_marketing = fetch_ucirepo(id=222)
        
        # Data (as pandas dataframes)
        X = bank_marketing.data.features
        y = bank_marketing.data.targets
        
        # Print dataset info
        print(f"\nDataset Shape: {X.shape}")
        print(f"Target Shape: {y.shape}")
        print(f"\nFeatures: {list(X.columns)}")
        print(f"\nTarget Variable: {list(y.columns)}")
        print(f"\nClass Distribution:\n{y.value_counts()}")
        
        return X, y
    
    except ImportError:
        print("\nError: ucimlrepo package not found.")
        print("Installing ucimlrepo...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'ucimlrepo'])
        print("Please run the script again.")
        exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nAlternatively, you can download the dataset manually from:")
        print("https://archive.ics.uci.edu/dataset/222/bank+marketing")
        exit(1)


def preprocess_data(X, y):
    """
    Preprocess the data for modeling
    """
    print("\n" + "=" * 80)
    print("Preprocessing Data")
    print("=" * 80)
    
    # Convert target to binary (0/1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.values.ravel())
    
    print(f"\nTarget Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical Features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")
    
    # Handle missing values if any
    X_processed = X.copy()
    
    # For non-CatBoost models, encode categorical variables
    X_encoded = X_processed.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
    
    return X_processed, X_encoded, y_encoded, categorical_cols, numerical_cols


def split_data(X_encoded, y_encoded, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    print("\n" + "=" * 80)
    print("Splitting Data")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"\nTrain Set: {X_train.shape}")
    print(f"Test Set: {X_test.shape}")
    print(f"Train Class Distribution: {np.bincount(y_train)}")
    print(f"Test Class Distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train and evaluate XGBoost model
    """
    print("\n" + "=" * 80)
    print("Training XGBoost")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba, training_time


def train_lightgbm(X_train, X_test, y_train, y_test):
    """
    Train and evaluate LightGBM model
    """
    print("\n" + "=" * 80)
    print("Training LightGBM")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba, training_time


def train_catboost(X_train, X_test, y_train, y_test, categorical_features):
    """
    Train and evaluate CatBoost model
    """
    print("\n" + "=" * 80)
    print("Training CatBoost")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create model
    model = cb.CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )
    
    # Train model
    # CatBoost can handle categorical features natively
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        cat_features=categorical_features
    )
    
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba, training_time


def train_ngboost(X_train, X_test, y_train, y_test):
    """
    Train and evaluate NGBoost model
    """
    print("\n" + "=" * 80)
    print("Training NGBoost")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create model
    model = NGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        random_state=42,
        verbose=False,
        Dist=Bernoulli
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return model, y_pred, y_pred_proba, training_time


def evaluate_model(name, y_test, y_pred, y_pred_proba, training_time):
    """
    Evaluate model performance
    """
    print(f"\n{name} Performance Metrics:")
    print("-" * 60)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Training Time': training_time
    }


def plot_feature_importance(models_dict, X_train, y_train, top_n=15):
    """
    Plot feature importance for each model
    """
    print("\n" + "=" * 80)
    print("Plotting Feature Importance")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models_dict.items()):
        if name == 'NGBoost':
            # Compute permutation importance for NGBoost (probabilistic model)
            # NGBoost doesn't have built-in feature_importances_, so use permutation importance
            print(f"\nComputing permutation importance for {name} (this may take a moment)...")
            try:
                def ngboost_auc_scorer(estimator, X, y):
                    return roc_auc_score(y, estimator.predict_proba(X)[:, 1])
                
                r = permutation_importance(
                    model,
                    X_train,
                    y_train,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                    scoring=ngboost_auc_scorer
                )
                importance = r.importances_mean
                feature_importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importance
                }).sort_values('Importance', ascending=False).head(top_n)
                
                axes[idx].barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
                axes[idx].set_yticks(range(len(feature_importance_df)))
                axes[idx].set_yticklabels(feature_importance_df['Feature'])
                axes[idx].invert_yaxis()
                axes[idx].set_xlabel('Permutation Importance (mean decrease in ROC-AUC)')
                axes[idx].set_title(f'{name} - Top {top_n} Features (Permutation)')
                axes[idx].grid(axis='x', alpha=0.3)
                print(f"✓ Permutation importance computed successfully for {name}")
            except Exception as e:
                print(f"✗ Error computing permutation importance for {name}: {str(e)}")
                axes[idx].text(0.5, 0.5, f'Could not compute\npermutation importance\n\nError: {str(e)[:50]}...',
                              ha='center', va='center', fontsize=10, wrap=True)
                axes[idx].set_title(f'{name} - Feature Importance (Error)')
                axes[idx].axis('off')
            continue
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            importance = model.get_feature_importance()
        
        # Create dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        axes[idx].barh(range(len(feature_importance_df)), feature_importance_df['Importance'])
        axes[idx].set_yticks(range(len(feature_importance_df)))
        axes[idx].set_yticklabels(feature_importance_df['Feature'])
        axes[idx].invert_yaxis()
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{name} - Top {top_n} Features')
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved as 'feature_importance_comparison.png'")
    # plt.show()


def plot_performance_comparison(results_df):
    """
    Plot performance comparison across models
    """
    print("\n" + "=" * 80)
    print("Plotting Performance Comparison")
    print("=" * 80)
    
    # Metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time']
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        
        if metric == 'Training Time':
            axes[row, col].bar(results_df['Model'], results_df[metric], color='coral')
            axes[row, col].set_ylabel('Time (seconds)')
        else:
            axes[row, col].bar(results_df['Model'], results_df[metric], color='skyblue')
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_ylim([0, 1])
        
        axes[row, col].set_title(f'{metric} Comparison')
        axes[row, col].set_xlabel('Model')
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(results_df[metric]):
            if metric == 'Training Time':
                axes[row, col].text(i, v, f'{v:.2f}', ha='center', va='bottom')
            else:
                axes[row, col].text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPerformance comparison plot saved as 'performance_comparison.png'")
    # plt.show()


def print_summary_table(results_df):
    """
    Print summary comparison table
    """
    print("\n" + "=" * 80)
    print("SUMMARY: Model Comparison")
    print("=" * 80)
    print("\n", results_df.to_string(index=False))
    
    # Find best model for each metric
    print("\n" + "=" * 80)
    print("Best Models by Metric")
    print("=" * 80)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for metric in metrics:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"{metric:12s}: {best_model:12s} ({best_score:.4f})")
    
    # Fastest model
    fastest_idx = results_df['Training Time'].idxmin()
    fastest_model = results_df.loc[fastest_idx, 'Model']
    fastest_time = results_df.loc[fastest_idx, 'Training Time']
    print(f"{'Fastest':12s}: {fastest_model:12s} ({fastest_time:.2f}s)")


def main():
    """
    Main function to run the comparison
    """
    print("\n" + "=" * 80)
    print("GRADIENT BOOSTING ALGORITHMS COMPARISON")
    print("Dataset: Bank Marketing (UCI ML Repository)")
    print("=" * 80)
    
    # Load data
    X, y = load_bank_marketing_data()
    
    # Preprocess data
    X_processed, X_encoded, y_encoded, categorical_cols, numerical_cols = preprocess_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_encoded, y_encoded)
    
    # Store results
    results = []
    models = {}
    
    # Train XGBoost
    xgb_model, xgb_pred, xgb_proba, xgb_time = train_xgboost(X_train, X_test, y_train, y_test)
    xgb_results = evaluate_model('XGBoost', y_test, xgb_pred, xgb_proba, xgb_time)
    results.append(xgb_results)
    models['XGBoost'] = xgb_model
    
    # Train LightGBM
    lgb_model, lgb_pred, lgb_proba, lgb_time = train_lightgbm(X_train, X_test, y_train, y_test)
    lgb_results = evaluate_model('LightGBM', y_test, lgb_pred, lgb_proba, lgb_time)
    results.append(lgb_results)
    models['LightGBM'] = lgb_model
    
    # Train CatBoost
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]
    cat_model, cat_pred, cat_proba, cat_time = train_catboost(X_train, X_test, y_train, y_test, cat_indices)
    cat_results = evaluate_model('CatBoost', y_test, cat_pred, cat_proba, cat_time)
    results.append(cat_results)
    models['CatBoost'] = cat_model
    
    # Train NGBoost
    ngb_model, ngb_pred, ngb_proba, ngb_time = train_ngboost(X_train, X_test, y_train, y_test)
    ngb_results = evaluate_model('NGBoost', y_test, ngb_pred, ngb_proba, ngb_time)
    results.append(ngb_results)
    models['NGBoost'] = ngb_model
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print summary
    print_summary_table(results_df)
    
    # Plot comparisons
    plot_feature_importance(models, X_train, y_train)
    plot_performance_comparison(results_df)
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")
    
    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
