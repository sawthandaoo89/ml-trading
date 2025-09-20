#!/usr/bin/env python3
"""
Boosting Analysis Script
Compares bagging vs boosting performance and demonstrates overfitting
"""

import numpy as np
import matplotlib.pyplot as plt
import BagLearner as bl
import LinRegLearner as lrl
import DTLearner as dt

def create_overfitting_dataset(n_samples=200, noise_level=0.1):
    """
    Create a dataset designed to show overfitting with boosting
    - Simple underlying function with noise
    - Small training set relative to complexity
    """
    np.random.seed(42)
    
    # Create a simple underlying function: y = x^2 + noise
    X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    y_true = X.flatten() ** 2
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_true + noise
    
    # Split into train/test
    train_size = int(0.7 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

def evaluate_learner(learner, X_train, X_test, y_train, y_test):
    """Evaluate a learner and return in-sample and out-of-sample metrics"""
    learner.add_evidence(X_train, y_train)
    
    # In-sample predictions
    y_pred_train = learner.query(X_train)
    train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
    
    # Out-of-sample predictions
    y_pred_test = learner.query(X_test)
    test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    test_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
    
    return {
        'train_rmse': train_rmse,
        'train_corr': train_corr,
        'test_rmse': test_rmse,
        'test_corr': test_corr,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }

def compare_bagging_boosting():
    """Compare bagging vs boosting performance across different numbers of bags"""
    print("=== Boosting vs Bagging Performance Analysis ===\n")
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_overfitting_dataset()
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Test different numbers of bags
    bag_counts = [1, 5, 10, 20, 50, 100]
    results = {'bagging': [], 'boosting': []}
    
    for bags in bag_counts:
        print(f"\nTesting with {bags} bags...")
        
        # Test bagging
        bag_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, 
                                   bags=bags, boost=False, verbose=False)
        bag_results = evaluate_learner(bag_learner, X_train, X_test, y_train, y_test)
        results['bagging'].append(bag_results)
        
        # Test boosting
        boost_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, 
                                     bags=bags, boost=True, verbose=False)
        boost_results = evaluate_learner(boost_learner, X_train, X_test, y_train, y_test)
        results['boosting'].append(boost_results)
        
        print(f"  Bagging  - Train RMSE: {bag_results['train_rmse']:.4f}, Test RMSE: {bag_results['test_rmse']:.4f}")
        print(f"  Boosting - Train RMSE: {boost_results['train_rmse']:.4f}, Test RMSE: {boost_results['test_rmse']:.4f}")
    
    return bag_counts, results

def analyze_overfitting():
    """Analyze overfitting behavior with increasing number of bags"""
    print("\n=== Overfitting Analysis ===\n")
    
    # Create a smaller, noisier dataset to encourage overfitting
    X_train, X_test, y_train, y_test = create_overfitting_dataset(n_samples=100, noise_level=0.2)
    print(f"Overfitting dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    bag_counts = [1, 2, 5, 10, 20, 50, 100, 200]
    overfitting_results = {'bagging': [], 'boosting': []}
    
    for bags in bag_counts:
        # Test bagging
        bag_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, 
                                   bags=bags, boost=False, verbose=False)
        bag_results = evaluate_learner(bag_learner, X_train, X_test, y_train, y_test)
        overfitting_results['bagging'].append(bag_results)
        
        # Test boosting
        boost_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, 
                                     bags=bags, boost=True, verbose=False)
        boost_results = evaluate_learner(boost_learner, X_train, X_test, y_train, y_test)
        overfitting_results['boosting'].append(boost_results)
        
        # Calculate overfitting gap (difference between train and test RMSE)
        bag_gap = bag_results['train_rmse'] - bag_results['test_rmse']
        boost_gap = boost_results['train_rmse'] - boost_results['test_rmse']
        
        print(f"Bags: {bags:3d} | Bagging gap: {bag_gap:6.4f} | Boosting gap: {boost_gap:6.4f}")
    
    return bag_counts, overfitting_results

def test_with_different_learners():
    """Test boosting with different base learners"""
    print("\n=== Testing Boosting with Different Base Learners ===\n")
    
    X_train, X_test, y_train, y_test = create_overfitting_dataset()
    
    learners = [
        ('LinRegLearner', lrl.LinRegLearner, {}),
        ('DTLearner', dt.DTLearner, {'leaf_size': 5})
    ]
    
    for name, learner_class, kwargs in learners:
        print(f"Testing {name}:")
        
        # Test single learner
        single_learner = learner_class(**kwargs)
        single_results = evaluate_learner(single_learner, X_train, X_test, y_train, y_test)
        
        # Test bagging
        bag_learner = bl.BagLearner(learner=learner_class, kwargs=kwargs, 
                                   bags=20, boost=False, verbose=False)
        bag_results = evaluate_learner(bag_learner, X_train, X_test, y_train, y_test)
        
        # Test boosting
        boost_learner = bl.BagLearner(learner=learner_class, kwargs=kwargs, 
                                     bags=20, boost=True, verbose=False)
        boost_results = evaluate_learner(boost_learner, X_train, X_test, y_train, y_test)
        
        print(f"  Single:  Train RMSE: {single_results['train_rmse']:.4f}, Test RMSE: {single_results['test_rmse']:.4f}")
        print(f"  Bagging: Train RMSE: {bag_results['train_rmse']:.4f}, Test RMSE: {bag_results['test_rmse']:.4f}")
        print(f"  Boosting:Train RMSE: {boost_results['train_rmse']:.4f}, Test RMSE: {boost_results['test_rmse']:.4f}")
        print()

def main():
    """Main analysis function"""
    print("Boosting Analysis Report")
    print("=" * 50)
    
    # Run all analyses
    bag_counts, results = compare_bagging_boosting()
    overfitting_bags, overfitting_results = analyze_overfitting()
    test_with_different_learners()
    
    # Summary
    print("\n=== Summary of Findings ===")
    print("1. Boosting generally achieves lower training error than bagging")
    print("2. Boosting can overfit more easily with many bags")
    print("3. Bagging tends to be more stable and generalizes better")
    print("4. The choice between bagging and boosting depends on the dataset and base learner")
    
    return bag_counts, results, overfitting_bags, overfitting_results

if __name__ == "__main__":
    main()
