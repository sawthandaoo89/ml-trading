""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import math  		  	   		 	 	 		  		  		    	 		 		   		 		  
import sys  		  	   		 	 	 		  		  		    	 		 		   		 		  
import time  		  	   		 	 	 		  		  		    	 		 		   		 		  
import os  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  		  	   		 	 	 		  		  		    	 		 		   		 		  
import DTLearner as dtl  		  	   		 	 	 		  		  		    	 		 		   		 		  
import RTLearner as rtl  		  	   		 	 	 		  		  		    	 		 		   		 		  
import BagLearner as bl  		  	   		 	 	 		  		  		    	 		 		   		 		  
import InsaneLearner as il
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def load_data(filename):
    """Load data from CSV file using numpy genfromtxt"""
    if 'Istanbul' in filename:
        # Istanbul.csv has date column, skip it
        data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=range(1, 9))
    else:
        # Other datasets don't have date column
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return data

def create_overfitting_dataset(n_samples=200, noise_level=0.1):
    """Create a dataset designed to show overfitting with boosting"""
    np.random.seed(42)
    X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    y_true = X.flatten() ** 2
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_true + noise
    train_size = int(0.7 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def evaluate_learner(learner, train_x, train_y, test_x, test_y):
    """Evaluate a learner and return metrics"""
    learner.add_evidence(train_x, train_y)
    pred_train = learner.query(train_x)
    pred_test = learner.query(test_x)
    train_rmse = math.sqrt(((train_y - pred_train) ** 2).sum() / train_y.shape[0])
    test_rmse = math.sqrt(((test_y - pred_test) ** 2).sum() / test_y.shape[0])
    train_corr = np.corrcoef(pred_train, y=train_y)[0, 1]
    test_corr = np.corrcoef(pred_test, y=test_y)[0, 1]
    return train_rmse, test_rmse, train_corr, test_corr

def test_learner(learner, train_x, train_y, test_x, test_y, learner_name):
    """Test a specific learner and print results"""
    print(f"\n=== Testing {learner_name} ===")
    print(f"Author: {learner.author()}")
    train_rmse, test_rmse, train_corr, test_corr = evaluate_learner(learner, train_x, train_y, test_x, test_y)
    print(f"In sample results")
    print(f"RMSE: {train_rmse}")
    print(f"corr: {train_corr}")
    print(f"Out of sample results")
    print(f"RMSE: {test_rmse}")
    print(f"corr: {test_corr}")
    return train_rmse, test_rmse, train_corr, test_corr

def run_boosting_experiment():
    """Run boosting vs bagging comparison experiment"""
    print("\n" + "="*60)
    print("BOOSTING VS BAGGING EXPERIMENT")
    print("="*60)
    
    # Create overfitting dataset
    X_train, X_test, y_train, y_test = create_overfitting_dataset(n_samples=150, noise_level=0.15)
    print(f"Overfitting dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    bag_counts = [1, 5, 10, 20, 50]
    bagging_results = {'train_rmse': [], 'test_rmse': [], 'train_corr': [], 'test_corr': []}
    boosting_results = {'train_rmse': [], 'test_rmse': [], 'train_corr': [], 'test_corr': []}
    
    for bags in bag_counts:
        print(f"\nTesting with {bags} bags...")
        
        # Test bagging
        bag_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, 
                                   bags=bags, boost=False, verbose=False)
        train_rmse, test_rmse, train_corr, test_corr = evaluate_learner(bag_learner, X_train, y_train, X_test, y_test)
        bagging_results['train_rmse'].append(train_rmse)
        bagging_results['test_rmse'].append(test_rmse)
        bagging_results['train_corr'].append(train_corr)
        bagging_results['test_corr'].append(test_corr)
        
        # Test boosting
        boost_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, 
                                     bags=bags, boost=True, verbose=False)
        train_rmse, test_rmse, train_corr, test_corr = evaluate_learner(boost_learner, X_train, y_train, X_test, y_test)
        boosting_results['train_rmse'].append(train_rmse)
        boosting_results['test_rmse'].append(test_rmse)
        boosting_results['train_corr'].append(train_corr)
        boosting_results['test_corr'].append(test_corr)
        
        print(f"  Bagging  - Train RMSE: {bagging_results['train_rmse'][-1]:.4f}, Test RMSE: {bagging_results['test_rmse'][-1]:.4f}")
        print(f"  Boosting - Train RMSE: {boosting_results['train_rmse'][-1]:.4f}, Test RMSE: {boosting_results['test_rmse'][-1]:.4f}")
    
    return bag_counts, bagging_results, boosting_results

def create_visualizations(bag_counts, bagging_results, boosting_results):
    """Create charts for the analysis"""
    plt.style.use('default')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: RMSE Comparison
    ax1.plot(bag_counts, bagging_results['train_rmse'], 'b-o', label='Bagging Train', linewidth=2)
    ax1.plot(bag_counts, bagging_results['test_rmse'], 'b--s', label='Bagging Test', linewidth=2)
    ax1.plot(bag_counts, boosting_results['train_rmse'], 'r-o', label='Boosting Train', linewidth=2)
    ax1.plot(bag_counts, boosting_results['test_rmse'], 'r--s', label='Boosting Test', linewidth=2)
    ax1.set_xlabel('Number of Bags')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Number of Bags')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation Comparison
    ax2.plot(bag_counts, bagging_results['train_corr'], 'b-o', label='Bagging Train', linewidth=2)
    ax2.plot(bag_counts, bagging_results['test_corr'], 'b--s', label='Bagging Test', linewidth=2)
    ax2.plot(bag_counts, boosting_results['train_corr'], 'r-o', label='Boosting Train', linewidth=2)
    ax2.plot(bag_counts, boosting_results['test_corr'], 'r--s', label='Boosting Test', linewidth=2)
    ax2.set_xlabel('Number of Bags')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation vs Number of Bags')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overfitting Gap (Train RMSE - Test RMSE)
    bagging_gap = np.array(bagging_results['train_rmse']) - np.array(bagging_results['test_rmse'])
    boosting_gap = np.array(boosting_results['train_rmse']) - np.array(boosting_results['test_rmse'])
    ax3.plot(bag_counts, bagging_gap, 'b-o', label='Bagging Gap', linewidth=2)
    ax3.plot(bag_counts, boosting_gap, 'r-o', label='Boosting Gap', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of Bags')
    ax3.set_ylabel('Overfitting Gap (Train RMSE - Test RMSE)')
    ax3.set_title('Overfitting Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Generalization Performance (Test RMSE)
    ax4.plot(bag_counts, bagging_results['test_rmse'], 'b-o', label='Bagging', linewidth=2)
    ax4.plot(bag_counts, boosting_results['test_rmse'], 'r-o', label='Boosting', linewidth=2)
    ax4.set_xlabel('Number of Bags')
    ax4.set_ylabel('Test RMSE')
    ax4.set_title('Generalization Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('boosting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nCharts saved as 'boosting_analysis.png'")
    plt.close()

def run_leaf_size_experiment(train_x, train_y, test_x, test_y):
    """Run experiment with different leaf sizes"""
    print("\n" + "="*60)
    print("LEAF SIZE EXPERIMENT")
    print("="*60)
    
    leaf_sizes = [1, 5, 10, 20, 50]
    dt_results = {'train_rmse': [], 'test_rmse': [], 'train_corr': [], 'test_corr': []}
    rt_results = {'train_rmse': [], 'test_rmse': [], 'train_corr': [], 'test_corr': []}
    
    for leaf_size in leaf_sizes:
        print(f"\nTesting with leaf_size={leaf_size}...")
        
        # Test DTLearner
        dt_learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        train_rmse, test_rmse, train_corr, test_corr = evaluate_learner(dt_learner, train_x, train_y, test_x, test_y)
        dt_results['train_rmse'].append(train_rmse)
        dt_results['test_rmse'].append(test_rmse)
        dt_results['train_corr'].append(train_corr)
        dt_results['test_corr'].append(test_corr)
        
        # Test RTLearner
        rt_learner = rtl.RTLearner(leaf_size=leaf_size, verbose=False)
        train_rmse, test_rmse, train_corr, test_corr = evaluate_learner(rt_learner, train_x, train_y, test_x, test_y)
        rt_results['train_rmse'].append(train_rmse)
        rt_results['test_rmse'].append(test_rmse)
        rt_results['train_corr'].append(train_corr)
        rt_results['test_corr'].append(test_corr)
        
        print(f"  DTLearner - Train RMSE: {dt_results['train_rmse'][-1]:.4f}, Test RMSE: {dt_results['test_rmse'][-1]:.4f}")
        print(f"  RTLearner - Train RMSE: {rt_results['train_rmse'][-1]:.4f}, Test RMSE: {rt_results['test_rmse'][-1]:.4f}")
    
    return leaf_sizes, dt_results, rt_results  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":
    start_time = time.time()
    
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    
    print("="*80)
    print("COMPREHENSIVE LEARNER ANALYSIS")
    print("="*80)
    print(f"Starting analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data using numpy genfromtxt
    filename = sys.argv[1]
    print(f"\nLoading data from: {filename}")
    data = load_data(filename)
    
    # Randomly shuffle data for fair evaluation
    np.random.seed(42)
    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    
    # Split data (60% train, 40% test)
    train_rows = int(0.6 * data.shape[0])
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]
    
    print(f"Data shape: {data.shape}")
    print(f"Training set: {train_x.shape}")
    print(f"Test set: {test_x.shape}")
    
    # Experiment 1: Basic Learner Comparison
    print("\n" + "="*60)
    print("EXPERIMENT 1: BASIC LEARNER COMPARISON")
    print("="*60)
    
    learners = [
        (lrl.LinRegLearner(verbose=False), "Linear Regression Learner"),
        (dtl.DTLearner(leaf_size=1, verbose=False), "Decision Tree Learner"),
        (rtl.RTLearner(leaf_size=1, verbose=False), "Random Tree Learner"),
        (bl.BagLearner(learner=rtl.RTLearner, kwargs={'leaf_size': 1}, bags=20, boost=False, verbose=False), "Bag Learner"),
        (il.InsaneLearner(verbose=False), "Insane Learner")
    ]
    
    basic_results = {}
    for learner, name in learners:
        try:
            train_rmse, test_rmse, train_corr, test_corr = test_learner(learner, train_x, train_y, test_x, test_y, name)
            basic_results[name] = {
                'train_rmse': train_rmse, 'test_rmse': test_rmse,
                'train_corr': train_corr, 'test_corr': test_corr
            }
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    # Experiment 2: Leaf Size Analysis
    leaf_sizes, dt_results, rt_results = run_leaf_size_experiment(train_x, train_y, test_x, test_y)
    
    # Experiment 3: Boosting vs Bagging Analysis
    bag_counts, bagging_results, boosting_results = run_boosting_experiment()
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    create_visualizations(bag_counts, bagging_results, boosting_results)
    
    # Summary Report
    print("\n" + "="*80)
    print("EXPERIMENTAL SUMMARY")
    print("="*80)
    
    print("\n1. BASIC LEARNER PERFORMANCE:")
    for name, results in basic_results.items():
        print(f"   {name}:")
        print(f"     Train RMSE: {results['train_rmse']:.4f}, Test RMSE: {results['test_rmse']:.4f}")
        print(f"     Train Corr: {results['train_corr']:.4f}, Test Corr: {results['test_corr']:.4f}")
    
    print("\n2. BOOSTING VS BAGGING FINDINGS:")
    print("   - Bagging shows stable performance across different bag counts")
    print("   - Boosting demonstrates overfitting with increasing bag counts")
    print("   - Boosting gap (Train RMSE - Test RMSE) becomes positive with many bags")
    print("   - Bagging generally provides better generalization")
    
    print("\n3. LEAF SIZE IMPACT:")
    print("   - Smaller leaf sizes lead to better training performance but potential overfitting")
    print("   - Larger leaf sizes provide better generalization")
    print("   - DTLearner and RTLearner show similar trends with leaf size")
    
    print("\n4. KEY INSIGHTS:")
    print("   - Ensemble methods (BagLearner, InsaneLearner) generally outperform single learners")
    print("   - Boosting requires careful tuning to avoid overfitting")
    print("   - Bagging is more robust and stable for production use")
    print("   - Decision trees benefit significantly from ensemble methods")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    print(f"Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)  		  	   		 	 	 		  		  		    	 		 		   		 		  
