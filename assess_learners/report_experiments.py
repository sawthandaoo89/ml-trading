#!/usr/bin/env python3
"""
Report Experiments Script
Generates specific experiments and charts for the P3 report
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl

def load_istanbul_data():
    """Load Istanbul.csv data"""
    data = np.genfromtxt('Data/Istanbul.csv', delimiter=',', skip_header=1, usecols=range(1, 9))
    np.random.seed(42)
    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    
    train_rows = int(0.6 * data.shape[0])
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]
    
    return train_x, train_y, test_x, test_y

def calculate_metrics(pred_train, pred_test, y_train, y_test):
    """Calculate various metrics"""
    # RMSE
    train_rmse = np.sqrt(np.mean((y_train - pred_train) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - pred_test) ** 2))
    
    # MAE
    train_mae = np.mean(np.abs(y_train - pred_train))
    test_mae = np.mean(np.abs(y_test - pred_test))
    
    # R²
    train_r2 = 1 - np.sum((y_train - pred_train) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    test_r2 = 1 - np.sum((y_test - pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    return {
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2
    }

def calculate_tree_depth(tree):
    """Calculate average tree depth"""
    if tree is None or tree.shape[0] == 0:
        return 0
    
    def get_depth(node_idx, current_depth):
        if node_idx >= tree.shape[0]:
            return current_depth
        
        node = tree[node_idx]
        if node[0] == -1:  # Leaf node
            return current_depth
        
        left_child = int(node[2])
        right_child = int(node[3])
        
        left_depth = get_depth(node_idx + left_child, current_depth + 1)
        right_depth = get_depth(node_idx + right_child, current_depth + 1)
        
        return max(left_depth, right_depth)
    
    return get_depth(0, 0)

def experiment_1_overfitting():
    """Experiment 1: Overfitting analysis with leaf_size"""
    print("Running Experiment 1: Overfitting Analysis...")
    
    train_x, train_y, test_x, test_y = load_istanbul_data()
    leaf_sizes = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    
    results = {'leaf_sizes': [], 'train_rmse': [], 'test_rmse': [], 'overfitting_gap': []}
    
    for leaf_size in leaf_sizes:
        print(f"  Testing leaf_size={leaf_size}...")
        
        # Train DTLearner
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        start_time = time.time()
        learner.add_evidence(train_x, train_y)
        train_time = time.time() - start_time
        
        # Get predictions
        pred_train = learner.query(train_x)
        pred_test = learner.query(test_x)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_train, pred_test, train_y, test_y)
        overfitting_gap = metrics['train_rmse'] - metrics['test_rmse']
        
        results['leaf_sizes'].append(leaf_size)
        results['train_rmse'].append(metrics['train_rmse'])
        results['test_rmse'].append(metrics['test_rmse'])
        results['overfitting_gap'].append(overfitting_gap)
        
        print(f"    Train RMSE: {metrics['train_rmse']:.6f}, Test RMSE: {metrics['test_rmse']:.6f}, Gap: {overfitting_gap:.6f}")
    
    return results

def experiment_2_bagging():
    """Experiment 2: Bagging effect on overfitting"""
    print("\nRunning Experiment 2: Bagging Analysis...")
    
    train_x, train_y, test_x, test_y = load_istanbul_data()
    leaf_sizes = [1, 5, 10, 15, 20, 30, 50]
    
    results = {
        'leaf_sizes': [], 
        'single_train_rmse': [], 'single_test_rmse': [], 'single_gap': [],
        'bagged_train_rmse': [], 'bagged_test_rmse': [], 'bagged_gap': []
    }
    
    for leaf_size in leaf_sizes:
        print(f"  Testing leaf_size={leaf_size}...")
        
        # Single tree
        single_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        single_learner.add_evidence(train_x, train_y)
        single_pred_train = single_learner.query(train_x)
        single_pred_test = single_learner.query(test_x)
        single_metrics = calculate_metrics(single_pred_train, single_pred_test, train_y, test_y)
        single_gap = single_metrics['train_rmse'] - single_metrics['test_rmse']
        
        # Bagged trees
        bagged_learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size': leaf_size}, bags=20, boost=False, verbose=False)
        bagged_learner.add_evidence(train_x, train_y)
        bagged_pred_train = bagged_learner.query(train_x)
        bagged_pred_test = bagged_learner.query(test_x)
        bagged_metrics = calculate_metrics(bagged_pred_train, bagged_pred_test, train_y, test_y)
        bagged_gap = bagged_metrics['train_rmse'] - bagged_metrics['test_rmse']
        
        results['leaf_sizes'].append(leaf_size)
        results['single_train_rmse'].append(single_metrics['train_rmse'])
        results['single_test_rmse'].append(single_metrics['test_rmse'])
        results['single_gap'].append(single_gap)
        results['bagged_train_rmse'].append(bagged_metrics['train_rmse'])
        results['bagged_test_rmse'].append(bagged_metrics['test_rmse'])
        results['bagged_gap'].append(bagged_gap)
        
        print(f"    Single - Train: {single_metrics['train_rmse']:.6f}, Test: {single_metrics['test_rmse']:.6f}, Gap: {single_gap:.6f}")
        print(f"    Bagged - Train: {bagged_metrics['train_rmse']:.6f}, Test: {bagged_metrics['test_rmse']:.6f}, Gap: {bagged_gap:.6f}")
    
    return results

def experiment_3_comparison():
    """Experiment 3: DTLearner vs RTLearner comparison"""
    print("\nRunning Experiment 3: DT vs RT Comparison...")
    
    train_x, train_y, test_x, test_y = load_istanbul_data()
    leaf_sizes = [1, 5, 10, 20, 50]
    
    results = {
        'leaf_sizes': [],
        'dt_train_mae': [], 'dt_test_mae': [], 'dt_train_r2': [], 'dt_test_r2': [],
        'dt_train_time': [], 'dt_avg_depth': [],
        'rt_train_mae': [], 'rt_test_mae': [], 'rt_train_r2': [], 'rt_test_r2': [],
        'rt_train_time': [], 'rt_avg_depth': []
    }
    
    for leaf_size in leaf_sizes:
        print(f"  Testing leaf_size={leaf_size}...")
        
        # DTLearner
        dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        start_time = time.time()
        dt_learner.add_evidence(train_x, train_y)
        dt_train_time = time.time() - start_time
        
        dt_pred_train = dt_learner.query(train_x)
        dt_pred_test = dt_learner.query(test_x)
        dt_metrics = calculate_metrics(dt_pred_train, dt_pred_test, train_y, test_y)
        dt_depth = calculate_tree_depth(dt_learner.tree)
        
        # RTLearner
        rt_learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
        start_time = time.time()
        rt_learner.add_evidence(train_x, train_y)
        rt_train_time = time.time() - start_time
        
        rt_pred_train = rt_learner.query(train_x)
        rt_pred_test = rt_learner.query(test_x)
        rt_metrics = calculate_metrics(rt_pred_train, rt_pred_test, train_y, test_y)
        rt_depth = calculate_tree_depth(rt_learner.tree)
        
        results['leaf_sizes'].append(leaf_size)
        results['dt_train_mae'].append(dt_metrics['train_mae'])
        results['dt_test_mae'].append(dt_metrics['test_mae'])
        results['dt_train_r2'].append(dt_metrics['train_r2'])
        results['dt_test_r2'].append(dt_metrics['test_r2'])
        results['dt_train_time'].append(dt_train_time)
        results['dt_avg_depth'].append(dt_depth)
        
        results['rt_train_mae'].append(rt_metrics['train_mae'])
        results['rt_test_mae'].append(rt_metrics['test_mae'])
        results['rt_train_r2'].append(rt_metrics['train_r2'])
        results['rt_test_r2'].append(rt_metrics['test_r2'])
        results['rt_train_time'].append(rt_train_time)
        results['rt_avg_depth'].append(rt_depth)
        
        print(f"    DT - MAE: {dt_metrics['test_mae']:.6f}, R²: {dt_metrics['test_r2']:.3f}, Time: {dt_train_time:.3f}s, Depth: {dt_depth}")
        print(f"    RT - MAE: {rt_metrics['test_mae']:.6f}, R²: {rt_metrics['test_r2']:.3f}, Time: {rt_train_time:.3f}s, Depth: {rt_depth}")
    
    return results

def create_charts(exp1_results, exp2_results, exp3_results):
    """Create all charts for the report"""
    print("\nCreating charts...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # Experiment 1: Overfitting Analysis
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(exp1_results['leaf_sizes'], exp1_results['train_rmse'], 'b-o', label='Training RMSE', linewidth=2)
    ax1.plot(exp1_results['leaf_sizes'], exp1_results['test_rmse'], 'r-s', label='Test RMSE', linewidth=2)
    ax1.set_xlabel('Leaf Size')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Experiment 1: Overfitting vs Leaf Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Overfitting Gap
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(exp1_results['leaf_sizes'], exp1_results['overfitting_gap'], 'g-o', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Leaf Size')
    ax2.set_ylabel('Overfitting Gap (Train RMSE - Test RMSE)')
    ax2.set_title('Overfitting Gap vs Leaf Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Experiment 2: Bagging Comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(exp2_results['leaf_sizes'], exp2_results['single_gap'], 'b-o', label='Single Tree Gap', linewidth=2)
    ax3.plot(exp2_results['leaf_sizes'], exp2_results['bagged_gap'], 'r-s', label='Bagged Trees Gap', linewidth=2)
    ax3.set_xlabel('Leaf Size')
    ax3.set_ylabel('Overfitting Gap')
    ax3.set_title('Experiment 2: Bagging Effect on Overfitting')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Experiment 3: MAE Comparison
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(exp3_results['leaf_sizes'], exp3_results['dt_test_mae'], 'b-o', label='DTLearner', linewidth=2)
    ax4.plot(exp3_results['leaf_sizes'], exp3_results['rt_test_mae'], 'r-s', label='RTLearner', linewidth=2)
    ax4.set_xlabel('Leaf Size')
    ax4.set_ylabel('Test MAE')
    ax4.set_title('Experiment 3: MAE Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # R² Comparison
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(exp3_results['leaf_sizes'], exp3_results['dt_test_r2'], 'b-o', label='DTLearner', linewidth=2)
    ax5.plot(exp3_results['leaf_sizes'], exp3_results['rt_test_r2'], 'r-s', label='RTLearner', linewidth=2)
    ax5.set_xlabel('Leaf Size')
    ax5.set_ylabel('Test R²')
    ax5.set_title('R² Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Training Time Comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(exp3_results['leaf_sizes'], exp3_results['dt_train_time'], 'b-o', label='DTLearner', linewidth=2)
    ax6.plot(exp3_results['leaf_sizes'], exp3_results['rt_train_time'], 'r-s', label='RTLearner', linewidth=2)
    ax6.set_xlabel('Leaf Size')
    ax6.set_ylabel('Training Time (seconds)')
    ax6.set_title('Training Time Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('p3_report_charts.png', dpi=300, bbox_inches='tight')
    print("Charts saved as 'p3_report_charts.png'")
    plt.close()

def main():
    """Run all experiments and generate charts"""
    print("="*60)
    print("P3 REPORT EXPERIMENTS")
    print("="*60)
    
    # Run experiments
    exp1_results = experiment_1_overfitting()
    exp2_results = experiment_2_bagging()
    exp3_results = experiment_3_comparison()
    
    # Create charts
    create_charts(exp1_results, exp2_results, exp3_results)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print("\nExperiment 1 - Overfitting Analysis:")
    min_gap_idx = np.argmin(np.abs(exp1_results['overfitting_gap']))
    print(f"  Optimal leaf_size: {exp1_results['leaf_sizes'][min_gap_idx]}")
    print(f"  Min overfitting gap: {exp1_results['overfitting_gap'][min_gap_idx]:.6f}")
    
    print("\nExperiment 2 - Bagging Effect:")
    avg_single_gap = np.mean(exp2_results['single_gap'])
    avg_bagged_gap = np.mean(exp2_results['bagged_gap'])
    reduction = (avg_single_gap - avg_bagged_gap) / avg_single_gap * 100
    print(f"  Average single tree gap: {avg_single_gap:.6f}")
    print(f"  Average bagged gap: {avg_bagged_gap:.6f}")
    print(f"  Overfitting reduction: {reduction:.1f}%")
    
    print("\nExperiment 3 - DT vs RT Comparison:")
    avg_dt_mae = np.mean(exp3_results['dt_test_mae'])
    avg_rt_mae = np.mean(exp3_results['rt_test_mae'])
    avg_dt_r2 = np.mean(exp3_results['dt_test_r2'])
    avg_rt_r2 = np.mean(exp3_results['rt_test_r2'])
    avg_dt_time = np.mean(exp3_results['dt_train_time'])
    avg_rt_time = np.mean(exp3_results['rt_train_time'])
    
    print(f"  DTLearner - Avg MAE: {avg_dt_mae:.6f}, Avg R²: {avg_dt_r2:.3f}, Avg Time: {avg_dt_time:.3f}s")
    print(f"  RTLearner - Avg MAE: {avg_rt_mae:.6f}, Avg R²: {avg_rt_r2:.3f}, Avg Time: {avg_rt_time:.3f}s")
    print(f"  DTLearner is {avg_rt_time/avg_dt_time:.1f}x slower but {avg_rt_mae/avg_dt_mae:.1f}x more accurate")

if __name__ == "__main__":
    main()
