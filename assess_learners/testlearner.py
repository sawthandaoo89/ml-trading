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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  		  	   		 	 	 		  		  		    	 		 		   		 		  
import DTLearner as dtl  		  	   		 	 	 		  		  		    	 		 		   		 		  
import RTLearner as rtl  		  	   		 	 	 		  		  		    	 		 		   		 		  
import BagLearner as bl  		  	   		 	 	 		  		  		    	 		 		   		 		  
import InsaneLearner as il  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def test_learner(learner, train_x, train_y, test_x, test_y, learner_name):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """Test a specific learner and print results"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"\n=== Testing {learner_name} ===")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Author: {learner.author()}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Train the learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
    learner.add_evidence(train_x, train_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Evaluate in sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		 	 	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"In sample results")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Evaluate out of sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions  		  	   		 	 	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Out of sample results")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])  		  	   		 	 	 		  		  		    	 		 		   		 		  
    data = np.array(  		  	   		 	 	 		  		  		    	 		 		   		 		  
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]  # Skip header row and date column  		  	   		 	 	 		  		  		    	 		 		   		 		  
    )  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Data shape: {data.shape}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Training set: {train_x.shape}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Test set: {test_x.shape}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Test all learners  		  	   		 	 	 		  		  		    	 		 		   		 		  
    learners = [  		  	   		 	 	 		  		  		    	 		 		   		 		  
        (lrl.LinRegLearner(verbose=True), "Linear Regression Learner"),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        (dtl.DTLearner(leaf_size=1, verbose=True), "Decision Tree Learner"),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        (rtl.RTLearner(leaf_size=1, verbose=True), "Random Tree Learner"),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        (bl.BagLearner(learner=rtl.RTLearner, kwargs={'leaf_size': 1}, bags=20, verbose=True), "Bag Learner"),  		  	   		 	 	 		  		  		    	 		 		   		 		  
        (il.InsaneLearner(verbose=True), "Insane Learner")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    ]  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    for learner, name in learners:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        try:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            test_learner(learner, train_x, train_y, test_x, test_y, name)  		  	   		 	 	 		  		  		    	 		 		   		 		  
        except Exception as e:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            print(f"Error testing {name}: {e}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
