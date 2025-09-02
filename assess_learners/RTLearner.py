""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""MC3-P1: Random Tree Learner.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
class RTLearner(object):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This is a Random Tree Learner. It is implemented correctly.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to leave at a leaf node  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param verbose: If "verbose" is True, your code can print out information for debugging.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.leaf_size = leaf_size  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.tree = None  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return "soo7"  # replace soo7 with your Georgia Tech username  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Build the random tree  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.tree = self._build_tree(data_x, data_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def _build_tree(self, data_x, data_y):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """Build the random tree recursively"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # If we have fewer samples than leaf_size, create a leaf  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if data_x.shape[0] <= self.leaf_size:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.array([[0, np.mean(data_y), np.nan, np.nan]])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # If all y values are the same, create a leaf  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if np.all(data_y == data_y[0]):  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.array([[0, data_y[0], np.nan, np.nan]])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Randomly select a feature to split on  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if data_x.shape[1] == 1:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            best_feature = 0  		  	   		 	 	 		  		  		    	 		 		   		 		  
        else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            best_feature = np.random.randint(0, data_x.shape[1])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Get the split value (median of the feature)  		  	   		 	 	 		  		  		    	 		 		   		 		  
        split_val = np.median(data_x[:, best_feature])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Split the data  		  	   		 	 	 		  		  		    	 		 		   		 		  
        left_mask = data_x[:, best_feature] <= split_val  		  	   		 	 	 		  		  		    	 		 		   		 		  
        right_mask = ~left_mask  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # If split doesn't separate data, create a leaf  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.array([[0, np.mean(data_y), np.nan, np.nan]])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Build left and right subtrees  		  	   		 	 	 		  		  		    	 		 		   		 		  
        left_tree = self._build_tree(data_x[left_mask], data_y[left_mask])  		  	   		 	 	 		  		  		    	 		 		   		 		  
        right_tree = self._build_tree(data_x[right_mask], data_y[right_mask])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Create the current node  		  	   		 	 	 		  		  		    	 		 		   		 		  
        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Combine the trees  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return np.vstack([root, left_tree, right_tree])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def query(self, points):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if self.tree is None:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            raise ValueError("Model not trained yet. Call add_evidence first.")  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        predictions = np.zeros(points.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
        for i, point in enumerate(points):  		  	   		 	 	 		  		  		    	 		 		   		 		  
            predictions[i] = self._query_single(point)  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return predictions  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def _query_single(self, point):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """Query a single point by traversing the tree"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
        node_idx = 0  		  	   		 	 	 		  		  		    	 		 		   		 		  
        while True:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            node = self.tree[node_idx]  		  	   		 	 	 		  		  		    	 		 		   		 		  
            feature = int(node[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
            split_val = node[1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
            left_idx = int(node[2])  		  	   		 	 	 		  		  		    	 		 		   		 		  
            right_idx = int(node[3])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
            # If this is a leaf node (feature == 0)  		  	   		 	 	 		  		  		    	 		 		   		 		  
            if feature == 0:  		  	   		 	 	 		  		  		    	 		 		   		 		  
                return split_val  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
            # Traverse left or right based on split value  		  	   		 	 	 		  		  		    	 		 		   		 		  
            if point[feature] <= split_val:  		  	   		 	 	 		  		  		    	 		 		   		 		  
                node_idx = left_idx  		  	   		 	 	 		  		  		    	 		 		   		 		  
            else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
                node_idx = right_idx  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'") 