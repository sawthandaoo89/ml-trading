""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Decision Tree Learner.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
class DTLearner(object):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This is a Decision Tree Learner.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to allow at a leaf node.  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
        return "soo7"  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def study_group(self):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The study group of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return "soo7" 		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		 	 	 		  		  		    	 		  		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.tree = self.build_tree(data_x, data_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def build_tree(self, data_x, data_y):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Build a decision tree recursively using NDArray representation  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Tree structure: [feature, split_value, left_child_index, right_child_index]  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Leaf nodes: [-1, prediction_value, -1, -1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Base case: if we have leaf_size or fewer samples, create a leaf  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if data_x.shape[0] <= self.leaf_size:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.array([[-1, np.mean(data_y), -1, -1]])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # If all y values are the same, create a leaf  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if np.all(data_y == data_y[0]):  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.array([[-1, data_y[0], -1, -1]])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Find the best feature to split on (highest absolute correlation with Y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
        best_feature = self.find_best_feature(data_x, data_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Split on the median value of the best feature  		  	   		 	 	 		  		  		    	 		 		   		 		  
        split_value = np.median(data_x[:, best_feature])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Split the data  		  	   		 	 	 		  		  		    	 		 		   		 		  
        left_mask = data_x[:, best_feature] <= split_value  		  	   		 	 	 		  		  		    	 		 		   		 		  
        right_mask = ~left_mask  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # If split doesn't create meaningful separation, create a leaf  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.array([[-1, np.mean(data_y), -1, -1]])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Recursively build left and right subtrees  		  	   		 	 	 		  		  		    	 		 		   		 		  
        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])  		  	   		 	 	 		  		  		    	 		 		   		 		  
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Create the root node  		  	   		 	 	 		  		  		    	 		 		   		 		  
        root = np.array([[best_feature, split_value, 1, left_tree.shape[0] + 1]])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Combine the trees  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return np.vstack([root, left_tree, right_tree])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def find_best_feature(self, data_x, data_y):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Find the feature with the highest absolute correlation with Y  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        best_corr = -1  		  	   		 	 	 		  		  		    	 		 		   		 		  
        best_feature = 0  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        for feature in range(data_x.shape[1]):  		  	   		 	 	 		  		  		    	 		 		   		 		  
            # Calculate correlation between this feature and Y  		  	   		 	 	 		  		  		    	 		 		   		 		  
            corr = np.corrcoef(data_x[:, feature], data_y)[0, 1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
            if not np.isnan(corr):  		  	   		 	 	 		  		  		    	 		 		   		 		  
                abs_corr = abs(corr)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                if abs_corr > best_corr:  		  	   		 	 	 		  		  		    	 		 		   		 		  
                    best_corr = abs_corr  		  	   		 	 	 		  		  		    	 		 		   		 		  
                    best_feature = feature  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return best_feature  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def query(self, points):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if self.tree is None:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.zeros(points.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        predictions = np.zeros(points.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
        for i, point in enumerate(points):  		  	   		 	 	 		  		  		    	 		 		   		 		  
            predictions[i] = self.query_single_point(point, 0)  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return predictions  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def query_single_point(self, point, node_index):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Query a single point using the decision tree  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        node = self.tree[node_index]  		  	   		 	 	 		  		  		    	 		 		   		 		  
        feature = int(node[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
        split_value = node[1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
        left_child = int(node[2])  		  	   		 	 	 		  		  		    	 		 		   		 		  
        right_child = int(node[3])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # If this is a leaf node  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if feature == -1:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return split_value  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Traverse the tree  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if point[feature] <= split_value:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return self.query_single_point(point, node_index + left_child)  		  	   		 	 	 		  		  		    	 		 		   		 		  
        else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return self.query_single_point(point, node_index + right_child)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		 	 	 		  		  		    	 		 		   		 		  