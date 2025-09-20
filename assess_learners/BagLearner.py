""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Bag Learner.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
class BagLearner(object):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This is a Bag Learner.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param learner: The learning algorithm to use for bagging.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type learner: class  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param kwargs: Keyword arguments to be passed on to the learner's constructor.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type kwargs: dict  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param bags: Number of bags to use in the bagging process.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type bags: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param boost: If True, use boosting instead of bagging.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type boost: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param verbose: If "verbose" is True, your code can print out information for debugging.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.learner = learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.kwargs = kwargs  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.bags = bags  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.boost = boost  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.learners = []  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.learner_weights = []  # For boosting weights  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        return "soo7"  # replace with your Georgia Tech username  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
        # Clear any existing learners and weights  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.learners = []  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.learner_weights = []  		  	   		 	 	 		  		  		    	 		 		   		 		  
        n_samples = data_x.shape[0]  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if self.boost:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            # AdaBoost implementation  		  	   		 	 	 		  		  		    	 		 		   		 		  
            sample_weights = np.ones(n_samples) / n_samples  # Initialize weights  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
            for i in range(self.bags):  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Create weighted bootstrap sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                bag_x = data_x[bootstrap_indices]  		  	   		 	 	 		  		  		    	 		 		   		 		  
                bag_y = data_y[bootstrap_indices]  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Create and train learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
                learner_instance = self.learner(**self.kwargs)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                learner_instance.add_evidence(bag_x, bag_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Get predictions on original data  		  	   		 	 	 		  		  		    	 		 		   		 		  
                predictions = learner_instance.query(data_x)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Calculate weighted error  		  	   		 	 	 		  		  		    	 		 		   		 		  
                errors = np.abs(predictions - data_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                weighted_error = np.sum(sample_weights * errors) / np.sum(sample_weights)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Calculate learner weight (alpha) with bounds checking  		  	   		 	 	 		  		  		    	 		 		   		 		  
                if weighted_error > 0 and weighted_error < 0.5:  		  	   		 	 	 		  		  		    	 		 		   		 		  
                    alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                elif weighted_error >= 0.5:  		  	   		 	 	 		  		  		    	 		 		   		 		  
                    alpha = 0.1  # Small weight for poor learners  		  	   		 	 	 		  		  		    	 		 		   		 		  
                else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
                    alpha = 1.0  # Perfect learner  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Update sample weights with numerical stability  		  	   		 	 	 		  		  		    	 		 		   		 		  
                margin = 2 * (predictions > data_y) - 1  # +1 if prediction > actual, -1 otherwise  		  	   		 	 	 		  		  		    	 		 		   		 		  
                sample_weights *= np.exp(-alpha * margin)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                sample_weights = np.nan_to_num(sample_weights, nan=1.0/n_samples)  # Handle NaN values  		  	   		 	 	 		  		  		    	 		 		   		 		  
                sample_weights /= np.sum(sample_weights)  # Normalize  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Store learner and weight  		  	   		 	 	 		  		  		    	 		 		   		 		  
                self.learners.append(learner_instance)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                self.learner_weights.append(alpha)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            # Standard bagging implementation  		  	   		 	 	 		  		  		    	 		 		   		 		  
            for i in range(self.bags):  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Create bootstrap sample (sample with replacement)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                bag_x = data_x[bootstrap_indices]  		  	   		 	 	 		  		  		    	 		 		   		 		  
                bag_y = data_y[bootstrap_indices]  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Create a new learner instance with the provided kwargs  		  	   		 	 	 		  		  		    	 		 		   		 		  
                learner_instance = self.learner(**self.kwargs)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Train the learner on the bootstrap sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
                learner_instance.add_evidence(bag_x, bag_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
                # Add the trained learner to our ensemble  		  	   		 	 	 		  		  		    	 		 		   		 		  
                self.learners.append(learner_instance)  		  	   		 	 	 		  		  		    	 		 		   		 		  
                self.learner_weights.append(1.0)  # Equal weights for bagging  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def query(self, points):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 	 	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if not self.learners:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.zeros(points.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        # Get predictions from all learners in the ensemble  		  	   		 	 	 		  		  		    	 		 		   		 		  
        predictions = np.zeros((len(self.learners), points.shape[0]))  		  	   		 	 	 		  		  		    	 		 		   		 		  
        for i, learner in enumerate(self.learners):  		  	   		 	 	 		  		  		    	 		 		   		 		  
            predictions[i] = learner.query(points)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
        if self.boost:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            # Weighted average for boosting  		  	   		 	 	 		  		  		    	 		 		   		 		  
            weights = np.array(self.learner_weights)  		  	   		 	 	 		  		  		    	 		 		   		 		  
            weights = weights / np.sum(weights)  # Normalize weights  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.average(predictions, axis=0, weights=weights)  		  	   		 	 	 		  		  		    	 		 		   		 		  
        else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
            # Average the predictions from all learners (bootstrap aggregation)  		  	   		 	 	 		  		  		    	 		 		   		 		  
            return np.mean(predictions, axis=0)  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		 	 	 		  		  		    	 		 		   		 		  