import abc 

class AlgoBase(abc.ABC):
    """Abstract class where is defined the basic behavior of a prediction algorithm.
    Keyword Args: 
    """
    def __init__(self, n_arms, **kwargs):
        self.n_arms = n_arms
    
    @abc.abstractmethod
    def train(self, *args):
        """Update model on a given training set.
        To be defined in children classes.
        """
        pass 

    @abc.abstractmethod
    def predict(self, *args):
        """Predict rewards/logits for each arm
        To be defined in children classes.
        """
        pass 

    @abc.abstractmethod
    def reset(self):
        """Initialize variables of policy
        To be defined in children classes.
        """
        pass 

    @abc.abstractmethod
    def default_prediction(self):
        """Might be Used when user features not ready. Due to the following reasons
            - 1. Large delay to get the features from database
            - 2. Users features doesn't exist in the database
        By default, return the global mean of all ratings (can be overridden in
        child classes).
        Returns:
            (float): The mean of all ratings in the trainset.
        """
        pass 