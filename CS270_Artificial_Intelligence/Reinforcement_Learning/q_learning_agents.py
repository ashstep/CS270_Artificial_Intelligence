'''
In this class a Q-learning agen is implemented.
Run this with Run2.py.
@author Ashka Stephen
DATE: April 15th
'''
from learning_agents import ReinforcementAgent


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - get_q_value
      - get_action
      - get_value
      - get_policy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state

        
    """

    def __init__(self, **args):
        """You can initialize Q-values here..."""
        super(QLearningAgent, self).__init__(**args)

        "*** YOUR CODE HERE ***"

    def get_q_value(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we never seen
        a state or (state,action) tuple
        """
        #"*** YOUR CODE HERE ***"
        if(self.getLegalActions(state) is None):
            print "get_q_value: state never seen "
            return 0.0
        else:
            print "get_q_value: state seen and returned"
            return (state, self.getLegalActions(state))
        #raise NotImplementedError()

    def get_value(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        #"*** YOUR CODE HERE ***"
        #raise NotImplementedError()
        ret = {}
        if (self.getLegalActions(state) is None):
            print "get_value: this is the terminal state"
            return 0.0
        else:
            for each in self.getLegalActions(state):
                print "get_value: each in self.getLegalActions(state) is:  ", each
                #ret[each] = number
                ### NOT COMPLETED


    def get_policy(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        "*** YOUR CODE HERE ***"
        raise NotImplementedError()

    def get_action(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
  
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.get_legal_actions(state)

        action = None
        "*** YOUR CODE HERE ***"
        raise NotImplementedError()

        return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here
  
        NOTE: You should never call this function,
        it will be called on your behalf


        most of the code will go here
        initial state -> get action -> update (prev, action, new state, reward)
        """
        "*** YOUR CODE HERE ***"
        raise NotImplementedError()
