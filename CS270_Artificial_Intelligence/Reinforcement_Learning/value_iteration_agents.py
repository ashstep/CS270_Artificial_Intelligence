'''
Implemented algorithms for use with Markov Decision Processes (MDPs) in reinforcement learning.
These algorithms will be applied in a version of Sutton and Barto’s ”cliff walking” domain.
In this class a value iteration agent is implemented.
Run this with Run1.py.
@author Ashka Stephen
DATE: April 15th
'''
from learning_agents import ValueEstimationAgent
from mdp import MarkovDecisionProcess

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.1, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probabilities(state, action)
              mdp.get_reward(state, action, nextState)

              4 for loops  in constructor 
        num of iterations , all states, maximum, sum

        """

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = dict((s, 0.0) for s in mdp.get_states()) #u
        print "self.values", self.values

        for iteration in range(0, self.iterations):
          delta = 0
          newdict = self.values.copy()
          for state in mdp.get_states():
            for eachAction in mdp.get_possible_actions(state):
              print " mdp.get_reward(state) ",  mdp.get_reward(state) 
              newdict[state] = mdp.get_reward(state) + (discount * max([sum([ prob * self.values[newState] for (newState, prob) in mdp.get_transition_states_and_probabilities(state, eachAction)]) for eachAction in mdp.get_possible_actions(state)]))
              

            #delta = max(delta, newdict[state] - self.values[state])
            #if delta < epsilon * (1 - discount) / discount:
            #  return self.values

              """

                      for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])



              print "newState " , newState
              print "prob ", prob
              print "self.values[newState] ", self.values[newState]
              print "mdp.get_reward(state) ", mdp.get_reward(state)
              print "prob ", prob
        """

        self.values = newdict
        print "print self.values", self.values


    def get_value(self, state):
        """
           * *\Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def get_q_value(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.

          get_q_value not implemented
        """
        #*** YOUR CODE HERE ***
        raise NotImplementedError()

    def get_policy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          TODO implement this 
          *How the game picks your move at a given state

        """
 


        
        maxVal = float('-inf')
        bestItem = self.mdp.get_possible_actions(state)[0]
        currDir = None
        if len(self.mdp.get_possible_actions(state)) == 0 or self.mdp.is_terminal(state):
          return None

        for eachDir in self.mdp.get_possible_actions(state):
          for (newState, prob) in self.mdp.get_transition_states_and_probabilities(state, eachDir):
            print "current direction: ", eachDir, " with val ", self.values[newState]
            #loop thru new states
            if self.values[newState] >= maxVal:
              bestItem = newState
              maxVal = self.values[newState] 
              currDir = eachDir
        return currDir
        



        #*** YOUR CODE HERE ***
        raise NotImplementedError()

    def get_action(self, state):
        "Returns the policy at the state (no exploration)."
        return self.get_policy(state)

