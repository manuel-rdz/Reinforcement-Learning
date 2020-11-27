# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 200):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    self.q_values = {}
    self.best_action = {}

    # calculate utilities values
    for i in range(self.iterations):
      next_values = util.Counter()
      for s in mdp.getStates():
        updated = False
        for a in mdp.getPossibleActions(s):
          action_value = 0.0

          for t in mdp.getTransitionStatesAndProbs(s, a):
            r = mdp.getReward(s, a, t[0])
            action_value += t[1] * (r + discount * self.values[t[0]])

          if not updated or action_value > next_values[s]:
            next_values[s] = action_value
            updated = True
      self.values = next_values

    # with the given utilities, calculate q-values
    p = False
    for s in mdp.getStates():
      self.best_action[s] = None
      max_action_value = -10000000
      for a in mdp.getPossibleActions(s):
        action_value = 0.0
        for t in mdp.getTransitionStatesAndProbs(s, a):
          r = mdp.getReward(s, a, t[0])
          action_value += t[1] * (r + discount * self.values[t[0]])
        self.q_values[(s, a)] = action_value
        if action_value > max_action_value:
          max_action_value = action_value
          self.best_action[s] = a
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    return self.q_values[(state, action)]

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    return self.best_action[state]

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
