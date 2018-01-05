# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
          tempvalues = util.Counter()
          for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
              tempvalues[state] = 0
            else:
              maxscore = -100000
              for action in self.mdp.getPossibleActions(state):
                score = 0
                for Tstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                  score += prob * (self.mdp.getReward(state, action, Tstate) + (self.discount * self.values[Tstate]))
                maxscore = max(score, maxscore)
                tempvalues[state] = maxscore
          self.values = tempvalues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        score = 0
        for Tstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          score += prob * (self.mdp.getReward(state, action, Tstate) + (self.discount * self.values[Tstate]))
        return score

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        maxscore = -100000
        bestaction = None
        
        for action in self.mdp.getPossibleActions(state):
          Qscore = self.computeQValueFromValues(state, action)
          if Qscore > maxscore:
            maxscore = Qscore
            bestaction = action
        return bestaction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        iteration = 0
        while iteration < self.iterations:
          for state in self.mdp.getStates():
            values = util.Counter()
            for action in self.mdp.getPossibleActions(state):
              values[action] = self.computeQValueFromValues(state, action)
            index = values.argMax()
            self.values[state] = values[index]
            iteration += 1
            if iteration >= self.iterations:
              return

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        preds = {}

        for state in self.mdp.getStates():
          preds[state] = set()

        for state in self.mdp.getStates():
          for action in self.mdp.getPossibleActions(state):
            for Tstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
              preds[Tstate].add(state)

        fringe = util.PriorityQueue()

        for s in self.mdp.getStates():
          values = util.Counter()
          for action in self.mdp.getPossibleActions(s):
            values[action] = self.computeQValueFromValues(s, action)
          diff = values[values.argMax()] - self.values[s]
          if diff < 0:
            diff *= -1
          fringe.update(s, -diff)

        for i in range(self.iterations):
          if fringe.isEmpty():
            return
          state = fringe.pop()
          if not self.mdp.isTerminal(state):
            values = util.Counter()
            for action in self.mdp.getPossibleActions(state):
              values[action] = self.computeQValueFromValues(state, action)
            self.values[state] = values[values.argMax()]

          for pred in preds[state]:
            values = util.Counter()
            for action in self.mdp.getPossibleActions(pred):
              values[action] = self.computeQValueFromValues(pred, action)
            diff = values[values.argMax()] - self.values[pred]
            if diff < 0:
              diff *= -1
            if diff > self.theta:
              fringe.update(pred, -diff)


