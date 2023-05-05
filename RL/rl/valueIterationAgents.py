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
import sys

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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        iterations = self.iterations
        while iterations:
            current_value = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    max_value = -sys.maxsize
                    for action in self.mdp.getPossibleActions(state):
                        action_value = 0
                        for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                            reward = self.mdp.getReward(state, action, next_state)
                            action_value += probability * (reward + self.discount * self.values[next_state])
                        if action_value > max_value:
                            max_value = action_value
                    current_value[state] = max_value
            self.values = current_value
            iterations -= 1

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

        action_value = 0
        for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            action_value += probability * (reward + self.discount * self.values[next_state])
        return action_value

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """

        if self.mdp.isTerminal(state):
            return None
        else:
            max_value = -sys.maxsize
            best_action = self.mdp.getPossibleActions(state)[0]
            for action in self.mdp.getPossibleActions(state):
                Q_value = self.computeQValueFromValues(state, action)
                if Q_value > max_value:
                    max_value = Q_value
                    best_action = action
            return best_action

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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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
        iterations = self.iterations
        states = self.mdp.getStates()
        for iteration in range(iterations):
            state_index = iteration % len(states)
            state = states[state_index]
            if not self.mdp.isTerminal(state):
                max_value = -sys.maxsize
                for action in self.mdp.getPossibleActions(state):
                    action_value = 0
                    for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, next_state)
                        action_value += probability * (reward + self.discount * self.values[next_state])
                    if action_value > max_value:
                        max_value = action_value
                self.values[state] = max_value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = dict()
        states = self.mdp.getStates()

        for state in states:
            predecessors[state] = set()

        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                    if probability != 0:
                        predecessors[next_state].add(state)

        queue = util.PriorityQueue()

        for state in states:
            if not self.mdp.isTerminal(state):
                Q_values = list()
                max_Q_value = -sys.maxsize
                current = self.values[state]
                for action in self.mdp.getPossibleActions(state):
                    Q_value = self.computeQValueFromValues(state, action)
                    Q_values.append(Q_value)
                    if Q_value > max_Q_value:
                        max_Q_value = Q_value

                if current > max_Q_value:
                    queue.update(state, max_Q_value - current)
                else:
                    queue.update(state, current - max_Q_value)

        for i in range(self.iterations):
            if queue.isEmpty():
                return

            state = queue.pop()
            if not self.mdp.isTerminal(state):
                values = list()
                max_value = -sys.maxsize
                for action in self.mdp.getPossibleActions(state):
                    value = 0
                    for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, next_state)
                        next_Q_value = self.values[next_state]
                        value += probability * (reward + self.discount * next_Q_value)
                    values.append(value)
                    if value > max_value:
                        max_value = value
                self.values[state] = max_value

            for previous in predecessors[state]:
                Q_values = list()
                max_Q_value = -sys.maxsize
                current = self.values[previous]
                for action in self.mdp.getPossibleActions(previous):
                    Q_value = self.computeQValueFromValues(previous, action)
                    Q_values.append(Q_value)
                    if Q_value > max_Q_value:
                        max_Q_value = Q_value

                if abs(current - max_Q_value) > self.theta:
                    queue.update(previous, -abs(current - max_Q_value))
