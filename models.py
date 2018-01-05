import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .01
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"

        #function nodes are, multiply, add vector, relu, matrix multiply, add vector
        #variables are w1, w2, b1, b2
        #size of the input vector
        i = x.shape[1]
        #to test and modify
        h = 100

        if not self.w1:
            self.w1 = nn.Variable(i, h)
        if not self.w2:
            self.w2 = nn.Variable(h, i)
        if not self.b1:
            self.b1 = nn.Variable(h)
        if not self.b2:
            self.b2 = nn.Variable(i)

        graph = nn.Graph([self.w1, self.w2, self.b1, self.b2])

        input_nodeX = nn.Input(graph, x)
        # print x.shape

        # xm = MatrixMultiply(graph, input_x, m)
        # xm_plus_b = MatrixVectorAdd(graph, xm, b)

        multiply1 = nn.MatrixMultiply(graph, input_nodeX, self.w1)
        add1 = nn.MatrixVectorAdd(graph, multiply1, self.b1)
        relu = nn.ReLU(graph, add1)
        multiply2 = nn.MatrixMultiply(graph, relu, self.w2)
        add2 = nn.MatrixVectorAdd(graph, multiply2, self.b2)
        
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            input_nodeY = nn.Input(graph, y)
            loss_node = nn.SquareLoss(graph, add2, input_nodeY)
            graph.add(loss_node)

            return graph
    
            "*** YOUR CODE HERE ***"
        else:
            return graph.get_output(add2)
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"

class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .01
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"

        #function nodes are, multiply, add vector, relu, matrix multiply, add vector
        #variables are w1, w2, b1, b2
        #size of the input vector
        i = x.shape[1]
        #to test and modify
        h = 100

        if not self.w1:
            self.w1 = nn.Variable(i, h)
        if not self.w2:
            self.w2 = nn.Variable(h, i)
        if not self.b1:
            self.b1 = nn.Variable(h)
        if not self.b2:
            self.b2 = nn.Variable(i)

        graph = nn.Graph([self.w1, self.w2, self.b1, self.b2])

        input_nodeX = nn.Input(graph, x)
        neg_X = np.negative(x)
        neg_inputX = nn.Input(graph, neg_X)
        # print x.shape

        # xm = MatrixMultiply(graph, input_x, m)
        # xm_plus_b = MatrixVectorAdd(graph, xm, b)

        multiply1 = nn.MatrixMultiply(graph, input_nodeX, self.w1)
        add1 = nn.MatrixVectorAdd(graph, multiply1, self.b1)
        relu = nn.ReLU(graph, add1)
        multiply2 = nn.MatrixMultiply(graph, relu, self.w2)
        add2 = nn.MatrixVectorAdd(graph, multiply2, self.b2)

        #for the f(-x)
        neg_multiply1 = nn.MatrixMultiply(graph, neg_inputX, self.w1)
        neg_add1 = nn.MatrixVectorAdd(graph, neg_multiply1, self.b1)
        neg_relu = nn.ReLU(graph, neg_add1)
        neg_multiply2 = nn.MatrixMultiply(graph, neg_relu, self.w2)
        neg_add2 = nn.MatrixVectorAdd(graph, neg_multiply2, self.b2)

        ones = np.ones((1, 1))
        ones = np.negative(ones)
        neg_one = nn.Input(graph, ones)

        neg_negate = nn.MatrixMultiply(graph, neg_add2, neg_one)

        final_add = nn.Add(graph,neg_negate, add2)
        
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            input_nodeY = nn.Input(graph, y)
            loss_node = nn.SquareLoss(graph, final_add, input_nodeY)
            graph.add(loss_node)

            return graph
    
            "*** YOUR CODE HERE ***"
        else:
            # print graph.get_output(add2).shape
            return graph.get_output(final_add)
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = .3
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        i = x.shape[1]
        # j = x.shape[0]
        #to test and modify
        h = 200

        if not self.w1:
            self.w1 = nn.Variable(i, h)
        if not self.w2:
            self.w2 = nn.Variable(h, 10)
        if not self.b1:
            self.b1 = nn.Variable(h)
        if not self.b2:
            self.b2 = nn.Variable(10)

        graph = nn.Graph([self.w1, self.w2, self.b1, self.b2])

        input_nodeX = nn.Input(graph, x)
        multiply1 = nn.MatrixMultiply(graph, input_nodeX, self.w1)
        add1 = nn.MatrixVectorAdd(graph, multiply1, self.b1)
        relu = nn.ReLU(graph, add1)
        multiply2 = nn.MatrixMultiply(graph, relu, self.w2)
        add2 = nn.MatrixVectorAdd(graph, multiply2, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            input_nodeY = nn.Input(graph, y)
            loss_node = nn.SoftmaxLoss(graph, add2, input_nodeY)
            graph.add(loss_node)

            return graph
    
            "*** YOUR CODE HERE ***"
        else:
            # print graph.get_output(add2).shape
            return graph.get_output(add2)
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = .01
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        #size of the input vector
        i = states.shape[1] * 4
        #to test and modify
        h = 200

        if not self.w1:
            self.w1 = nn.Variable(self.state_size, h)
        if not self.w2:
            self.w2 = nn.Variable(h, self.num_actions)
        if not self.b1:
            self.b1 = nn.Variable(h)
        if not self.b2:
            self.b2 = nn.Variable(self.num_actions)

        graph = nn.Graph([self.w1, self.w2, self.b1, self.b2])

        inputNodeX = nn.Input(graph, states)

        mult1 = nn.MatrixMultiply(graph, inputNodeX, self.w1)
        add1 = nn.MatrixVectorAdd(graph, mult1, self.b1)
        relu = nn.ReLU(graph, add1)
        mult2 = nn.MatrixMultiply(graph, relu, self.w2)
        add2 = nn.MatrixVectorAdd(graph, mult2, self.b2)

        if Q_target is not None:
            inputNodeY = nn.Input(graph, Q_target)
            lossNode = nn.SquareLoss(graph, add2, inputNodeY)
            graph.add(lossNode)
            # print("q target is not none")
            return graph
        else:
            # return [get_action(state, .5), 
            # print("q target is none")
            return graph.get_output(add2)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .007
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.b1 = []
        self.output = []
        self.hidden_size = 0
        c = self.num_chars

        #size of the input vector
        # i = x.shape[1]
        #to test and modify
        d = 160

        if not self.w1:
            self.w1 = nn.Variable(d, c)
        if not self.w2:
            self.w2 = nn.Variable(c, c)
        if not self.w3:
            self.w3 = nn.Variable(c, d)
        if not self.b1:
            self.b1 = nn.Variable(d)
        if not self.output:
            self.output = nn.Variable(d, 5)

        graph = nn.Graph([self.w1, self.w2, self.w3, self.b1, self.output])
        h0 = np.zeros((batch_size, d), dtype = np.float)
    
        input_nodeH = nn.Input(graph, h0)
        # print x.shape
        #array of zeros

        # multiply1 = nn.MatrixMultiply(graph, input_nodeX, self.w1)
        # add1 = nn.MatrixVectorAdd(graph, multiply1, self.b1)
        # relu = nn.ReLU(graph, add1)
        # multiply2 = nn.MatrixMultiply(graph, relu, self.w2)
        # add2 = nn.MatrixVectorAdd(graph, multiply2, self.b2)
        i = 0
        while i < len(xs):
            input_nodeC = nn.Input(graph, xs[i])
            multiply1 = nn.MatrixMultiply(graph, input_nodeH, self.w1)
            multiply2 = nn.MatrixMultiply(graph, input_nodeC, self.w2)
            combine = nn.MatrixVectorAdd(graph, multiply1, multiply2)
            multiply3 = nn.MatrixMultiply(graph, combine, self.w3)
            add1 = nn.MatrixVectorAdd(graph, multiply3, self.b1)
            relu = nn.ReLU(graph, add1)
            input_nodeH = relu
            i = i + 1
        final = nn.MatrixMultiply(graph, relu, self.output)
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            input_nodeY = nn.Input(graph, y)
            loss_node = nn.SoftmaxLoss(graph, final, input_nodeY)
            graph.add(loss_node)

            return graph
    
            "*** YOUR CODE HERE ***"
        else:
            # print graph.get_output(add2).shape
            return graph.get_output(final)
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            