#!/usr/bin/env python

"""
nn.py

Implements feedforward complete artifical neural networks.

Python 2.7.12
A 'from scratch' implementation + a few code snippets from grad school coursework.
"""

import math
import random


def sigmoid(dblX):
    """The sigmoid function.  Given input dblX, sigmoid(dblX).
    sigmoid(0.0)
    0.5
    sigmoid(100.0)
    1.0
    sigmoid(-100.0) < 1.0e-10
    True
    """
    dblActVal = float(1.0/(1.0 + pow(math.e, -dblX)))
    return dblActVal

class Perceptron(object):
    """Implements a unit in a feed-forward neural network."""
    def __init__(self, listDblW, dblW0, ix):
        """
        listDblW: a list of weights
        dblW0: the threshold weight
        ix: the index of this perceptron in it's layer
        """
        self.listDblW = map(float,listDblW)
        self.dblW0 = float(dblW0)
        self.ix = int(ix)
    def __repr__(self):
        tplSFormat = (list(self.listDblW), self.dblW0, self.ix)
        return "Perceptron(%r, %r, %r)" % tplSFormat
    def input_size(self):
        return len(self.listDblW)

class NeuralNetLayer(object):
    """A single layer of a complete neural network"""
    def __init__(self, cInputs, listPcpt):
        """
        cInputs: number of inputs each perceptron in the layer receives.
                 same number in a complete network.
        listPcpt: list of perceptrons in the layer. The index of each perceptron
                  must match its zero-indexed position in this list. 
        """
        self.check_consistency(cInputs, listPcpt)
        self.cInputs = int(cInputs)
        self.listPcpt = listPcpt
    def layer_input_size(self):
        """Returns the number of inputs connected to each unit in this layer."""
        return self.cInputs
    def layer_output_size(self):
        """Returns the number of units in this layer."""
        return len(self.listPcpt)
    @classmethod
    def check_consistency(cls, cInputs, listPcpt):
        for ix,pcpt in enumerate(listPcpt):
            if not isinstance(pcpt, Perceptron):
                raise TypeError("Expected Perceptron")
            if pcpt.input_size() != cInputs:
                raise TypeError("Input size mismatch")
            if pcpt.ix != ix:
                raise ValueError("Index mismatch. Expected %d but found %d"
                                 % (ix, pcpt.ix))

def dot(listDbl1, listDbl2):
    """Takes the dot product of two equal-length lists of floats.
    dot([1.0, 2.0, 3.0], [-1.0, 0.25, 4.0])
    11.5"""
    if len(listDbl1) != len(listDbl2):
        print listDbl1
        print listDbl2
        raise ValueError("Incompatible lengths")
    return sum([dbl1*dbl2 for dbl1,dbl2 in zip(listDbl1,listDbl2)])

def output_error(dblActivation,dblTarget):
    """Computes the output error for perceptron activation level
    dblActivation and target value dblTarget.
    output_error(0.75, -1.0) 
    -1.75"""
    return float(dblTarget - dblActivation)

def hidden_error(listDblDownstreamDelta, pcpt, layerNext):
    """Determines the error on a hidden node from downstream deltas and weights.
    pcpt = Perceptron([], 0.0, 0)
    listPcpt = [Perceptron([1.5],0,0), Perceptron([2.0],0,1)]
    layer = NeuralNetLayer(1, listPcpt)
    hidden_error([1.0, 0.75], pcpt, layer)
    3.0"""
    curIx = pcpt.ix
    listNextWeights = map(lambda p: p.listDblW[curIx], layerNext.listPcpt)
    return dot(listDblDownstreamDelta, listNextWeights)

def compute_delta(dblActivation, dblError):
    """Computes a delta value from activation and error.
    compute_delta(0.5,0.5)
    0.125"""
    dblActivationDerv = float(dblActivation*(float(1.0) - dblActivation))
    return float(dblActivationDerv*dblError)

def update_weight(dblW, dblLearningRate, dblInput, dblDelta):
    """Computes the updated weight.
    update_weight(3.0, 0.1, 1.25, 2.0)
    3.25"""
    dblNewW = float(dblW + (dblLearningRate*dblInput*dblDelta))
    return dblNewW

def update_pcpt(pcpt, listDblInputs, dblDelta, dblLearningRate):
    """Updates the perceptron's weights in place, including threshold weights.
    pcpt = Perceptron([1.0,2.0,3.0], 4.0, 0)
    print pcpt
    Perceptron([1.0, 2.0, 3.0], 4.0, 0)
    update_pcpt(pcpt, [0.5,0.5,0.5], 0.25, 2.0)
    print pcpt
    Perceptron([1.25, 2.25, 3.25], 4.5, 0)"""
    pcpt.listDblW =\
        map(lambda dblInput, dblOldW:
            update_weight(dblOldW, dblLearningRate, dblInput, dblDelta),
            listDblInputs, pcpt.listDblW)
    dblOldW0 = pcpt.dblW0
    pcpt.dblW0 = update_weight(dblOldW0, dblLearningRate, 1.0, dblDelta)
    return None
 
def pcpt_activation(pcpt, listDblInput):
    """Compute a perceptron's activation function.
    pcpt = Perceptron([0.5,0.5,-1.5], 0.75, 0)
    pcpt_activation(pcpt, [0.5,1.0,1.0])
    0.5"""
    dblNoThreshDotInput = dot(listDblInput, pcpt.listDblW)
    dblDotInput = dblNoThreshDotInput + pcpt.dblW0
    return sigmoid(dblDotInput)

def feed_forward_layer(layer, listDblInput):
    """Build a list of activation levels for the perceptrons in the layer 
    recieving input listDblInput.
    pcpt1 = Perceptron([-1.0,2.0], 0.0, 0)
    pcpt2 = Perceptron([-2.0,4.0], 0.0, 1)
    layer = NeuralNetLayer(2, [pcpt1, pcpt2])
    listDblInput = [0.5, 0.25]
    feed_forward_layer(layer, listDblInput)
    [0.5, 0.5]"""
    return map(lambda pcpt: pcpt_activation(pcpt, listDblInput), layer.listPcpt)

class NeuralNet(object):
    """An artificial neural network."""
    def __init__(self, cInputs, listLayer):
        """Assemble the network from layers listLayer. cInputs specifies
        the number of inputs to each node in the first hidden layer of the
        network."""
        if not self.check_layers(cInputs, listLayer):
            raise TypeError("Incompatible neural network layers.")
        self.cInputs = cInputs
        for layer in listLayer:
            if not isinstance(layer, NeuralNetLayer):
                raise TypeError("NeuralNet layers must be of type "
                                "NeuralNetLayer.")
        self.listLayer = listLayer
    @classmethod
    def check_layers(cls, cInputs, listLayer):
        if not listLayer:
            return False
        if cInputs != listLayer[0].layer_input_size():
            return False
        for layerFst,layerSnd in zip(listLayer[:-1], listLayer[1:]):
            if layerFst.layer_output_size() != layerSnd.layer_input_size():
                return False
        return True
    def input_layer(self):
        return self.listLayer[0]
    def output_layer(self):
        return self.listLayer[-1]

def build_layer_inputs_and_outputs(net, listDblInput):
    """Build a pair of lists containing first the list of the input
    to each layer in the neural network, and second, a list of output
    from each layer in the network.

    The list of inputs contains as its first element listDblInput,
    the inputs to the first layer of the network.The list of outputs contains
    as its last element the output of the output layer.
    listCLayerSize = [2,2,1]
    net = init_net(listCLayerSize)
    build_layer_inputs_and_outputs(net, [-1.0, 1.0]) 
    ([[...], [...]], [[...], [...]])"""
    lstDblInputs = []
    lstDblOutputs = []
    curListDblInput = listDblInput
    curListDblOutput = []
    for layer in net.listLayer:
        curListDblOutput = feed_forward_layer(layer, curListDblInput)
        lstDblInputs.append(curListDblInput)
        lstDblOutputs.append(curListDblOutput)
        curListDblInput = curListDblOutput
    return (lstDblInputs,lstDblOutputs) 
              
def feed_forward(net, listDblInput):
    """Compute the neural net's output on input listDblInput."""
    return build_layer_inputs_and_outputs(net, listDblInput)[-1][-1]

def layer_deltas(listDblActivation, listDblError):
    """Compute the delta values for a layer which generated activation levels
    listDblActivation, resulting in error listDblError.

    layer_deltas([0.5, 0.25], [0.125, 0.0625])
    [0.03125, 0.01171875]"""
    return map(compute_delta, listDblActivation,listDblError) 

def update_layer(layer, listDblInputs, listDblDelta,  dblLearningRate):
    """Update all perceptrons in the neural net layer.

    The function updates the perceptrons in the layer in place, and does
    not return anything.

    listPcpt = [Perceptron([1.0,-1.0],0.0,0), Perceptron([-1.0,1.0],0.0,1)]
    layer = NeuralNetLayer(2, listPcpt)
    print layer.listPcpt
    [Perceptron([1.0, -1.0], 0.0, 0), Perceptron([-1.0, 1.0], 0.0, 1)]
    update_layer(layer, [0.5,-0.5], [2.0,2.0], 0.5) # do the update
    print layer.listPcpt
    [Perceptron([1.5, -1.5], 1.0, 0), Perceptron([-0.5, 0.5], 1.0, 1)]"""
    for pcpt, delta in zip(layer.listPcpt,listDblDelta):
        update_pcpt(pcpt, listDblInputs, delta, dblLearningRate)
    return None
            
def hidden_layer_error(layer, listDblDownstreamDelta, layerDownstream):
    """Determine the error produced by each node in a hidden layer, given the
    next layer downstream and the deltas produced by that layer.
    layer = NeuralNetLayer(0, [Perceptron([], 0.0, 0),
                               Perceptron([], 0.0, 1)])
    layerDownstream = NeuralNetLayer(2, [Perceptron([0.75,0.25], 0.0, 0)])
    hidden_layer_error(layer, [2.0], layerDownstream)
    [1.5, 0.5]"""
    return map(lambda pcpt: 
               hidden_error(listDblDownstreamDelta, pcpt, layerDownstream),
                    layer.listPcpt)
         
class Instance(object):
    def __init__(self, iLabel, listDblFeatures):
        self.iLabel = iLabel
        self.listDblFeatures = listDblFeatures

class ImageInstance(Instance):
    """Implements an instance composed of 2D data."""
    def __init__(self, iLabel, listListImage):
        listDblFeatures = []
        self.cRow = len(listListImage)
        self.cCol = None
        for listDblRow in listListImage:
            self.cCol = max(len(listDblRow), self.cCol)
            for dblCol in listDblRow:
                listDblFeatures.append(dblCol)
        super(ImageInstance,self).__init__(iLabel,listDblFeatures)
    def reconstruct_image(self):
        pass

def distributed_encode_label(iLabel):
    """Generate a distributed encoding of the integer label iLabel.
    distributed_encode_label(2)
    [0.05, 0.05, 0.95, 0.05, 0.05, 0.05, 0.05]"""
    listDblEncodedVector = []
    #numberDimensions = 2 # XOR test
    numberDimensions = 7 #covertype set
    dblNegVal = 0.05
    dblPosVal = 0.95
    for ix in range(numberDimensions):
        if ix is iLabel:
            listDblEncodedVector.append(dblPosVal)
        else:
            listDblEncodedVector.append(dblNegVal)
    return listDblEncodedVector
                        
def binary_encode_label(iLabel):
    """Generate a binary encoding of the integer label iLabel.
    binary_encode_label(5)
    [0.95, 0.05, 0.95, 0.05]"""
    curIntLabel = iLabel
    lstDblBinary = []
    value = None
    numberBits = 1
    dblNegVal = 0.05
    dblPosVal = 0.95
    while not(curIntLabel is 0):
         if curIntLabel%2 is 0:
             value = dblNegVal
         else: value = dblPosVal
         
         lstDblBinary.append(value)
         curIntLabel = (curIntLabel - curIntLabel%2)/2
    curBitLength = len(lstDblBinary)
    for ix in range(numberBits - curBitLength):
        lstDblBinary.append(dblNegVal)
    return lstDblBinary
                 
def distributed_decode_net_output(listDblOutput):
    """Decode the output of a neural network with distributed-encoded outputs.
    listDblOutput = [0.23, 0.4, 0.01, 0.2, 0.3, 0.78, 0.51, 0.15, 0.2, 0.1]
    distributed_decode_net_output(listDblOutput)
    5"""
    return listDblOutput.index(max(listDblOutput))

def binary_decode_net_output(listDblOutput):
    """Decode the output of a neural network with binary-encoded outputs.
    binary_decode_net_output([0.95, 0.44, 0.01, 0.51])
    9
    """
    dblThreshold = 0.50
    listInterpretation = []
    iCurBitSignificance = 1
    def convert_to_binary(dblVal):
        if dblVal < dblThreshold:
            return int(0)
        else:
            return int(1)   
    listBinaryValues = map(convert_to_binary, listDblOutput)
    for ix in range(len(listBinaryValues)):
        listInterpretation.append(iCurBitSignificance)
        iCurBitSignificance = iCurBitSignificance*2
    iOutput = dot(listBinaryValues,listInterpretation)
    return  iOutput
        
def update_net(net, inst, dblLearningRate, listTargetOutputs):
    """Updates the weights of a neural network in place based on one instance and
    returns the list of outputs after feeding forward.
    The outline is as follows:
    - feed forward the instance through the network
    - compute deltas 
    - update weights
    """
    listDblInputs = inst.listDblFeatures
    dblLayerInputsOutputs = build_layer_inputs_and_outputs(net, listDblInputs)
    listDblLayerInputs = dblLayerInputsOutputs[0]
    listDblLayerOutputs = dblLayerInputsOutputs[1]
    listDblOuputActivations = dblLayerInputsOutputs[-1][-1]
    listDblLayerDeltas = []

    #backpropagation
    #deal with output first
    listOutputError = \
        map(lambda dblOutputActivation, dblTargetOutput:
            output_error(dblOutputActivation, dblTargetOutput),
            listDblOuputActivations,listTargetOutputs)
    listOutputLayerDeltas = layer_deltas(listDblOuputActivations, listOutputError)

    #calculate deltas from the close to output hidden layer towards
    #the first hidden layer after the input
    listDblLayerDeltas.insert(0, listOutputLayerDeltas) #result in correct order!      
    listCurError = listOutputError
    for ixCurHiddenLayer in reversed(range(len(net.listLayer)-1)):
        curLayer = net.listLayer[ixCurHiddenLayer]
        downstreamLayer = net.listLayer[ixCurHiddenLayer+1]
        listDblDownstreamDelta = listDblLayerDeltas[0] 
        listDblCurError = \
            hidden_layer_error(curLayer, listDblDownstreamDelta, downstreamLayer)
        #here we need activation produced by teh current layer
        listDblCurActivation = listDblLayerOutputs[ixCurHiddenLayer]
        listDblCurDeltas = layer_deltas(listDblCurActivation, listDblCurError)
        listDblLayerDeltas.insert(0, listDblCurDeltas)

    #update all perceptrons in layers
    for layer, listDblInputs, listDblDelta in \
        zip(net.listLayer,listDblLayerInputs,listDblLayerDeltas):
        update_layer(layer, listDblInputs, listDblDelta,  dblLearningRate)

    return listDblOuputActivations
              
def init_net(listCLayerSize, dblScale=0.01):
    """Returns a complete feedforward neural network with zero or more hidden layers
    and initialize its weights to random values in (-dblScale,dblScale).

    listCLayerSize: specializes the structure of the NN, 
    [#inputs, #first hidden, #second hidden,...,#outputs]"""
    iCurInputs = listCLayerSize[0]
    listLayers = []
    for cLayerSize in listCLayerSize[1 :]:
        listPcpt = []
        for ixPcpt in range(cLayerSize):
            dblW0 = random.uniform(-dblScale, dblScale)
            listDblWeights = map(lambda dummy: random.uniform(-dblScale, dblScale), 
                                 range(iCurInputs))
            listPcpt.append(Perceptron(listDblWeights, dblW0, ixPcpt))
        listLayers.append(NeuralNetLayer(iCurInputs, listPcpt))
        iCurInputs = cLayerSize
    return NeuralNet(listCLayerSize[0], listLayers)   

def print_net(net):
    """Convenience routine for printing a network to standard out."""
    for layer in net.listLayer:
        print ""
        for pcpt in layer.listPcpt:
            print pcpt
            #break
        #break