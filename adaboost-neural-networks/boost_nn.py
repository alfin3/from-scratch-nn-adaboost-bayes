#!/usr/bin/env python

"""
boost_nn.py

Implements adaptive boosting for neural network classifiers.

Python 2.7.12 
A 'from scratch' implementation.
"""
import math
import nn
import run_nn
import numpy
import sys 
import csv

class BoostInstance(object):
    """data instance is in instance and the instance weight in dblWeight."""
    def __init__(self, listAttrs, iLabel, dblWeight=1.0):
        self.listAttrs = listAttrs
        self.iLabel = iLabel
        self.dblWeight = dblWeight
        
XOR_INSTANCES = [BoostInstance([-1.0,-1.0], 0), 
                 BoostInstance([-1.0,1.0], 1),
                 BoostInstance([1.0,-1.0], 1), 
                 BoostInstance([1.0,1.0], 0)]
                 
XOR_CF_INSTANCES = [BoostInstance([-1.0,-1.0], 0), 
                    BoostInstance([-1.0,1.0], 1),
                    BoostInstance([1.0,-1.0], 1), 
                    BoostInstance([1.0,1.0], 0),
                    BoostInstance([1.0,-1.0], 0, 2.0),
                    BoostInstance([1.0,-1.0], 0, 2.0)]

def inst_to_boostinst(listInst):
    """converts a data set of instances as in nn.py to a data set of 
    BoostInstances, each with dblWeight of 1.0
    """
    return map(lambda inst: BoostInstance(inst.listDblFeatures, inst.iLabel), listInst)

def init_weights(listBoostInst):
    """Initialize the weights of the instances in listInst so that each
    instance has weight 1/(# of instances). This function modifies
    the weights in place and does not return anything.
    listInst = [BoostInstance([],True,0.5), BoostInstance([],True,0.25)]
    init_weights(listInst)
    print listInst
    [BoostInstance([], True, 0.50), BoostInstance([], True, 0.50)]
    """
    numbInstances = len(listBoostInst)
    for inst in listBoostInst:
        inst.dblWeight = float(1)/float(numbInstances)
    return

def normalize_weights(listBoostInst):
    """Normalize the weights of all the instances in listInst so that the sum
    of their weights totals to 1.0. The function modifies the weights of the 
    instances in-place and does not return anything.
    listInst = [BoostInstance([],True,0.1), BoostInstance([],False,0.3)]
    normalize_weights(listInst)
    print listInst
    [BoostInstance([], True, 0.25), BoostInstance([], False, 0.75)]
    """
    if len(listBoostInst) is 0: raise 'normalization on empty data'
    totalWeight = \
        reduce(lambda h, t: h+t,
            map(lambda inst: inst.dblWeight, listBoostInst))
    for inst in listBoostInst:
        instCurWeight = inst.dblWeight
        inst.dblWeight = float(instCurWeight/totalWeight)
    return
    
def learn_nn_classifier(listCLayerSize, lstTrainBoostInst, fxnEncode, fxnDecode):
    """Given the test data set of BoostInstances, learns and 
    returns the learned neural net.
    net = nn.init_net([2,2,1])
    learn_nn_classifier(net, XOR_INSTANCES, nn.binary_encode_label, 
                        nn.binary_decode_net_output)
    Learning rate: 1.923447Round 200 complete.  Training Accuracy: 0.250000
    Learning rate: 1.905125Round 250 complete.  Training Accuracy: 0.750000
    Learning rate: 1.887149Round 300 complete.  Training Accuracy: 0.750000
    Learning rate: 1.869508Round 350 complete.  Training Accuracy: 0.750000
    Learning rate: 1.852195Round 400 complete.  Training Accuracy: 1.000000
    Out[99]: <nn.NeuralNet at 0x9504950>
    """
    # learn through rounds number of epochs
    rounds = 500
    interval = 50
    stopRound = 350
    iInst = len(lstTrainBoostInst)
    net = nn.init_net(listCLayerSize)
    
    for ixRound in xrange(rounds):
        dblAlpha = 2.0*rounds/(ixRound + rounds)
        
        # learn through one epoch and compute the error
        errors = 0
        for boostInst in lstTrainBoostInst:
            inst = nn.Instance(boostInst.iLabel, boostInst.listAttrs)
            listDblOut = nn.update_net(net, inst, dblAlpha, fxnEncode(inst.iLabel))
            iGuess = fxnDecode(listDblOut)
            if iGuess != inst.iLabel:
              errors += 1
          
        # print result after an interval of rounds
        if not((ixRound+1) % interval):
            sys.stderr.write('Learning rate: %f ' % dblAlpha)
            sys.stderr.write("Epoch: %d Training Accuracy: %f \n" % (ixRound + 1,
            1 - errors * 1.0 / iInst))
            
        # implement a stopping condition.
        if (ixRound+1) == stopRound:
            return net
    return net
    
def classify(net, boostInst, fxnDecode):
    """Using the neural net, return the predicted label for the boostInst.
    net = nn.init_net([2,2,1])
    net = learn_nn_classifier(net, XOR_INSTANCES, nn.binary_encode_label, 
                              nn.binary_decode_net_output)
    classify(net, BoostInstance([-1.0,-1.0], 0), nn.binary_decode_net_output)
    Out[108]: 0
    classify(net, BoostInstance([1.0,-1.0], 1), nn.binary_decode_net_output)
    Out[109]: 1"""
    listDblOut = nn.feed_forward(net, boostInst.listAttrs)
    return fxnDecode(listDblOut)     
  
def classifier_error(net, listBoostInst, fxnDecode):
    """Given a neural net, a list of BoostInstances and a decode function, 
    returns the weighted error of the neural net classifier. 
    net = nn.init_net([2,2,1])
    net = learn_nn_classifier(net, XOR_INSTANCES, nn.binary_encode_label, 
                              nn.binary_decode_net_output)
    classifier_error(net, XOR_CF_INSTANCES, nn.binary_decode_net_output)
    Out[11]: 0.5
    Note: nn.update_net function is used within each epoch. In contrast classify 
    uses nn.feed_forward after an epochs.
    """
    correctWeight = 0.0
    cCorrect = 0
    incorrectWeight = 0.0
    cIncorrect = 0
    for boostInst in listBoostInst:
        if classify(net, boostInst, fxnDecode) == boostInst.iLabel:
            correctWeight += boostInst.dblWeight
            cCorrect += 1
        else:
            incorrectWeight += boostInst.dblWeight
            cIncorrect += 1
    totalWeight = correctWeight + incorrectWeight
    if totalWeight is 0.0: raise 'Total Weight is 0 in Classifier error'
    return incorrectWeight/totalWeight
            
def classifier_weight(dblError):
    """Returns the classifier weight alpha from the classifier's training
    error."""
    dblAlpha = math.log((float(1) - float(dblError))/float(dblError))*0.5
    return dblAlpha

def update_weight(boostInst, dblClassifierWeight, intClassifiedLabel):
    """Re-weight a BoostInstance given the classifier weight, and the label
    assigned to the BoostInstance by the classifier. This function acts in place
    and does not return anything."""
    oldWeight = boostInst.dblWeight
    if not (intClassifiedLabel is boostInst.iLabel):
        newWeight = oldWeight * math.pow(math.e,dblClassifierWeight)
        boostInst.dblWeight = newWeight
    else:
        newWeight = oldWeight * math.pow(math.e,-dblClassifierWeight)
        boostInst.dblWeight = newWeight
        
def sample_data(listBoostInst):
    """Given a list of BoostInstances, returns a list of BoostInstances of the 
    same lenth. This list is sampled from the original list by 
    treating weights in the original list as sampling probabilities
    listBoostInst = [BoostInstance([-1.0,-1.0], 0, 0.0),
                     BoostInstance([-1.0,-1.0], 0, 0.0),
                     BoostInstance([-1.0,-1.0], 0, 0.0),
                     BoostInstance([-1.0,-1.0], 0, 0.0),
                     BoostInstance([1.0,1.0], 0, 0.3),
                     BoostInstance([1.0,-1.0], 0, 0.7)]
    sampledList = sample_data(listBoostInst)
   [boostInst.listAttrs for boostInst in sampledList]
   [[1.0, -1.0], [1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]"""
    cSize = len(listBoostInst)
    probs = [boostInst.dblWeight for boostInst in listBoostInst]
    return numpy.random.choice(listBoostInst, cSize, p=probs)

def one_round_boost(listCLayerSize, fxnEncode, fxnDecode, listBoostInst):
    """
    Given the configuration of a complete neural net classifier (as in init.net),
    encode and decode functions, and a list of BoostInstances with normalized
    weights,
    - runs a single round of boosting, updates the weights of the data set in place,
      normalizes weights in place,
    - returns a triple (classifier, error, classifier weight). 
    """
    listSampledBoostInst = sample_data(listBoostInst)
    new_net = learn_nn_classifier(listCLayerSize, listSampledBoostInst, fxnEncode, fxnDecode)
    classifierError = classifier_error(new_net, listBoostInst, fxnDecode)
    if classifierError < 0.0000001:
        return (new_net, 0, 'inf')
    classifierWeight = classifier_weight(classifierError)
    for boostInst in listBoostInst:
        update_weight(boostInst, classifierWeight,
                      classify(new_net, boostInst, fxnDecode))
    normalize_weights(listBoostInst)
    return (new_net, classifierError, classifierWeight)
    
class BoostResult(object):
    def __init__(self, listDblWeights, listClassifiers):
        self.listDblWeights = listDblWeights
        self.listClassifiers = listClassifiers
               
def boost_classify(boostResult, boostInst, fxnDecode):
    """Given a BoostResult and an instance, return the label
    predicted for the instance by the boosted classifier.
    """
    dictOutcomeToWeight = {}
    for c, cweight in zip(boostResult.listClassifiers, boostResult.listDblWeights):
        #inputLayer = c.input_layer()
        #print inputLayer.listPcpt[0].listDblW
        #nn.print_net(c) 
        outcome = classify(c, boostInst, fxnDecode)
        if outcome in dictOutcomeToWeight:
            prevOutcomeSum = dictOutcomeToWeight[outcome]
            dictOutcomeToWeight[outcome] = prevOutcomeSum + cweight 
        else:
            dictOutcomeToWeight[outcome] = cweight        
    weightedMajorityVote = sorted(dictOutcomeToWeight.items(),
                                  key = lambda pair: pair[1], reverse = True)
    return weightedMajorityVote[0][0]
    
def boost_num_correct(boostResult, listBoostInst, fxnDecode):
    """returns the number of instances, correctly classified with a boostResult"""
    cCorrect = 0
    for boostInst in listBoostInst:
        iGuess = boost_classify(boostResult, boostInst, fxnDecode)
        cCorrect += int(boostInst.iLabel == iGuess)
    return cCorrect

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
#+++++++++++++++++++++++++Run NN Boosting experiments++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def experiment(opts): 
    """Run the adaboost experiment with feedforward complete NNs"""
    dictSeen = {}
    writer = csv.writer(open("adaboost+NN_350e_50n.csv", "wb"))
    boostResult = BoostResult([], [])
    #iMaxEpochs = opts.maximal_epochs
    #iPrintInterval = opts.print_interval
    #iStopRound = opts.stop_epoch
    iBoostRounds = opts.boost_rounds
    config = [opts.num_inputs, opts.num_hidden_units, opts.num_outputs]
    
    def load(sAttrFilename, sClassFilename):
        if sAttrFilename in dictSeen:
            return dictSeen[sAttrFilename]
        sys.stderr.write("Loading %s and %s..." % (sAttrFilename, sClassFilename))
        listInst = run_nn.load_separate_data(sAttrFilename, sClassFilename, None)
        sys.stderr.write("done.\n")
        dictSeen[sAttrFilename] = listInst
        return listInst
    
    listInstTrain = load("attr_trainset_NN.csv", "class_trainset_NN.csv")
    listInstTrainAdd = load("attr_valset_NN.csv", "class_valset_NN.csv")
    listInstTest = load("attr_testset_NN.csv", "class_testset_NN.csv")
    listInstTrain.extend(listInstTrainAdd)
    listBoostInstTrain = inst_to_boostinst(listInstTrain)
    listBoostInstTest = inst_to_boostinst(listInstTest)
    init_weights(listBoostInstTrain)
    
    lengthTrain = len(listBoostInstTrain)
    lengthTest = len(listBoostInstTest)

    for ixRound in range(iBoostRounds):
        cNet, cError, cWeight = one_round_boost(config, 
                                                nn.distributed_encode_label,  
                                                nn.distributed_decode_net_output,  
                                                listBoostInstTrain)                                        
        boostResult.listClassifiers.append(run_nn.copy_net(cNet))
        boostResult.listDblWeights.append(cWeight)
        cCorrectTrain = boost_num_correct(boostResult, 
                                          listBoostInstTrain, 
                                          nn.distributed_decode_net_output)
        cCorrectTest = boost_num_correct(boostResult, 
                                         listBoostInstTest, 
                                         nn.distributed_decode_net_output)
        sys.stderr.write(
        "Boost Round %d complete. Training Accuracy: %f, Test Accuracy: %f\n" % (
          ixRound + 1,
          cCorrectTrain * 1.0 / lengthTrain,
          cCorrectTest * 1.0 / lengthTest))
        
        # RECORDING
        sFileName = "NN_%d.csv" % (ixRound)
        run_nn.save_net(cNet, sFileName)  
        writer.writerow([ixRound, cWeight, 
                         cCorrectTrain * 1.0 / lengthTrain, 
                         cCorrectTest * 1.0 / lengthTest])
        
        if cError is 'inf':
            return BoostResult([1], [cNet])

def main(argv):
    import optparse
    parser = optparse.OptionParser() 
    parser.add_option("-d", "--doc-test", action="store_true", dest="doctest",
                      help="run doctests in boost_nn.py")                   
    parser.add_option("-b", "--rounds", action="store", dest="boost_rounds",
                      default=1000, type=int, 
                      help="number of boost rounds")
    parser.add_option("--hidden", action="store",
                      dest="num_hidden_units",
                      default=50, type=int,
                      help="number of hidden units in a single layer to use.")
    parser.add_option("--inputs", action="store",
                      dest="num_inputs",
                      default=(54), type=int,
                      help="number of input units to use.")
    parser.add_option("--outputs", action="store",
                      dest="num_outputs",
                      default=(7), type=int,
                      help="number of output units.")
    opts,args = parser.parse_args(argv)
    if opts.doctest:
        import doctest
        doctest.testmod()
        return 0
    experiment(opts)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
  

""" parser.add_option("-e", "--epochs", action="store", dest="maximal_epochs",
                      default=500, type=int, 
                      help="max number of epochs in each boost round")
    parser.add_option("-s", "--stop", action="store", dest="stop_epoch",
                      default=100, type=int, 
                      help="number of epochs actually run in each boost round")
    parser.add_option("-i", "--interval", action="store", dest="print_interval",
                      default=10, type=int, 
                      help="after how many epochs nn parameters are printed") """


    
