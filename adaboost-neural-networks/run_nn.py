#!/usr/bin/env python

"""
run_nn.py

Implement run functions for feedforward complete artifical neural networks in nn.py.

Python 2.7.12 
"""
import sys
import nn
import csv
import copy

#++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++ Data loading ++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++

def parse_input(dataFile, fn_type, numExamples):
    """
    Parses the data file and creates a list data set of numExamples instances.
    lstData = parse_input(open('covtype.csv', "r"), 11340)
    """
    lstData = []
    ct = 0
    for line in dataFile:
        instance = line.split(",")
        lstData.append(map(lambda x:fn_type(x),instance))
        ct += 1
        if not numExamples is None and ct >= numExamples:
          break
    return lstData
      
def load_separate_data(sAttrFilename, sClassFilename, cInstances):
    """Load cInstances attribute instances from sAttrFilename and 
    class instances from sClassFilename (two separate files), 
    and combine them into Instances"""      
    listClass = parse_input(open(sClassFilename,"r"), int, cInstances)
    listAttr = parse_input(open(sAttrFilename,"r"), float, cInstances)
    return map(lambda cl, attr: nn.Instance(cl[-1], attr), listClass, listAttr)
      
def load_data(sFilename, cMaxInstances=None):
    """Load at most cMaxInstances instances from sFilename, or all instance
    if cMaxInstances is None."""
    listInst = []
    try:
        infile = open(sFilename)
        listInputs = []
        iLabel = None
        for sLine in infile:
            if sLine.startswith('#'):
                if iLabel is not None:
                    listInst.append(nn.ImageInstance(iLabel, listInputs))
                    if (cMaxInstances is not None and
                        len(listInst) >= cMaxInstances):
                        break
                    listInputs = []
                iLabel = int(sLine.split('#')[-1])
            else:
                listInputs.append([float(s)/255.0 for s in sLine.split()])
        if iLabel is not None:
            listInst.append(nn.ImageInstance(iLabel, listInputs))
    finally:
        infile.close()
    return listInst
      
#TRAINING_9K = "training-9k.txt"
#TEST_1K = "validation-1k.txt"    

def load_instances(sFilename, cMaxInstances=None):
    return load_data(sFilename, cMaxInstances)

#def load_training_9k(cMaxInstances):
#    return load_instances(TRAINING_9K,cMaxInstances)

#def load_test_1k(cMaxInstances):
#    return load_instances(TEST_1K,cMaxInstances)
    
    
#++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++ NN evaluation +++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++

def num_correct(net, listInst, fxnDecode):
    """
    listInstTrain = load_training_9k(9000)
    listInstTest = load_test_1k(1000)
    net = nn.init_net([14*14,40,10])
    num_correct(net, listInstTest,nn.distributed_decode_net_output)
    Out: 105
    num_correct(net, listInstTrain,nn.distributed_decode_net_output)
    Out: 893"""
    cCorrect = 0
    for inst in listInst:
        listDblOut = nn.feed_forward(net,inst.listDblFeatures)
        iGuess = fxnDecode(listDblOut)
        cCorrect += int(inst.iLabel == iGuess)
    return cCorrect


def evaluate_net(net,listInst,fxnDecode):
    """
    listInstTrain = load_training_9k(9000)
    listInstTest = load_test_1k(1000)
    net = nn.init_net([14*14,15,10])
    evaluate_net(net,listInstTest,nn.distributed_decode_net_output)
    Out: 0.105"""
    cCorrect = 0
    for inst in listInst:
        iResult = fxnDecode(nn.feed_forward(net,inst.listDblFeatures))
        cCorrect += int(iResult == inst.iLabel)
    return float(cCorrect)/float(len(listInst))

def build_and_measure_net(net,listInstTrain,listInstTest,
                          fxnEncode,fxnDecode,dblLearningRate,
                          cRounds): 
    """
    listInstTrain = load_training_9k(9000)
    listInstTest = load_test_1k(1000)
    net = nn.init_net([14*14,40,10])
    build_and_measure_net(net,listInstTrain, listInstTest, nn.distributed_encode_label,nn.distributed_decode_net_output,0.1, 3)
    Out: (0.203, 0.21766666666666667)
    build_and_measure_net(net,listInstTrain, listInstTest, nn.distributed_encode_label,nn.distributed_decode_net_output,0.1, 3)
    Out: (0.412, 0.432)
    build_and_measure_net(net,listInstTrain, listInstTest, nn.distributed_encode_label,nn.distributed_decode_net_output,0.1, 3)
    Out: (0.705, 0.7373333333333333)
    build_and_measure_net(net,listInstTrain, listInstTest, nn.distributed_encode_label,nn.distributed_decode_net_output,0.1, 3)
    Out: (0.815, 0.8391111111111111)"""
    for _ in xrange(cRounds):
        for inst in listInstTrain:
            listDblTarget = fxnEncode(inst.iLabel)
            nn.update_net(net, inst, dblLearningRate, listDblTarget)
        dblTestError = evaluate_net(net, listInstTest, fxnDecode)
        dblTrainingError = evaluate_net(net, listInstTrain, fxnDecode)
        return dblTestError,dblTrainingError
        

#++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++ XOR example/testing +++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++

XOR_INSTANCES = [nn.Instance(0.1, [-1.0,-1.0]), nn.Instance(0.9, [-1.0,1.0]),
                 nn.Instance(0.9, [1.0,-1.0]), nn.Instance(0.1, [1.0,1.0])]

def build_xor_net():
    HIDDEN_NODES = 2
    ROUNDS = 5000
    assert XOR_INSTANCES
    net = nn.init_net([2, HIDDEN_NODES, 1], 0.001)
    for ixRound in xrange(ROUNDS):
        dblAlpha = 2.0*ROUNDS/(ixRound + ROUNDS)
        for inst in XOR_INSTANCES:
            nn.update_net(net, inst, dblAlpha, [inst.iLabel])
    return net

def learn_xor():
    """Build a neural network which solves XOR."""
    net = nn.init_net([2,2,1])
    for _ in xrange(5000):
        for inst in XOR_INSTANCES:
            nn.update_net(net,inst, 0.5, [inst.iLabel])
    for inst in XOR_INSTANCES:
        print inst.iLabel, nn.feed_forward(net,inst.listDblFeatures)
    nn.print_net(net)
    
def save_net(net, sFileName):
    """Creates a CSV file where the network is printed out:
       the layers are separated by '&', followed by the cInput
       the last value of each full line is the number of Pcpt
       the second last value is the threshold value of each Pcpt"""
    writer = csv.writer(open(sFileName, "wb"))
    for layer in net.listLayer:
        writer.writerow(['&'])
        writer.writerow([layer.cInputs]) 
        for pcpt in layer.listPcpt:
            pcptInst = pcpt.listDblW
            pcptInst.append(pcpt.dblW0)
            pcptInst.append(pcpt.ix)
            writer.writerow(pcptInst)
            
def load_net(sFileName):
    """The file must start with & and then a single integer 
       of the number of input units into the first hidden layer;
       & must always be followed by an integer and then perceptron lines"""       
    lstLayers = []
    reader = csv.reader(open(sFileName, "r"))
    line = next(reader)
    lstPcpt= []
    line = next(reader)
    cInputs = int(line[-1])
    cInputsFirst = cInputs
    
    for line in reader:
        if line[-1] is '&':
            lstLayers.append(nn.NeuralNetLayer(cInputs, lstPcpt))
            lstPcpt= []
        elif len(line) is 1:
            cInputs = int(line[-1])   
        else:
            pcptLine = map(lambda x: float(x), line)
            lstPcpt.append(nn.Perceptron(pcptLine[:-2], pcptLine[-2], int(pcptLine[-1])))
    lstLayers.append(nn.NeuralNetLayer(cInputs, lstPcpt)) 
       
    #nn.print_net(nn.NeuralNet(cInputsFirst, lstLayers))
    return nn.NeuralNet(cInputsFirst, lstLayers)
    
def copy_net(net):
    """returns a separate copy of net"""
    newListLayer = []
    newCInputsFirst = copy.copy(net.cInputs)
    for layer in net.listLayer:
        newListPcpt = []
        for pcpt in layer.listPcpt:
            newListDblW = copy.copy(pcpt.listDblW)
            newDblW0 = copy.copy(pcpt.dblW0)
            newPcptIx = copy.copy(pcpt.ix)
            newListPcpt.append(nn.Perceptron(newListDblW, newDblW0, newPcptIx))
        newCInputs = copy.copy(layer.cInputs)
        newListLayer.append(nn.NeuralNetLayer(newCInputs, newListPcpt))
    return nn.NeuralNet(newCInputsFirst, newListLayer)
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
#++++++++++++++++++++++++++++++Run NN experiments++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
def experiment(opts):
    """Run a neural net experiment."""
    dictSeen = {}
    writer = csv.writer(open("500epochs_80Nodes_4.csv", "wb"))
    
    def load(sAttrFilename, sClassFilename):
        if sAttrFilename in dictSeen:
            return dictSeen[sAttrFilename]
        sys.stderr.write("Loading %s and %s..." % (sAttrFilename, sClassFilename))
        listInst = load_separate_data(sAttrFilename, 
                                      sClassFilename, 
                                      opts.max_inst)
        sys.stderr.write("done.\n")
        dictSeen[sAttrFilename] = listInst
        return listInst
    listInstTrain = load(opts.attrtrain, opts.classtrain)
    listInstVal = load(opts.attrvalidation, opts.classvalidation)
    listInstTest = load(opts.attrtest, opts.classtest)
    config = [opts.num_inputs]
    if opts.hidden_units:
      print 'Adding a hidden layer with %d units' % opts.hidden_units
      config.append(opts.hidden_units)
      
    config.append(7) #Covertype data set
    #config.append(10)
    #config.append(4)
    
    dblPrevAccuracy = 0.0
    dblCurAccuracy = 0.0
    dblStopDeltaAccuracy = 0.05
    iEpochInterval = 5
    
    net = nn.init_net(config) 
    for ixRound in xrange(opts.rounds):
        dblAlpha = 2.0*opts.rounds/(ixRound + opts.rounds)
        print 'Learning rate: %f' % dblAlpha
        # Count error
        errors = 0
        for inst in listInstTrain:
            listDblOut = nn.update_net(net,inst,dblAlpha,nn.distributed_encode_label(inst.iLabel))
            iGuess = nn.distributed_decode_net_output(listDblOut)
            if iGuess != inst.iLabel:
              errors += 1
        # Compute validation error
        validation_correct = num_correct(net, listInstVal, nn.distributed_decode_net_output)
        sys.stderr.write(
        "Round %d complete.  Training Accuracy: %f, Validation Accuracy: %f\n" % (
          ixRound + 1,
          1 - errors * 1.0 / len(listInstTrain),
          validation_correct * 1.0 / len(listInstVal)))
        # RECORDING
        writer.writerow([ixRound + 1, dblAlpha, 1 - errors * 1.0/ len(listInstTrain),
        validation_correct * 1.0 / len(listInstVal)])
        if opts.stopping_condition:
            if (ixRound+1)%iEpochInterval is 0:
                dblCurAccuracy = validation_correct * 1.0 / len(listInstVal)
                if (dblCurAccuracy - dblPrevAccuracy) < dblStopDeltaAccuracy:
                    break
                else:
                    dblPrevAccuracy = dblCurAccuracy
                
    cCorrect = 0
    for inst in listInstTest:
        listDblOut = nn.feed_forward(net,inst.listDblFeatures)
        iGuess = nn.distributed_decode_net_output(listDblOut)
        cCorrect += int(inst.iLabel == iGuess)
    print "correct:",cCorrect, "out of", len(listInstTest),
    print "(%.1f%%)" % (100.0*cCorrect/len(listInstTest))

"""
def main(argv):
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("-d", "--doc-test", action="store_true", dest="doctest",
                      help="run doctests in run_nn.py")
    parser.add_option("--ar", "--attrtrain", action="store", dest="attrtrain",
                      help="file containing attribute training instances",
                      default="attr_trainset_NN.csv")
    parser.add_option("--cr", "--classtrain", action="store", dest="classtrain",
                      help="file containing class training instances",
                      default="class_trainset_NN.csv")                      
    parser.add_option("--at", "--attrtest", action="store", dest="attrtest",
                      default="attr_testset_NN.csv",
                      help="file containing attribute test instances")
    parser.add_option("--ct", "--classtest", action="store", dest="classtest",
                      default="class_testset_NN.csv",
                      help="file containing class test instances")                      
    parser.add_option("--av", "--attrvalidation", action="store", dest="attrvalidation",
                      default="attr_valset_NN.csv",
                      help="file containing attribute validation instances")
    parser.add_option("--cv", "--classvalidation", action="store", dest="classvalidation",
                      default="class_valset_NN.csv",
                      help="file containing class validation instances")                      
    parser.add_option("-n", "--rounds", action="store", dest="rounds",
                      default=10, type=int, help="number of training rounds")
    parser.add_option("-m", "--max-instances", action="store", dest="max_inst",
                      default=None, type=int,
                      help="maximum number of instances to load")
    parser.add_option("-l", "--learning_rate", action="store",
                      dest="learning_rate",
                      default=1.0, type=float,
                      help="the learning rate to use")
    parser.add_option("--hidden", action="store",
                      dest="hidden_units",
                      default=None, type=int,
                      help="number of hidden units to use.")
    parser.add_option("--num_inputs", action="store",
                      dest="num_inputs",
                      default=(54), type=int,
                      help="number of hidden units to use.")
    parser.add_option("--enable-stopping", action="store_true",
                      dest="stopping_condition", default=False,
                      help="detect when to stop training early (TODO)")
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
  
"""

  
    

