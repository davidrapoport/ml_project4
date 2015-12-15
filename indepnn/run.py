import subprocess, sys, os

# two variables you need to set
pdnndir='~/Desktop/pdnn'  # pointer to PDNN
device='gpu0'  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
os.environ['PYTHONPATH'] = '$PYTHONPATH:~/Desktop/pdnn'

# split the dataset to training, validation and testing sets
# you will see train.pickle.gz, valid.pickle.gz, test.pickle.gz

# train DNN model

print "Training the DNN model ..."
for i in range(20):

	subprocess.call(['export', 'PYTHONPATH=$PYTHONPATH:/home/abe/Desktop/pdnn', '&'
,'''python''', '/home/abe/Desktop/pdnn/cmds/run_DNN.py', '--train-data "data/train_{0}.pickle.gz"'.format(i), 
	                                '--valid-data "data/validate_{0}.pickle.gz"'.format(i),
	                                '--nnet-spec "4096:512:512:2"', '--wdir ./',
	                                '--lrate "C:0.1:200"', '--model-save-step 20', 
	                                '--param-output-file dnn_{0}.param'.format(i),  '--cfg-output-file dnn_{0}.cfg'.format(i),
	                                '--batch-size 40', '--dropout-factor 0.8,0.8' '--input-dropout-factor 0.5'])
	print "Done with model %d" % i
# classification on the testing data; -1 means the final layer, that is, the classification softmax layer
# echo "Classifying with the DNN model ..."
# python $pdnndir/cmds/run_Extract_Feats.py --data "data/validate_2.pickle.gz" \
                                          # --nnet-param dnn.param --nnet-cfg dnn.cfg \
                                          # --output-file "dnn.classify.pickle.gz" --layer-index -1 \
                                          # --batch-size 40 >& dnn.testing.log

# python show_results.py dnn.classify.pickle.gz
