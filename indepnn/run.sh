#!/bin/bash

# two variables you need to set
pdnndir=~/Desktop/pdnn  # pointer to PDNN
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir


# split the dataset to training, validation and testing sets
# you will see train.pickle.gz, valid.pickle.gz, test.pickle.gz

# train DNN model

for i in `seq 1 20`;
do
	echo "Training the DNN model $i ..."
	python $pdnndir/cmds/run_DNN.py --train-data "data/train_$i.pickle.gz" \
	                                --valid-data "data/validate_$i.pickle.gz" \
	                                --nnet-spec "4096:512:512:2" --wdir ./ \
	                                --lrate "C:0.1:50" --model-save-step 50 \
	                                --param-output-file dnn_$i.param --cfg-output-file dnn_$i.cfg \
	                                --batch-size 40 >& dnn_$i.log #--dropout-factor 0.8,0.8 --input-dropout-factor 0.5

	# classification on the testing data; -1 means the final layer, that is, the classification softmax layer
	echo "Classifying with the DNN model $i ..."
	python $pdnndir/cmds/run_Extract_Feats.py --data "data/test_$i.pickle.gz" \
	                                          --nnet-param dnn_$i.param --nnet-cfg dnn_$i.cfg \
	                                          --output-file "dnn_$i.classify.pickle.gz" --layer-index -1 \
	                                          --batch-size 40 >& dnn_$i.testing.log

	python show_results.py dnn_$i.classify.pickle.gz $i
done



