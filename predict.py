import tensorflow as tf
import numpy as np


"""
Helpful links:
https://www.tensorflow.org/tutorials/recurrent#lstm
https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell
https://www.tensorflow.org/api_docs/python/tf/one_hot
"""


fasta_file = "chr1.fa"
num_features = 5


#create a one-hot vector from base pair, must be lower case
base_number = {'a': 0, 't':1, 'c':2, 'g':3, 'n':4}
def base_to_vector(base): 
	base = base.lower()
	num = base_number[base]
	array = np.zeros(num_features)
	array[num] = 1.0
	return array 

#open fasta file and conver to array of one-hot arrays
num_lines = sum(1 for line in open(fasta_file))
genome = np.empty((num_lines*50+100, num_features))
i = 0
with open(fasta_file, 'r') as openfile:
	#ignore first line 
	openfile.readline()
	#read through each character and convert to one-hot vector
	for line in openfile:
		print(i)
		if line == 'N'*50+'\n':
			continue
		if i > 10000:
			break
		for base in line:
			if base.lower() in "atcgn":
				genome[i] = base_to_vector(base)
				i += 1
genome_max_index = i
print(genome[300:350])
print(genome.shape)

def create_batches(num_batches, batch_size):
	data = np.empty((num_batches, batch_size, num_features))
	start_indices = np.empty(batch_size)
	start_indices.fill(np.random.randint(0, genome_max_index - num_batches))
	for t in range(num_batches):
		for n in range(batch_size):
			index = int(start_indices[n] + t)
			data[t,n] = genome[index]
	return data
		
	

num_features = 5
batch_size = 100
num_batches = 300
lstm_size = 10#no idea what to set this
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)


sess = tf.Session()


bases_dataset_placeholder = tf.placeholder(tf.float32, [num_batches, batch_size, num_features])
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
hidden_state = tf.zeros([batch_size, lstm_size])
current_state = tf.zeros([batch_size, lstm_size])
state = hidden_state, current_state
probabilities = []
loss = 0.0
training_data = create_batches(num_batches, batch_size)
for current_batch in bases_dataset_placeholder:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += tf.nn.softmax_cross_entropy_with_logits( labels=training_data, logits=logits)

    #loss += loss_function(probabilities, target_words)
sess.run()