import os
import csv
import tensorflow as tf

tf.app.flags.DEFINE_integer('N', 12, "TIME_STEPS")
FLAGS = tf.app.flags.FLAGS

env_dist = os.environ
CX_WORD_DIR = env_dist['CX_WORD_DIR']

def main(_):
	accuracy = []

	file_path = CX_WORD_DIR + "/results/" + str(FLAGS.N) + ".txt"
	file = open(file_path, 'r')
	last_line_item = []
	for line in file.readlines():
		line_item = line.split()
		if len(line_item) == 0:
			continue
		if line_item[0] == "Epoch":
			accuracy.append(last_line_item[-1])
		last_line_item = line_item
	file.close()

	out_file_path = CX_WORD_DIR + "/csv/" + str(FLAGS.N) + ".csv"
	out_file = open(out_file_path, 'w')
	for acc in accuracy:
		out_file.write(acc)
		out_file.write(", ")
	out_file.close()

if __name__ == '__main__':
	tf.app.run()