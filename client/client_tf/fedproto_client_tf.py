from client.client_tf.client_base_tf import ClientBaseTf
import numpy as np
import tensorflow as tf
import time
import sys
import copy
from pathlib import Path

from model_definition_tf import ModelCreation

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class FedProtoClientTf(ClientBaseTf):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 fraction_fit		= 0,
				 non_iid            = False):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 epochs=epochs,
						 model_name=model_name,
						 client_selection=client_selection,
						 solution_name=solution_name,
						 aggregation_method=aggregation_method,
						 dataset=dataset,
						 perc_of_clients=perc_of_clients,
						 decay=decay,
						 fraction_fit=fraction_fit,
						 non_iid=non_iid)

		self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
		self.train_acc_list = []
		self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
		self.val_acc_list = []
		# Instantiate an optimizer to train the model.
		self.optimizer = tf.keras.optimizers.SGD()
		# Instantiate a loss function.
		self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
		self.global_protos = None
		self.protos = {i: tf.constant([]) for i in range(self.num_classes)}
		self.protos_samples_per_class = {i: 0 for i in range(self.num_classes)}
		self.loss_mse = tf.keras.losses.MSE
		self.lamda = 1
		self.batch_size = 64
		self.train_dataset = None
		self.val_dataset = None
		self.saved_parameters = None
		self.modify_dataset()
		self.create_folder()

	def create_folder(self):
		Path("""fedproto_saved_weights/{}/{}/""".format(self.model_name, self.cid, self.cid)).mkdir(parents=True, exist_ok=True)

	@tf.function
	def train_step(self, x, y):
		with tf.GradientTape() as tape:
			logits, rep = self.model(x, training=True)
			loss_value = self.loss_fn(y, logits)
			# loss_value += sum(self.model.losses)
			mse_loss = 0
			if self.global_protos != None:
				proto_new = np.zeros(rep.shape)
				for i in range(len(y)):
					yy = self.classes[i]
					y_c = tf.get_static_value(yy)
					proto_new[i] = self.global_protos[y_c]
					self.protos_samples_per_class[y_c] += 1

				proto_new = tf.constant(proto_new)
				mse_loss = tf.reduce_mean(self.loss_mse(proto_new, rep))
				loss_value += mse_loss * self.lamda
		grads = tape.gradient(loss_value, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
		self.train_acc_metric.update_state(y, logits)
		return loss_value, rep, mse_loss

	def create_model(self):
		input_shape = self.x_train.shape

		if self.model_name == 'Logist Regression':
			return ModelCreation().create_LogisticRegression_proto(input_shape, self.num_classes)

		elif self.model_name == 'DNN':
			return ModelCreation().create_DNN_proto(input_shape, self.num_classes)

		elif self.model_name == 'CNN':
			return ModelCreation().create_CNN_proto(input_shape, self.num_classes)

		else:
			raise Exception("Wrong model name")

	def modify_dataset(self):

		self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
		# self.x_train = None
		# self.y_train = None
		self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

		self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
		# self.x_test = None
		# self.y_test = None
		self.val_dataset = self.val_dataset.batch(self.batch_size)

	def get_parameters(self, config):
		return self.model.get_weights()

	def get_parameters_of_model(self):
		return self.model.get_weights()

	def set_proto(self, protos):
		self.global_protos = protos

	def fit(self, parameters, config):
		try:
			selected_clients = []
			selected = 0

			loss_train_history = []
			acc_train_history = []

			if config['selected_clients'] != '':
				selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]

			start_time = time.process_time()
			# print("entrada: ", len(parameters), parameters[0].shape)
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				if int(config['round']) > 1:
					self.set_proto(parameters)
					# the parameters are saved in a file because in each round new instances of client are created
					self.load_and_set_parameters()

				selected = 1

				# training
				# =========================================================
				for epoch in range(self.local_epochs):
					# print("\nStart of epoch %d" % (epoch,))


					# Iterate over the batches of the dataset.
					for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):


						self.classes = []
						for i, yy in enumerate(y_batch_train):
							self.classes.append(yy)

						with tf.GradientTape() as tape:
							#print("antes", y_batch_train)
							logits, rep = self.model(x_batch_train, training=True)
							loss_value = self.loss_fn(y_batch_train, logits)
							# loss_value += sum(self.model.losses)
							#print("perda n: ", loss_value)
							if self.global_protos != None:
								proto_new = np.zeros(rep.shape)
								for i in range(len(y_batch_train)):
									yy = self.classes[i]
									y_c = tf.get_static_value(yy)
									if not np.isnan(self.global_protos[y_c]).any() or np.sum(
											self.global_protos[y_c] == 0):
										proto_new[i] = self.global_protos[y_c]
									else:
										proto_new[i] = tf.gather(rep, i)

								proto_new = tf.constant(proto_new)
								mse_loss = tf.reduce_mean(self.loss_mse(proto_new, rep))
								loss_value = tf.math.add(loss_value, mse_loss)

						grads = tape.gradient(loss_value, self.model.trainable_weights)
						self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
						self.train_acc_metric.update_state(y_batch_train, logits)

						for i, yy in enumerate(y_batch_train):
							y_c = int(yy)
							if self.protos[y_c].shape[0] == 0:
								self.protos[y_c] = tf.gather_nd(rep, tf.constant([[i]]))
							else:
								self.protos[y_c] = tf.add(self.protos[y_c], (tf.gather_nd(rep, tf.constant([[i]]))))
							self.protos_samples_per_class[y_c] += 1

						loss_train_history.append(loss_value)

					# Display metrics at the end of each epoch.
					train_acc = float(self.train_acc_metric.result())
					# print("Training acc over epoch: %.4f" % (float(train_acc),), " id: ", self.cid)

					# Reset training metrics at the end of each epoch
					self.train_acc_metric.reset_states()
					acc_train_history.append(train_acc)

				# =========================================================

				trained_parameters = self.model.get_weights()
				# the parameters are saved in a file because in each round new instances of client are created
				self.save_parameters()
				self.saved_parameters = copy.deepcopy(trained_parameters)

			avg_loss_train = float(np.mean(loss_train_history))
			avg_acc_train = float(np.mean(acc_train_history))
			# print("loss media: ", avg_loss_train)
			# print("acc media: ", avg_acc_train)

			total_time = time.process_time() - start_time
			size_of_parameters = sum(map(sys.getsizeof, self.protos))

			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self._write_output(
				filename=self.train_client_filename,
				data=data)

			protos_result = self.agg_func(self.protos)

			fit_response = {
				'cid': self.cid,
				'protos_samples_per_class': self.protos_samples_per_class,
				'proto': protos_result
			}

			return protos_result, len(self.x_train), fit_response
		except Exception as e:
			print("fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def dict_to_numpy(self, data):

		new_data = {}
		for key in data:
			new_data[key] = np.array(data[key])
		return new_data

	def evaluate(self, proto, config):
		try:
			self.set_proto(proto)
			self.load_and_set_parameters()
			avg_loss_test, avg_acc_test = self.evaluate_step()
			size_of_parameters = sum(map(sys.getsizeof, proto))
			data = [config['round'], self.cid, size_of_parameters, avg_loss_test, avg_acc_test]
			self._write_output(filename=self.evaluate_client_filename,
							   data=data)

			evaluation_response = {
				"cid": self.cid,
				"accuracy": float(avg_acc_test),
				"mse_loss": avg_loss_test
			}

			return avg_loss_test, len(self.x_test), evaluation_response
		except Exception as e:
			print("evaluate")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def evaluate_step(self):
		try:
			acc_history = []
			loss_history = []
			for x_batch_val, y_batch_val in self.val_dataset:
				val_logits, rep = self.model(x_batch_val, training=False)
				self.val_acc_metric.update_state(y_batch_val, val_logits)
				val_loss = self.loss_fn(y_batch_val, val_logits).numpy()

				# acc_history.append(val_acc)
				# output = np.ones((y_batch_val.shape[0], self.num_classes))
				# # val_logits, rep = self.model(x, training=False)
				# loss_list = []
				# test_acc = 0
				# test_num = 0
				#
				# for i in range(len(rep)):
				# 	r = rep[i]
				# 	for j in range(len(self.global_protos)):
				# 		pro = self.global_protos[j][0]
				# 		loss_value = tf.reduce_mean(self.loss_mse(r, pro))
				# 		print("peu")
				# 		tf.tensor_scatter_nd_update(loss_list, loss_value, [i])
				# 		output[i, j] = loss_value
				# 		print("antes", tf.keras.backend.eval(tf.argmin(output)))
				# 		test_acc += (tf.reduce_sum(tf.argmin(output) == y_batch_val))
				# 		test_num += y_batch_val.shape[0]
				# 		print("soma")
				loss_history.append(val_loss)

				# self.val_acc_metric.update_state(self.global_protos, output)

			acc_history = self.val_acc_metric.result()
			self.val_acc_metric.reset_states()

			avg_loss_test = float(np.mean(loss_history))
			avg_acc_test = float(np.mean(acc_history))

			return avg_loss_test, avg_acc_test

		except Exception as e:
			print("evaluate step")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def agg_func(self, protos):
		"""
		Returns the average of the weights.
		If the client does not has a prototype of a given class it leverages the one from the 'self.global_protos'
		"""
		try:

			for label in protos:
				protos[label] = np.array(protos[label]) / self.protos_samples_per_class[label]

			proto_shape = protos[label].shape

			if self.global_protos is not None:
				numpy_protos = copy.deepcopy(self.global_protos)
			else:
				numpy_protos = [np.zeros(proto_shape) for i in range(self.num_classes)]
			for label in protos:
				numpy_protos[label] = protos[label]

			return numpy_protos
		except Exception as e:
			print("agg func")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def load_and_set_parameters(self):
		try:
			filename = """./fedproto_saved_weights/{}/{}/saved_model/my_model""".format(self.model_name, self.cid)
			if Path(filename+"/saved_model.pb").exists():
				self.model = tf.keras.models.load_model(filename)
		except Exception as e:
			print("load and set parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters(self):
		try:
			filename = """./fedproto_saved_weights/{}/{}/saved_model/my_model""".format(self.model_name, self.cid)
			self.model.save(filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

