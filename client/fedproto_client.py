from client.client_base import ClientBase
import flwr as fl
import numpy as np
import tensorflow as tf
import os
import time
import sys
import copy

from model_definition import ModelCreation

# import warnings
# warnings.simplefilter("ignore")
#
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)\




class FedProtoClient(ClientBase):

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

	@tf.function
	def train_step(self, x, y):
		with tf.GradientTape() as tape:
			logits, rep = self.model(x, training=True)
			loss_value = self.loss_fn(y, logits)
			loss_value += sum(self.model.losses)
			mse_loss = 0
			if self.global_protos != None:
				proto_new = np.zeros(rep.shape)
				for i in range(len(y)):
					yy = self.classes[i]
					y_c = tf.get_static_value(yy)
					proto_new[i] = self.global_protos[y_c][0]
					self.protos_samples_per_class[y_c] += 1

				proto_new = tf.constant(proto_new)
				mse_loss = tf.reduce_mean(self.loss_mse(proto_new, rep))
				loss_value += mse_loss * self.lamda
		grads = tape.gradient(loss_value, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
		self.train_acc_metric.update_state(y, logits)
		return loss_value, rep, mse_loss

	@tf.function
	def test_step(self, x, y):
		# output = tf.ones((y.shape[0], self.num_classes))
		# loss_list = tf.zeros(len(self.global_protos))
		# val_logits, rep = self.model(x, training=False)
		# loss_list = []
		# test_acc = 0
		# test_num = 0
		#
		# for i in range(len(rep)):
		# 	r = rep[i]
		# 	for j in range(len(self.global_protos)):
		# 		pro = self.global_protos[j][0]
		# 		loss_value = tf.reduce_mean(self.loss_mse(r, pro))
		# 		#tf.tensor_scatter_nd_update(loss_list, loss_value, [i])
		# 		#output[i, j] = loss_value
		# 		# print("antes", tf.keras.backend.eval(tf.argmin(output)))
		# 		# test_acc += (tf.reduce_sum(tf.argmin(output) == y))
		# 		# test_num += y.shape[0]
		# 		# print("soma")
		#
		# # self.val_acc_metric.update_state(self.global_protos, output)

		val_logits = self.model(x, training=False)
		self.val_acc_metric.update_state(y, val_logits)
		loss = self.loss_fn(y, val_logits)

		return loss

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

	def create_model(self):
		input_shape = self.x_train.shape

		if self.model_name == 'Logist Regression':
			return ModelCreation().create_LogisticRegression(input_shape, self.num_classes, use_proto=True)

		elif self.model_name == 'DNN':
			return ModelCreation().create_DNN(input_shape, self.num_classes, use_proto=True)

		elif self.model_name == 'CNN':
			return ModelCreation().create_CNN(input_shape, self.num_classes, use_proto=True)

		else:
			raise Exception("Wrong model name")

	def fit(self, parameters, config):
		selected_clients = []
		#self.protos = {}
		trained_parameters = []
		selected = 0

		loss_train_history = []
		acc_train_history = []

		if config['selected_clients'] != '':
			selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]

		start_time = time.process_time()
		# print(config)
		print("entrada: ", len(parameters), parameters[0].shape)
		if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
			if int(config['round']) > 1:
				self.set_proto(parameters)
				print("setou")
			#self.set_parameters_to_model(parameters)

			selected = 1
			#history = self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.local_epochs)
			#========================================================================================
			# for epoch in range(self.local_epochs):
			# 	print("\nStart of epoch %d" % (epoch,))
			#
			# 	# Iterate over the batches of the dataset.
			# 	for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
			# 		#loss_value = float(self.train_step(x_batch_train, y_batch_train))
			#
			# 		with tf.GradientTape() as tape:
			# 			logits, rep = self.model(x_batch_train, training=True)
			# 			loss_value = self.loss_fn(y_batch_train, logits)
			# 		grads = tape.gradient(loss_value, self.model.trainable_weights)
			# 		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
			# 		self.train_acc_metric.update_state(y_batch_train, logits)
			# 		loss_value = float(loss_value)
			#
			# 		#rep = self.model.proto
			# 		# print("classe: ", y_batch_train)
			# 		# print("representacao")
			# 		# print(logits)
			# 		# print(rep)
			#
			# 		loss_history.append(loss_value)
			# 		# # Log every 200 batches.
			# 		# if step % 200 == 0:
			# 		# 	print(
			# 		# 		"Training loss (for one batch) at step %d: %.4f"
			# 		# 		% (step, float(loss_value))
			# 		# 	)
			# 		# 	print("Seen so far: %d samples" % ((step + 1) * batch_size))
			#
			# 		# if self.global_protos != None:
			# 		# 	proto_new = {i: [] for i in range(self.num_classes)}
			# 		# 	for i, yy in enumerate(y_batch_train):
			# 		# 		y_c = int(yy)
			# 		# 		proto_new[y_c] = self.global_protos[y_c]
			# 		# 		self.protos_samples_per_class[y_c] += 1
			# 		#
			# 		# 	data = tf.constant([proto_new[key] for key in proto_new])
			# 		#
			# 		# 	loss_value += float(self.loss_mse(data, rep) )* self.lamda
			# 		#
			# 		# for i, yy in enumerate(y_batch_train):
			# 		# 	y_c = int(yy)
			# 		# 	self.protos[y_c].append(tf.gather_nd(rep, tf.constant([[i]])))
			# 		# 	self.protos_samples_per_class[y_c] += 1
			#
			# 	# Display metrics at the end of each epoch.
			# 	train_acc = self.train_acc_metric.result()
			# 	acc_history.append(train_acc)
			# 	print("Training acc over epoch: %.4f" % (float(train_acc),))
			#
			# 	# Reset training metrics at the end of each epoch
			# 	self.train_acc_metric.reset_states()
			#
			#
			# 	# Run a validation loop at the end of each epoch.
			# 	# for x_batch_test, y_batch_test in self.val_dataset:
			# 	# 	#self.test_step(x_batch_train, y_batch_train)
			# 	# 	val_logits, val_reg = self.model(x_batch_test, training=False)
			# 	# 	self.val_acc_metric.update_state(y_batch_test, val_logits)
			# 	# print("validou")
			# 	# val_acc = self.val_acc_metric.result()
			# 	# self.val_acc_metric.reset_states()
			# 	# print("Validation acc: %.4f" % (float(val_acc),))
			# 	# print("Time taken: %.2fs" % (time.time() - start_time))
			#
			# 	#acc_history = self.train_acc_metric.result()
			# 	#self.train_acc_metric.reset_states()
			if self.saved_parameters is not None:
				self.set_parameters_to_model(self.saved_parameters)
			try:
				for epoch in range(self.epochs):
					print("\nStart of epoch %d" % (epoch,))
					start_time = time.time()

					# Iterate over the batches of the dataset.
					for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
						# with tf.GradientTape() as tape:
						# 	logits = self.model(x_batch_train, training=True)
						# 	loss_value = self.loss_fn(y_batch_train, logits)
						# 	loss_history.append(loss_value)
						# grads = tape.gradient(loss_value, self.model.trainable_weights)
						# self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
						# Update training metric.
						#self.train_acc_metric.update_state(y_batch_train, logits)
						self.classes = []
						for i, yy in enumerate(y_batch_train):
							self.classes.append(yy)

						# training
						# =========================================================
						# loss_value, rep, loss_mse = self.train_step(x_batch_train, y_batch_train)
						with tf.GradientTape() as tape:
							logits, rep = self.model(x_batch_train, training=True)
							loss_value = self.loss_fn(y_batch_train, logits)
							loss_value += sum(self.model.losses)

							if self.global_protos != None:
								proto_new = np.zeros(rep.shape)
								for i in range(len(y_batch_train)):
									yy = self.classes[i]
									y_c = tf.get_static_value(yy)
									proto_new[i] = self.global_protos[y_c][0]
									self.protos_samples_per_class[y_c] += 1

								proto_new = tf.constant(proto_new)
								mse_loss = tf.reduce_mean(self.loss_mse(proto_new, rep))
								loss_value += mse_loss * self.lamda
						grads = tape.gradient(loss_value, self.model.trainable_weights)
						self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
						self.train_acc_metric.update_state(y_batch_train, logits)

						# =========================================================

						for i, yy in enumerate(y_batch_train):
							y_c = int(yy)
							# print("atribu")
							# print(len(self.protos), y_c)
							if self.protos[y_c].shape[0] == 0:
								self.protos[y_c] = tf.gather_nd(rep, tf.constant([[i]]))
							else:
								self.protos[y_c] = tf.add(self.protos[y_c], (tf.gather_nd(rep, tf.constant([[i]]))))
							self.protos_samples_per_class[y_c] += 1

						loss_train_history.append(loss_value)

					# Display metrics at the end of each epoch.
					train_acc = float(self.train_acc_metric.result())
					print("Training acc over epoch: %.4f" % (float(train_acc),), " id: ", self.cid)


					# Reset training metrics at the end of each epoch
					self.train_acc_metric.reset_states()

					# Run a validation loop at the end of each epoch.
					# for x_batch_val, y_batch_val in self.val_dataset:
					# 	# val_logits = self.model(x_batch_val, training=False)
					# 	# # Update val metrics
					# 	# self.val_acc_metric.update_state(y_batch_val, val_logits)
					# 	val_loss = self.test_step(x_batch_val, y_batch_val)
					# val_acc = self.val_acc_metric.result()
					acc_train_history.append(train_acc)
					#self.val_acc_metric.reset_states()
					# print("Validation acc: %.4f" % (float(val_acc),))
					# print("Time taken: %.2fs" % (time.time() - start_time))


			except Exception as e:
				print("ERROU")
				print(e)
				exit()

			# ========================================================================================
			trained_parameters = self.model.get_weights()
			self.saved_parameters = copy.deepcopy(trained_parameters)
			print("salvou ", self.saved_parameters)

		avg_loss_train = float(np.mean(loss_train_history))
		avg_acc_train = float(np.mean(acc_train_history))
		print("validacao no fit")
		self.validacao(v='no fit')
		print("loss media: ", avg_loss_train)
		print("acc media: ", avg_acc_train)

		total_time = time.process_time() - start_time
		size_of_parameters = sum(map(sys.getsizeof, self.protos))
		# avg_loss_train = np.mean(history.history['loss'])
		# avg_acc_train = np.mean(history.history['accuracy'])

		filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/train_client.csv"
		data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

		self._write_output(
			filename=filename,
			data=data)

		fit_response = {
			'cid': self.cid,
			'protos_samples_per_class': self.protos_samples_per_class
		}

		self.normalize_proto()
		protos_result = self.dict_to_numpy(self.protos)


		# print("antes: ", trained_parameters, type(trained_parameters))
		print("ttt: ", len(protos_result), protos_result[0].shape)


		# print("novo")
		# print(protos_result)

		return protos_result, len(self.x_train), fit_response
	def dict_to_numpy(self, data):

		list_data = []

		for key in data:

			list_data += [np.array(data[key])]
		#print("converte: ", np.array(list))
		return list_data

	def normalize_proto(self):

		for key in self.protos:

			self.protos[key] = self.protos[key]/self.protos_samples_per_class[key]

	def evaluate(self, proto, config):
		print("avaliar")
		self.set_proto(proto)
		#self.set_parameters_to_model(parameters)
		#loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)

		avg_loss_test, avg_acc_test = self.validacao(v='fora fit')

		#print("ola: ", loss, accuracy)

		size_of_parameters = sum(map(sys.getsizeof, proto))

		filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/evaluate_client.csv"
		data = [config['round'], self.cid, size_of_parameters, avg_loss_test, avg_acc_test]

		self._write_output(filename=filename,
						   data=data)

		evaluation_response = {
			"cid": self.cid,
			"accuracy": float(avg_acc_test)
		}

		return avg_loss_test, len(self.x_test), evaluation_response

	def validacao(self, v=""):

		if self.saved_parameters is not None:
			print("coloca")
			self.set_parameters_to_model(self.saved_parameters)

		acc_history = []
		loss_history = []
		for x_batch_val, y_batch_val in self.val_dataset:
			val_logits, rep = self.model(x_batch_val, training=False)
			# if self.global_protos is not None:
			# 	for i in range(len(rep)):
			# 		r = rep[i]
			#
			# 		for j in range(len(self.global_protos)):
			# 			pro = self.global_protos[j][0]
			# 			loss_value = tf.reduce_mean(self.loss_mse(r, pro))
			# 			print("fora", loss_value)
			# 			#
			# 			# print("ola: ", loss_value)




			self.val_acc_metric.update_state(y_batch_val, val_logits)
			val_loss = self.loss_fn(y_batch_val, val_logits)
			# loss_history.append(val_loss)
			# acc_history.append(val_acc)

			# output = tf.ones((y.shape[0], self.num_classes))
			# loss_list = tf.zeros(len(self.global_protos))
			# val_logits, rep = self.model(x, training=False)
			# loss_list = []
			# test_acc = 0
			# test_num = 0
			#
			# for i in range(len(rep)):
			# 	r = rep[i]
			# 	for j in range(len(self.global_protos)):
			# 		pro = self.global_protos[j][0]
			# 		loss_value = tf.reduce_mean(self.loss_mse(r, pro))
			# 		#tf.tensor_scatter_nd_update(loss_list, loss_value, [i])
			# 		#output[i, j] = loss_value
			# 		# print("antes", tf.keras.backend.eval(tf.argmin(output)))
			# 		# test_acc += (tf.reduce_sum(tf.argmin(output) == y))
			# 		# test_num += y.shape[0]
			# 		# print("soma")
			#
			# # self.val_acc_metric.update_state(self.global_protos, output)






			loss_history.append(val_loss)
		acc_history = self.val_acc_metric.result()
		self.val_acc_metric.reset_states()

		avg_loss_test = float(np.mean(loss_history))
		avg_acc_test = float(np.mean(acc_history))

		print("Val loss: ", avg_loss_test, v)
		print("Val acc: ", avg_acc_test, v)

		return avg_loss_test, avg_acc_test

	def agg_func(self, protos):
		"""
		Returns the average of the weights.
		"""
		size = len(protos)
		for key in protos:
			protos[key] = np.sum(protos[key], axis=0)/size

		return protos

