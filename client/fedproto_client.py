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
		self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.global_protos = None
		self.protos = {i: [] for i in range(self.num_classes)}
		self.protos_samples_per_class = {i: 0 for i in range(self.num_classes)}
		self.loss_mse = tf.keras.losses.MSE
		self.lamda = 1
		self.batch_size = 64
		self.train_dataset = None
		self.val_dataset = None
		self.modify_dataset()

	@tf.function
	def train_step(self, x, y):
		with tf.GradientTape() as tape:
			logits = self.model(x, training=True)
			loss_value = self.loss_fn(y, logits)
			loss_value += sum(self.model.losses)
		grads = tape.gradient(loss_value, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
		self.train_acc_metric.update_state(y, logits)
		#print("aqui: ", float(self.train_acc_metric.result()))
		return loss_value

	@tf.function
	def test_step(self, x, y):
		val_logits = self.model(x, training=False)
		loss_value = self.loss_fn(y, val_logits)
		self.val_acc_metric.update_state(y, val_logits)
		return loss_value

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
		self.protos = protos

	def create_model(self):
		input_shape = self.x_train.shape

		if self.model_name == 'Logist Regression':
			return ModelCreation().create_LogisticRegression(input_shape, self.num_classes, use_proto=True)

		elif self.model_name == 'DNN':
			return ModelCreation().create_DNN(input_shape, self.num_classes, use_proto=False)

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

		if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
			#self.set_proto(parameters)
			self.set_parameters_to_model(parameters)

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

						loss_value = self.train_step(x_batch_train, y_batch_train)
						loss_train_history.append(loss_value)

					# Display metrics at the end of each epoch.
					train_acc = float(self.train_acc_metric.result())
					print("Training acc over epoch: %.4f" % (float(train_acc),))

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





			#print("agregar")
			#self.protos = self.agg_func(self.protos)

			print("passou")
			#print(self.protos)

			avg_loss_train = float(np.mean(loss_train_history))
			avg_acc_train = float(np.mean(acc_train_history))

			print("loss media: ", avg_loss_train)
			print("acc media: ", avg_acc_train)

			# ========================================================================================
			trained_parameters = self.model.get_weights()
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
			'cid': self.cid
		}

		# protos_result = self.dict_to_numpy(self.protos)
		#
		# print("Resultados proto: ", protos_result)

		# return protos_result, 10, fit_response
		print("finalizou")
		return trained_parameters, len(self.x_train), fit_response
	def dict_to_numpy(self, data):

		list = []

		for key in data:

			list.append(data[key])
		#print("converte: ", np.array(list))
		return np.array(list)

	def evaluate(self, parameters, config):

		self.set_parameters_to_model(parameters)
		#loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)

		acc_history = []
		loss_history = []

		for x_batch_val, y_batch_val in self.val_dataset:
			val_loss = self.test_step(x_batch_val, y_batch_val)
			loss_history.append(val_loss)
		val_acc = self.val_acc_metric.result()
		acc_history.append(val_acc)
		self.val_acc_metric.reset_states()

		avg_loss_test = float(np.mean(loss_history))
		avg_acc_test = float(np.mean(acc_history))

		#print("ola: ", loss, accuracy)
		print("ola 2: ", avg_loss_test, avg_acc_test)

		size_of_parameters = sum(map(sys.getsizeof, parameters))

		filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/evaluate_client.csv"
		data = [config['round'], self.cid, size_of_parameters, avg_loss_test, avg_acc_test]

		self._write_output(filename=filename,
						   data=data)

		evaluation_response = {
			"cid": self.cid,
			"accuracy": float(avg_acc_test)
		}

		return avg_loss_test, len(self.x_test), evaluation_response

	def agg_func(self, protos):
		"""
		Returns the average of the weights.
		"""
		size = len(protos)
		for key in protos:
			protos[key] = np.sum(protos[key], axis=0)/size

		return protos

