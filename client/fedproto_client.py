from client.client_base import ClientBase
import flwr as fl
import numpy as np
import tensorflow as tf
import os
import time
import sys
import copy

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)\




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
		self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
		# Instantiate an optimizer to train the model.
		self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
		# Instantiate a loss function.
		self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.global_protos = None
		self.protos = {i: [] for i in range(self.num_classes)}
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
		grads = tape.gradient(loss_value, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
		self.train_acc_metric.update_state(y, logits)
		return loss_value

	@tf.function
	def test_step(self, x, y):
		val_logits = self.model(x, training=False)
		self.val_acc_metric.update_state(y, val_logits)

	def modify_dataset(self):

		self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
		self.x_train = None
		self.y_train = None
		self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

		self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
		self.x_test = None
		self.y_test = None
		self.val_dataset = self.val_dataset.batch(self.batch_size)

	def get_parameters(self, config):
		return self.model.get_weights()

	def get_parameters_of_model(self):
		return self.model.get_weights()

	def set_proto(self, protos):
		self.protos = protos



	def fit(self, parameters, config):
		selected_clients = []
		trained_parameters = []
		selected = 0

		loss_history = []
		acc_history = []

		if config['selected_clients'] != '':
			selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]

		start_time = time.process_time()
		# print(config)

		if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
			self.set_proto(parameters)

			selected = 1
			#history = self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.local_epochs)
			#========================================================================================
			for epoch in range(self.local_epochs):
				print("\nStart of epoch %d" % (epoch,))

				# Iterate over the batches of the dataset.
				for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
					#loss_value = float(self.train_step(x_batch_train, y_batch_train))

					with tf.GradientTape() as tape:
						logits = self.model(x_batch_train, training=True)
						loss_value = self.loss_fn(y_batch_train, logits)
					grads = tape.gradient(loss_value, self.model.trainable_weights)
					self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
					self.train_acc_metric.update_state(y_batch_train, logits)
					loss_value = float(loss_value)

					rep = self.model.proto
					loss_history.append(loss_value)
					# # Log every 200 batches.
					# if step % 200 == 0:
					# 	print(
					# 		"Training loss (for one batch) at step %d: %.4f"
					# 		% (step, float(loss_value))
					# 	)
					# 	print("Seen so far: %d samples" % ((step + 1) * batch_size))

					if self.global_protos != None:
						proto_new = {i: [] for i in range(self.num_classes)}
						for i, yy in enumerate(y_batch_train):
							y_c = int(yy)
							proto_new[y_c] = self.global_protos[y_c]

						data = tf.constant([proto_new[key] for key in proto_new])
						print("entre")
						print(data)
						print("representacao")
						print(rep)
						loss_value += float(self.loss_mse(data, rep) )* self.lamda
						print("perda")

					for i, yy in enumerate(y_batch_train):
						#print("Item antes: ", yy)
						y_c = int(yy)
						# print("Item: ", y_c, " i: ", i)
						# print("Linha: ", tf.gather_nd(rep, tf.constant([[i]])))
						self.protos[y_c].append(tf.gather_nd(rep, tf.constant([[i]])))
					print("mani")
				# Display metrics at the end of each epoch.
				train_acc = self.train_acc_metric.result()
				acc_history.append(train_acc)
				print("Training acc over epoch: %.4f" % (float(train_acc),))

				# Reset training metrics at the end of each epoch
				self.train_acc_metric.reset_states()


				# Run a validation loop at the end of each epoch.
				for step, (x_batch_test, y_batch_test) in enumerate(self.val_dataset):
					#self.test_step(x_batch_train, y_batch_train)
					val_logits = self.model(x_batch_test, training=False)
					self.val_acc_metric.update_state(y_batch_test, val_logits)
				print("validou")
				val_acc = self.val_acc_metric.result()
				self.val_acc_metric.reset_states()
				print("Validation acc: %.4f" % (float(val_acc),))
				print("Time taken: %.2fs" % (time.time() - start_time))

				#acc_history = self.train_acc_metric.result()
				self.train_acc_metric.reset_states()


			print("agregar")
			self.protos = self.agg_func(self.protos)

			print("passou")

			avg_loss_train = np.mean(loss_history)
			avg_acc_train = np.mean(acc_history)

			print("loss media: ", avg_loss_train)
			print("acc media: ", avg_acc_train)

			# ========================================================================================

			#trained_parameters = self.model.get_weights()

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

		return self.protos, len(self.x_train), fit_response

	def evaluate(self, parameters, config):

		self.set_parameters_to_model(parameters)
		loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
		size_of_parameters = sum(map(sys.getsizeof, parameters))

		filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/evaluate_client.csv"
		data = [config['round'], self.cid, size_of_parameters, loss, accuracy]

		self._write_output(filename=filename,
						   data=data)

		evaluation_response = {
			"cid": self.cid,
			"accuracy": float(accuracy)
		}

		return loss, len(self.x_test), evaluation_response

	def agg_func(self, protos):
		"""
		Returns the average of the weights.
		"""
		size = len(protos)
		for key in protos:
			protos[key] = np.sum(protos[key], axis=0)/size

		return protos

