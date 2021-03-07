import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.Qt3DCore import *
from PyQt5.Qt3DExtras import *
from PyQt5 import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from test import run_gui
from options.test_options import TestOptions
from utils.calculate_3Dheatmap import calculate_3Dheatmap

parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 12, 14, 15, 16, 12, 18, 19, 20]

class fVAE:
	def __init__(self):
		self.name = 'vaedmp'
		self.model = 'vaedmp'
		self.checkpoints_dir = None
		self.sigma = '0.05'
		self.dim_heatmap = '64'
		self.num_joints = '21'
		self.z_dim = '32'
		self.u_dim = '32'
		self.hidden_dim = '128'
		self.noise_dim = '32'
		self.transform_dim = '64'
		self.init_type = 'normal'
		self.init_gain = '0.8'
		self.output_path = None

class RTN():
	def __init__(self):
		self.name = 'rtn'
		self.model = 'rtn'
		self.checkpoints_dir = None
		self.sigma = '0.05'
		self.num_joints = '21'
		self.x_dim = '32'
		self.hidden_dim = '512'
		self.z_dim = '128'
		self.epoch = '100'
		self.init_type = 'xavier'
		self.init_gain = '0.8'
		self.output_path = None

class SetValueWindow(QWidget):
	procDone = QtCore.pyqtSignal(str, int)
	def __init__(self, idx):
		super(SetValueWindow, self).__init__()
		self.idx = idx
		self.initUI()

	def initUI(self):
		self.resize(200, 60)
		self.center()
		self.setWindowTitle('Set Value')

		self.lineEdit = QLineEdit()
		button = QPushButton('OK')
		button.clicked.connect(self.getTxt)
		layout = QVBoxLayout()
		layout.addWidget(self.lineEdit)
		layout.addWidget(button)
		self.setLayout(layout)

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def getTxt(self):
		self.procDone.emit(self.lineEdit.text(), self.idx)
		self.close()


class fVAEWindow(QWidget):
	procDone = QtCore.pyqtSignal(list)
	def __init__(self):
		super(fVAEWindow, self).__init__()
		self.fvae = fVAE()
		self.initUI()

	def initUI(self):
		self.resize(300, 30)
		self.center()
		self.setWindowTitle('Set fVAE')

		# Push Button
		checkpoints_dir_btn = QPushButton('checkpoints dir')
		output_path_btn = QPushButton('output dir')
		checkpoints_dir_btn.setFixedHeight(25)
		output_path_btn.setFixedHeight(25)
		checkpoints_dir_btn.setFixedWidth(120)
		output_path_btn.setFixedWidth(120)
		checkpoints_dir_btn.clicked.connect(lambda: self.openFileNameDialog())
		output_path_btn.clicked.connect(lambda: self.openDirectory())

		# with default values
		name_btn = QPushButton('name')
		model_btn = QPushButton('model')
		sigma_btn = QPushButton('sigma')
		dim_heatmap_btn = QPushButton('dim_heatmap')
		num_joints_btn = QPushButton('num_joints')
		z_dim_btn = QPushButton('z_dim')
		u_dim_btn = QPushButton('u_dim')
		hidden_dim_btn = QPushButton('hidden_dim_btn')
		noise_dim_btn = QPushButton('noise_dim_btn')
		transform_dim_btn = QPushButton('transform_dim')
		init_type_btn = QPushButton('init_type')
		init_gain_btn = QPushButton('init_gain')

		name_btn.setFixedHeight(25)
		model_btn.setFixedHeight(25)
		sigma_btn.setFixedHeight(25)
		dim_heatmap_btn.setFixedHeight(25)
		num_joints_btn.setFixedHeight(25)
		z_dim_btn.setFixedHeight(25)
		u_dim_btn.setFixedHeight(25)
		hidden_dim_btn.setFixedHeight(25)
		noise_dim_btn.setFixedHeight(25)
		transform_dim_btn.setFixedHeight(25)
		init_type_btn.setFixedHeight(25)
		init_gain_btn.setFixedHeight(25)

		name_btn.setFixedWidth(120)
		model_btn.setFixedWidth(120)
		sigma_btn.setFixedWidth(120)
		dim_heatmap_btn.setFixedWidth(120)
		num_joints_btn.setFixedWidth(120)
		z_dim_btn.setFixedWidth(120)
		u_dim_btn.setFixedWidth(120)
		hidden_dim_btn.setFixedWidth(120)
		noise_dim_btn.setFixedWidth(120)
		transform_dim_btn.setFixedWidth(120)
		init_type_btn.setFixedWidth(120)
		init_gain_btn.setFixedWidth(120)

		name_btn.clicked.connect(lambda: self.setValue(0))
		model_btn.clicked.connect(lambda: self.setValue(1))
		sigma_btn.clicked.connect(lambda: self.setValue(2))
		dim_heatmap_btn.clicked.connect(lambda: self.setValue(3))
		num_joints_btn.clicked.connect(lambda: self.setValue(4))
		z_dim_btn.clicked.connect(lambda: self.setValue(5))
		u_dim_btn.clicked.connect(lambda: self.setValue(6))
		hidden_dim_btn.clicked.connect(lambda: self.setValue(7))
		noise_dim_btn.clicked.connect(lambda: self.setValue(8))
		transform_dim_btn.clicked.connect(lambda: self.setValue(9))
		init_type_btn.clicked.connect(lambda: self.setValue(10))
		init_gain_btn.clicked.connect(lambda: self.setValue(11))

		accept_btn = QPushButton('OK')
		accept_btn.clicked.connect(self.getAllOpts)

		# Display text
		self.checkpoints_dir_dtxt = QLineEdit()
		self.output_path_dtxt = QLineEdit()
		self.checkpoints_dir_dtxt.setReadOnly(True)
		self.output_path_dtxt.setReadOnly(True)

		# with default values
		self.name_dtxt = QLineEdit()
		self.model_dtxt = QLineEdit()
		self.sigma_dtxt = QLineEdit()
		self.dim_heatmap_dtxt = QLineEdit()
		self.num_joints_dtxt = QLineEdit()
		self.z_dim_dtxt = QLineEdit()
		self.u_dim_dtxt = QLineEdit()
		self.hidden_dim_dtxt = QLineEdit()
		self.noise_dim_dtxt = QLineEdit()
		self.transform_dim_dtxt = QLineEdit()
		self.init_type_dtxt = QLineEdit()
		self.init_gain_dtxt = QLineEdit()

		self.name_dtxt.setText(self.fvae.name)
		self.model_dtxt.setText(self.fvae.model)
		self.sigma_dtxt.setText(self.fvae.sigma)
		self.dim_heatmap_dtxt.setText(self.fvae.dim_heatmap)
		self.num_joints_dtxt.setText(self.fvae.num_joints)
		self.z_dim_dtxt.setText(self.fvae.z_dim)
		self.u_dim_dtxt.setText(self.fvae.u_dim)
		self.hidden_dim_dtxt.setText(self.fvae.hidden_dim)
		self.noise_dim_dtxt.setText(self.fvae.noise_dim)
		self.transform_dim_dtxt.setText(self.fvae.transform_dim)
		self.init_type_dtxt.setText(self.fvae.init_type)
		self.init_gain_dtxt.setText(self.fvae.init_gain)

		self.name_dtxt.setReadOnly(True)
		self.model_dtxt.setReadOnly(True)
		self.sigma_dtxt.setReadOnly(True)
		self.dim_heatmap_dtxt.setReadOnly(True)
		self.num_joints_dtxt.setReadOnly(True)
		self.z_dim_dtxt.setReadOnly(True)
		self.u_dim_dtxt.setReadOnly(True)
		self.hidden_dim_dtxt.setReadOnly(True)
		self.noise_dim_dtxt.setReadOnly(True)
		self.transform_dim_dtxt.setReadOnly(True)
		self.init_type_dtxt.setReadOnly(True)
		self.init_gain_dtxt.setReadOnly(True)

		# Layout
		name_hbox = QHBoxLayout()
		name_hbox.addWidget(name_btn)
		name_hbox.addWidget(self.name_dtxt)

		model_hbox = QHBoxLayout()
		model_hbox.addWidget(model_btn)
		model_hbox.addWidget(self.model_dtxt)
		
		checkpoints_dir_hbox = QHBoxLayout()
		checkpoints_dir_hbox.addWidget(checkpoints_dir_btn)
		checkpoints_dir_hbox.addWidget(self.checkpoints_dir_dtxt)
		
		sigma_hbox = QHBoxLayout()
		sigma_hbox.addWidget(sigma_btn)
		sigma_hbox.addWidget(self.sigma_dtxt)
		
		dim_heatmap_hbox = QHBoxLayout()
		dim_heatmap_hbox.addWidget(dim_heatmap_btn)
		dim_heatmap_hbox.addWidget(self.dim_heatmap_dtxt)
		
		num_joints_hbox = QHBoxLayout()
		num_joints_hbox.addWidget(num_joints_btn)
		num_joints_hbox.addWidget(self.num_joints_dtxt)
		
		z_dim_hbox = QHBoxLayout()
		z_dim_hbox.addWidget(z_dim_btn)
		z_dim_hbox.addWidget(self.z_dim_dtxt)
		
		u_dim_hbox = QHBoxLayout()
		u_dim_hbox.addWidget(u_dim_btn)
		u_dim_hbox.addWidget(self.u_dim_dtxt)

		hidden_dim_hbox = QHBoxLayout()
		hidden_dim_hbox.addWidget(hidden_dim_btn)
		hidden_dim_hbox.addWidget(self.hidden_dim_dtxt)
		
		noise_dim_hbox = QHBoxLayout()
		noise_dim_hbox.addWidget(noise_dim_btn)
		noise_dim_hbox.addWidget(self.noise_dim_dtxt)
		
		transform_dim_hbox = QHBoxLayout()
		transform_dim_hbox.addWidget(transform_dim_btn)
		transform_dim_hbox.addWidget(self.transform_dim_dtxt)
		
		init_type_hbox = QHBoxLayout()
		init_type_hbox.addWidget(init_type_btn)
		init_type_hbox.addWidget(self.init_type_dtxt)

		init_gain_hbox = QHBoxLayout()
		init_gain_hbox.addWidget(init_gain_btn)
		init_gain_hbox.addWidget(self.init_gain_dtxt)

		output_path_hbox = QHBoxLayout()
		output_path_hbox.addWidget(output_path_btn)
		output_path_hbox.addWidget(self.output_path_dtxt)

		layout = QVBoxLayout()
		layout.addLayout(name_hbox)
		layout.addLayout(model_hbox)
		layout.addLayout(checkpoints_dir_hbox)
		layout.addLayout(sigma_hbox)
		layout.addLayout(dim_heatmap_hbox)
		layout.addLayout(num_joints_hbox)
		layout.addLayout(z_dim_hbox)
		layout.addLayout(u_dim_hbox)
		layout.addLayout(hidden_dim_hbox)
		layout.addLayout(noise_dim_hbox)
		layout.addLayout(transform_dim_hbox)
		layout.addLayout(init_type_hbox)
		layout.addLayout(init_gain_hbox)
		layout.addLayout(output_path_hbox)
		layout.addWidget(accept_btn)
		self.setLayout(layout)

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def setValue(self, idx):
		self.setValueWindow = SetValueWindow(idx)
		self.setValueWindow.show()
		self.setValueWindow.procDone.connect(self.updateTxt)

	def updateTxt(self, txt, idx):
		if idx == 0:
			self.name_dtxt.setText(txt)
		elif idx == 1:
			self.model_dtxt.setText(txt)
		elif idx == 2:
			self.sigma_dtxt.setText(txt)
		elif idx == 3:
			self.dim_heatmap_dtxt.setText(txt)
		elif idx == 4:
			self.num_joints_dtxt.setText(txt)
		elif idx == 5:
			self.z_dim_dtxt.setText(txt)
		elif idx == 6:
			self.u_dim_dtxt.setText(txt)
		elif idx == 7:
			self.hidden_dim_dtxt.setText(txt)
		elif idx == 8:
			self.noise_dim_dtxt.setText(txt)
		elif idx == 9:
			self.transform_dim_dtxt.setText(txt)
		elif idx == 10:
			self.init_type_dtxt.setText(txt)
		elif idx == 11:
			self.init_gain_dtxt.setText(txt)


	def openFileNameDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		name, _ = QFileDialog.getOpenFileName(self,"Select files", "/home/cz/cs/PG19/results","All Files (*);;PTH Files (*.pth)", options=options)
		self.checkpoints_dir_dtxt.setText(name)

	def openDirectory(self):
		path = QFileDialog.getExistingDirectory(self, 'Select directory', '/home/cz/cs/PG19/res')
		self.output_path_dtxt.setText(path)

	def showMessage(self):
		QMessageBox.about(self, 'Warning', 'Opts cannot be empty!')

	def getAllOpts(self):
		params = []
		checkpoints_dir = self.checkpoints_dir_dtxt.text()
		output_path = self.output_path_dtxt.text()

		name = self.name_dtxt.text()
		model = self.model_dtxt.text()
		sigma = self.sigma_dtxt.text()
		dim_heatmap = self.dim_heatmap_dtxt.text()
		num_joints = self.num_joints_dtxt.text()
		z_dim = self.z_dim_dtxt.text()
		u_dim = self.u_dim_dtxt.text()
		hidden_dim = self.hidden_dim_dtxt.text()
		noise_dim = self.noise_dim_dtxt.text()
		transform_dim = self.transform_dim_dtxt.text()
		init_type = self.init_type_dtxt.text()
		init_gain = self.init_gain_dtxt.text()

		params.append(name)
		params.append(model)
		params.append(checkpoints_dir)
		params.append(sigma)
		params.append(dim_heatmap)
		params.append(num_joints)
		params.append(z_dim)
		params.append(u_dim)
		params.append(hidden_dim)
		params.append(noise_dim)
		params.append(transform_dim)
		params.append(init_type)
		params.append(init_gain)
		params.append(output_path)

		if len(checkpoints_dir) == 0 or len(output_path) == 0:
			self.showMessage()
		else:
			self.procDone.emit(params)
			self.close()


class RTNWindow(QWidget):
	procDone = QtCore.pyqtSignal(list)
	def __init__(self):
		super(RTNWindow, self).__init__()
		self.rtn = RTN()
		self.initUI()

	def initUI(self):
		self.resize(300, 30)
		self.center()
		self.setWindowTitle('Set RTN')

		# Push Button
		checkpoints_dir_btn = QPushButton('checkpoints dir')
		output_path_btn = QPushButton('output dir')
		checkpoints_dir_btn.setFixedHeight(25)
		output_path_btn.setFixedHeight(25)
		checkpoints_dir_btn.setFixedWidth(120)
		output_path_btn.setFixedWidth(120)
		checkpoints_dir_btn.clicked.connect(lambda: self.openFileNameDialog())
		output_path_btn.clicked.connect(lambda: self.openDirectory())

		# with default values
		name_btn = QPushButton('name')
		model_btn = QPushButton('model')
		sigma_btn = QPushButton('sigma')
		num_joints_btn = QPushButton('num_joints')
		x_dim_btn = QPushButton('x_dim')
		z_dim_btn = QPushButton('z_dim')
		hidden_dim_btn = QPushButton('hidden_dim_btn')
		init_type_btn = QPushButton('init_type')
		init_gain_btn = QPushButton('init_gain')

		name_btn.setFixedHeight(25)
		model_btn.setFixedHeight(25)
		sigma_btn.setFixedHeight(25)
		num_joints_btn.setFixedHeight(25)
		x_dim_btn.setFixedHeight(25)
		z_dim_btn.setFixedHeight(25)
		hidden_dim_btn.setFixedHeight(25)
		init_type_btn.setFixedHeight(25)
		init_gain_btn.setFixedHeight(25)

		name_btn.setFixedWidth(120)
		model_btn.setFixedWidth(120)
		sigma_btn.setFixedWidth(120)
		num_joints_btn.setFixedWidth(120)
		x_dim_btn.setFixedWidth(120)
		z_dim_btn.setFixedWidth(120)
		hidden_dim_btn.setFixedWidth(120)
		init_type_btn.setFixedWidth(120)
		init_gain_btn.setFixedWidth(120)

		name_btn.clicked.connect(lambda: self.setValue(0))
		model_btn.clicked.connect(lambda: self.setValue(1))
		sigma_btn.clicked.connect(lambda: self.setValue(2))
		num_joints_btn.clicked.connect(lambda: self.setValue(3))
		x_dim_btn.clicked.connect(lambda: self.setValue(4))
		z_dim_btn.clicked.connect(lambda: self.setValue(5))
		hidden_dim_btn.clicked.connect(lambda: self.setValue(6))
		init_type_btn.clicked.connect(lambda: self.setValue(7))
		init_gain_btn.clicked.connect(lambda: self.setValue(8))

		accept_btn = QPushButton('OK')
		accept_btn.clicked.connect(self.getAllOpts)

		# Text Edit
		self.checkpoints_dir_dtxt = QLineEdit()
		self.output_path_dtxt = QLineEdit()
		self.checkpoints_dir_dtxt.setReadOnly(True)
		self.output_path_dtxt.setReadOnly(True)

		# with default values
		self.name_dtxt = QLineEdit()
		self.model_dtxt = QLineEdit()
		self.sigma_dtxt = QLineEdit()
		self.num_joints_dtxt = QLineEdit()
		self.x_dim_dtxt = QLineEdit()
		self.hidden_dim_dtxt = QLineEdit()
		self.z_dim_dtxt = QLineEdit()
		self.init_type_dtxt = QLineEdit()
		self.init_gain_dtxt = QLineEdit()

		self.name_dtxt.setText(self.rtn.name)
		self.model_dtxt.setText(self.rtn.model)
		self.sigma_dtxt.setText(self.rtn.sigma)
		self.num_joints_dtxt.setText(self.rtn.num_joints)
		self.x_dim_dtxt.setText(self.rtn.x_dim)
		self.z_dim_dtxt.setText(self.rtn.z_dim)
		self.hidden_dim_dtxt.setText(self.rtn.hidden_dim)
		self.init_type_dtxt.setText(self.rtn.init_type)
		self.init_gain_dtxt.setText(self.rtn.init_gain)

		self.name_dtxt.setReadOnly(True)
		self.model_dtxt.setReadOnly(True)
		self.sigma_dtxt.setReadOnly(True)
		self.num_joints_dtxt.setReadOnly(True)
		self.x_dim_dtxt.setReadOnly(True)
		self.z_dim_dtxt.setReadOnly(True)
		self.hidden_dim_dtxt.setReadOnly(True)
		self.init_type_dtxt.setReadOnly(True)
		self.init_gain_dtxt.setReadOnly(True)

		# Layout
		name_hbox = QHBoxLayout()
		name_hbox.addWidget(name_btn)
		name_hbox.addWidget(self.name_dtxt)

		model_hbox = QHBoxLayout()
		model_hbox.addWidget(model_btn)
		model_hbox.addWidget(self.model_dtxt)
		
		checkpoints_dir_hbox = QHBoxLayout()
		checkpoints_dir_hbox.addWidget(checkpoints_dir_btn)
		checkpoints_dir_hbox.addWidget(self.checkpoints_dir_dtxt)
		
		sigma_hbox = QHBoxLayout()
		sigma_hbox.addWidget(sigma_btn)
		sigma_hbox.addWidget(self.sigma_dtxt)
		
		num_joints_hbox = QHBoxLayout()
		num_joints_hbox.addWidget(num_joints_btn)
		num_joints_hbox.addWidget(self.num_joints_dtxt)
		
		x_dim_hbox = QHBoxLayout()
		x_dim_hbox.addWidget(x_dim_btn)
		x_dim_hbox.addWidget(self.x_dim_dtxt)
		
		z_dim_hbox = QHBoxLayout()
		z_dim_hbox.addWidget(z_dim_btn)
		z_dim_hbox.addWidget(self.z_dim_dtxt)

		hidden_dim_hbox = QHBoxLayout()
		hidden_dim_hbox.addWidget(hidden_dim_btn)
		hidden_dim_hbox.addWidget(self.hidden_dim_dtxt)
		
		init_type_hbox = QHBoxLayout()
		init_type_hbox.addWidget(init_type_btn)
		init_type_hbox.addWidget(self.init_type_dtxt)

		init_gain_hbox = QHBoxLayout()
		init_gain_hbox.addWidget(init_gain_btn)
		init_gain_hbox.addWidget(self.init_gain_dtxt)

		output_path_hbox = QHBoxLayout()
		output_path_hbox.addWidget(output_path_btn)
		output_path_hbox.addWidget(self.output_path_dtxt)


		layout = QVBoxLayout()
		layout.addLayout(name_hbox)
		layout.addLayout(model_hbox)
		layout.addLayout(checkpoints_dir_hbox)
		layout.addLayout(sigma_hbox)
		layout.addLayout(num_joints_hbox)
		layout.addLayout(x_dim_hbox)
		layout.addLayout(z_dim_hbox)
		layout.addLayout(hidden_dim_hbox)
		layout.addLayout(init_type_hbox)
		layout.addLayout(init_gain_hbox)
		layout.addLayout(output_path_hbox)
		layout.addWidget(accept_btn)
		self.setLayout(layout)


	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def setValue(self, idx):
		self.setValueWindow = SetValueWindow(idx)
		self.setValueWindow.show()
		self.setValueWindow.procDone.connect(self.updateTxt)

	def updateTxt(self, txt, idx):
		if idx == 0:
			self.name_dtxt.setText(txt)
		elif idx == 1:
			self.model_dtxt.setText(txt)
		elif idx == 2:
			self.sigma_dtxt.setText(txt)
		elif idx == 3:
			self.num_joints_dtxt.setText(txt)
		elif idx == 4:
			self.x_dim_dtxt.setText(txt)
		elif idx == 5:
			self.z_dim_dtxt.setText(txt)
		elif idx == 6:
			self.hidden_dim_dtxt.setText(txt)
		elif idx == 7:
			self.init_type_dtxt.setText(txt)
		elif idx == 8:
			self.init_gain_dtxt.setText(txt)

	def openFileNameDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		name, _ = QFileDialog.getOpenFileName(self,"Select files", "/home/cz/cs/PG19/results","All Files (*);;PTH Files (*.pth)", options=options)
		self.checkpoints_dir_dtxt.setText(name)

	def openDirectory(self):
		path = QFileDialog.getExistingDirectory(self, 'Select directory', '/home/cz/cs/PG19/res')
		self.output_path_dtxt.setText(path)

	def showMessage(self):
		QMessageBox.about(self, 'Warning', 'Opts cannot be empty!')

	def getAllOpts(self):
		params = []
		checkpoints_dir = self.checkpoints_dir_dtxt.text()
		output_path = self.output_path_dtxt.text()

		name = self.name_dtxt.text()
		model = self.model_dtxt.text()
		sigma = self.sigma_dtxt.text()
		num_joints = self.num_joints_dtxt.text()
		x_dim = self.x_dim_dtxt.text()
		z_dim = self.z_dim_dtxt.text()
		hidden_dim = self.hidden_dim_dtxt.text()
		init_type = self.init_type_dtxt.text()
		init_gain = self.init_gain_dtxt.text()

		params.append(name)
		params.append(model)
		params.append(checkpoints_dir)
		params.append(sigma)
		params.append(num_joints)
		params.append(x_dim)
		params.append(z_dim)
		params.append(hidden_dim)
		params.append(init_type)
		params.append(init_gain)
		params.append(output_path)

		if len(checkpoints_dir) == 0 or len(output_path) == 0:
			self.showMessage()
		else:
			self.procDone.emit(params)
			self.close()

class SelectWindow(QWidget):
	procDone = QtCore.pyqtSignal(np.ndarray, np.ndarray)
	def __init__(self):
		super(SelectWindow, self).__init__()
		self.initUI()

	def initUI(self):
		self.resize(600, 400)
		self.center()
		self.setWindowTitle('Select Frames')

		add_past_btn = QPushButton('Add')
		remove_past_btn = QPushButton('Remove')
		self.pastListWidget = QListWidget()
		self.pastListWidget.setDragDropMode(QAbstractItemView.InternalMove)
		add_past_btn.clicked.connect(self.openFileNamesDialog)
		remove_past_btn.clicked.connect(self.removeItem)

		add_target_btn = QPushButton('Select Target')
		self.targetListWidget = QListWidget()
		add_target_btn.clicked.connect(self.openFileNameDialog)

		left_vbox = QVBoxLayout()
		left_vbox.addWidget(self.pastListWidget)
		left_bottom_hbox = QHBoxLayout()
		left_bottom_hbox.addWidget(add_past_btn)
		left_bottom_hbox.addWidget(remove_past_btn)
		left_vbox.addLayout(left_bottom_hbox)

		right_vbox = QVBoxLayout()
		right_vbox.addWidget(self.targetListWidget)
		right_vbox.addWidget(add_target_btn)

		top_hbox = QHBoxLayout()
		top_hbox.addLayout(left_vbox)
		top_hbox.addLayout(right_vbox)

		bottom_hbox = QHBoxLayout()
		accept_btn = QPushButton('OK')
		cancel_btn = QPushButton('Cancel')
		accept_btn.clicked.connect(self.getAllItems)
		cancel_btn.clicked.connect(self.close)
		bottom_hbox.addWidget(accept_btn)
		bottom_hbox.addWidget(cancel_btn)

		layout = QVBoxLayout()
		layout.addLayout(top_hbox)
		layout.addLayout(bottom_hbox)
		self.setLayout(layout)

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def openFileNamesDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.past_names, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "../dataset/lafan/test_poses","All Files (*);;PKL Files (*.pkl)", options=options)
		for name in self.past_names:
			if not self.pastListWidget.findItems(name, QtCore.Qt.MatchFixedString | QtCore.Qt.MatchCaseSensitive):
				self.pastListWidget.addItem(name)

	def openFileNameDialog(self):
		self.targetListWidget.clear()
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.target_name, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;PKL Files (*.pkl)", options=options)
		self.targetListWidget.addItem(self.target_name)

	def removeItem(self):
		self.pastListWidget.takeItem(self.pastListWidget.currentRow())

	def showMessage(self):
		QMessageBox.about(self, 'Warning', 'Please set 10 past frames and 1 target frame!')

	def getAllItems(self):
		pastItems = []
		targetItems = []
		for index in range(self.pastListWidget.count()):
			pastItems.append(self.pastListWidget.item(index).text())
		self.pastItems = np.asarray(pastItems)
		for index in range(self.targetListWidget.count()):
			targetItems.append(self.targetListWidget.item(index).text())
		self.targetItems = np.asarray(targetItems)

		if len(pastItems) != 10 or len(targetItems) != 1:
			self.showMessage()
		else:
			self.procDone.emit(self.pastItems, self.targetItems)
			self.close()


class scriptWrapper():
	def __init__(self):
		self.fvae = fVAE()
		self.rtn = RTN()
		self.opt = TestOptions().parse()

	def run_fvae(self, past, target):
		"""Run fVAE
		Parameters:
			past: ndarray of past frames
			target: target frame
		"""
		self.gather_fvae_opts()
		past = self.toHeatmap(past, self.opt.dim_heatmap, self.opt.sigma)
		target = self.toHeatmap(target, self.opt.dim_heatmap, self.opt.sigma)
		past = past[np.newaxis, :]
		target = target[np.newaxis, :]
		pastLatentItems = run_gui(self.opt, torch.tensor(past))
		targetLatentItems = run_gui(self.opt, torch.tensor(target))

		return pastLatentItems, targetLatentItems

	def run_rtn(self, past, target):
		"""Run RTN
		Parameters:
			past: ndarray of past frames
			target: target frame
		"""
		self.gather_rtn_opts()
		run_gui(self.opt, past, target)

	def toHeatmap(self, data, dim_heatmap, sigma):
		heatmaps = []
		for d in data:
			heatmaps.append(calculate_3Dheatmap(d, dim_heatmap, sigma))
		heatmaps = np.asarray(heatmaps)
		return heatmaps

	def gather_fvae_opts(self):
		self.opt.name = self.fvae.name
		self.opt.model = self.fvae.model
		# extract checkpoints_dir and epoch from original checkpoints_dir
		checkpoints_dir = Path(self.fvae.checkpoints_dir)
		epoch = checkpoints_dir.parts[-1]
		self.opt.checkpoints_dir = str(checkpoints_dir.parent.parent)
		self.opt.epoch = int(epoch[:epoch.find('_')])
		self.opt.sigma = float(self.fvae.sigma)
		self.opt.dim_heatmap = int(self.fvae.dim_heatmap)
		self.opt.num_joints = int(self.fvae.num_joints)
		self.opt.z_dim = int(self.fvae.z_dim)
		self.opt.u_dim = int(self.fvae.u_dim)
		self.opt.hidden_dim = int(self.fvae.hidden_dim)
		self.opt.noise_dim = int(self.fvae.noise_dim)
		self.opt.transform_dim = int(self.fvae.transform_dim)
		self.opt.init_type = self.fvae.init_type
		self.opt.init_gain = float(self.fvae.init_gain)
		self.opt.output_path = self.fvae.output_path

	def gather_rtn_opts(self):
		self.opt.name = self.rtn.name
		self.opt.model = self.rtn.model
		# extract checkpoints_dir and epoch from original checkpoints_dir
		checkpoints_dir = Path(self.rtn.checkpoints_dir)
		epoch = checkpoints_dir.parts[-1]
		self.opt.checkpoints_dir = str(checkpoints_dir.parent.parent)
		self.opt.epoch = int(epoch[:epoch.find('_')])
		self.opt.sigma = float(self.rtn.sigma)
		self.opt.num_joints = int(self.rtn.num_joints)
		self.opt.x_dim = int(self.rtn.x_dim)
		self.opt.z_dim = int(self.rtn.z_dim)
		self.opt.hidden_dim = int(self.rtn.hidden_dim)
		self.opt.init_type = self.rtn.init_type
		self.opt.init_gain = float(self.rtn.init_gain)
		self.opt.output_path = self.rtn.output_path


class Viewer(QMainWindow):
	def __init__(self):
		super(Viewer, self).__init__()
		self.sw = scriptWrapper()
		self.initUI()

	def initUI(self):
		exitAct = QAction(QIcon('exit.png'), '&Exit', self)
		exitAct.setShortcut('Ctrl+Q')
		exitAct.setStatusTip('Exit application')
		exitAct.triggered.connect(qApp.quit)

		fVAEAct = QAction('fVAE', self)
		fVAEAct.setShortcut('Ctrl+F')
		fVAEAct.setStatusTip('Set fVAE')
		fVAEAct.triggered.connect(self.showfVAEWindow)

		RTNAct = QAction('RTN', self)
		RTNAct.setShortcut('Ctrl+R')
		RTNAct.setStatusTip('Set RTN')
		RTNAct.triggered.connect(self.showRTNWindow)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(exitAct)
		fileMenu.addAction(fVAEAct)
		fileMenu.addAction(RTNAct)

		self.resize(1280, 600)
		self.center()
		self.setWindowTitle('TransitionViewer')

		# top HBox
		top_hbox = QHBoxLayout()
		selectButton = QPushButton('Select Past/Target')
		selectButton.clicked.connect(self.showSelectWindow)
		transitionButton = QPushButton('Generate Transition')
		transitionButton.clicked.connect(self.generate)
		top_hbox.addWidget(selectButton)
		top_hbox.addWidget(transitionButton)

		# Middle HBox
		middle_hbox = QHBoxLayout()
		self.leftGLViewer = gl.GLViewWidget()
		self.rightGLViewer = gl.GLViewWidget()
		self.middleGLViewer = gl.GLViewWidget()
		middle_hbox.addWidget(self.leftGLViewer)
		middle_hbox.addWidget(self.middleGLViewer)
		middle_hbox.addWidget(self.rightGLViewer)

		# Bottom HBox
		bottom_hbox = QHBoxLayout()
		# Left
		leftLeftButton = QPushButton('<< Prev')
		leftLeftButton.clicked.connect(self.leftLeftAction)
		self.leftMiddleButton = QPushButton('Auto')
		self.leftMiddleButton.clicked.connect(self.leftMiddleAction)
		leftRightButton = QPushButton('Next >>')
		leftRightButton.clicked.connect(self.leftRightAction)
		left_bottom_hbox = QHBoxLayout()
		left_bottom_hbox.addWidget(leftLeftButton)
		left_bottom_hbox.addWidget(self.leftMiddleButton)
		left_bottom_hbox.addWidget(leftRightButton)

		# Middle window
		middleLeftButton = QPushButton('<< Prev')
		middleMiddleButton = QPushButton('Auto')
		middleRightButton = QPushButton('Next >>')
		middle_bottom_hbox = QHBoxLayout()
		middle_bottom_hbox.addWidget(middleLeftButton)
		middle_bottom_hbox.addWidget(middleMiddleButton)
		middle_bottom_hbox.addWidget(middleRightButton)

		# Right window
		rightLeftButton = QPushButton('<< Prev')
		rightLeftButton.clicked.connect(self.rightLeftAction)
		rightMiddleButton = QPushButton('Auto')
		rightRightButton = QPushButton('Next >>')
		rightRightButton.clicked.connect(self.rightRightAction)
		right_bottom_hbox = QHBoxLayout()
		right_bottom_hbox.addWidget(rightLeftButton)
		right_bottom_hbox.addWidget(rightMiddleButton)
		right_bottom_hbox.addWidget(rightRightButton)

		bottom_hbox.addLayout(left_bottom_hbox)
		bottom_hbox.addLayout(middle_bottom_hbox)
		bottom_hbox.addLayout(right_bottom_hbox)

		# Set layout
		layout = QVBoxLayout()
		layout.addLayout(top_hbox)
		layout.addLayout(middle_hbox)
		layout.addLayout(bottom_hbox)

		widget = QWidget()
		widget.setLayout(layout)
		self.setCentralWidget(widget)

		self.current_past_item = 0
		self.current_target_item = 0

		self.show()

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def showfVAEWindow(self):
		self.fVAEWindow = fVAEWindow()
		self.fVAEWindow.show()
		self.fVAEWindow.procDone.connect(self.getfVAEOpts)

	def showRTNWindow(self):
		self.RTNWindow = RTNWindow()
		self.RTNWindow.show()
		self.RTNWindow.procDone.connect(self.getRTNOpts)

	def showSelectWindow(self):
		self.selectWindow = SelectWindow()
		self.selectWindow.show()
		self.selectWindow.procDone.connect(self.getItems)

	def getfVAEOpts(self, opts):
		self.sw.fvae.name = opts[0]
		self.sw.fvae.model = opts[1]
		self.sw.fvae.checkpoints_dir = opts[2]
		self.sw.fvae.sigma = opts[3]
		self.sw.fvae.dim_heatmap = opts[4]
		self.sw.fvae.num_joints = opts[5]
		self.sw.fvae.z_dim = opts[6]
		self.sw.fvae.u_dim = opts[7]
		self.sw.fvae.hidden_dim = opts[8]
		self.sw.fvae.noise_dim = opts[9]
		self.sw.fvae.transform_dim = opts[10]
		self.sw.fvae.init_type = opts[11]
		self.sw.fvae.init_gain = opts[12]
		self.sw.fvae.output_path = opts[13]

	def getRTNOpts(self, opts):
		self.sw.rtn.name = opts[0]
		self.sw.rtn.model = opts[1]
		self.sw.rtn.checkpoints_dir = opts[2]
		self.sw.rtn.sigma = opts[3]
		self.sw.rtn.num_joints = opts[4]
		self.sw.rtn.x_dim = opts[5]
		self.sw.rtn.z_dim = opts[6]
		self.sw.rtn.hidden_dim = opts[7]
		self.sw.rtn.init_type = opts[8]
		self.sw.rtn.init_gain = opts[9]
		self.sw.rtn.output_path = opts[10]

	def getItems(self, pitems, titems):
		self.pastItems = []
		self.targetItems = []
		self.pastOriginalItems = []
		self.targetOriginalItems = []
		for i in pitems:
			with open(i, 'rb') as f:
				data = pickle.load(f, encoding='latin1')
			self.pastOriginalItems.append(data)
			root = np.zeros((1, 3))
			data = np.concatenate((root, data), axis=0)
			self.pastItems.append(data)
		for i in titems:
			with open(i, 'rb') as f:
				data = pickle.load(f, encoding='latin1')
			self.targetOriginalItems.append(data)
			root = np.zeros((1, 3))
			data = np.concatenate((root, data), axis=0)
			self.targetItems.append(data)

		self.pastItems = np.asarray(self.pastItems)
		self.targetItems = np.asarray(self.targetItems)
		self.pastOriginalItems = np.asarray(self.pastOriginalItems)
		self.targetOriginalItems = np.asarray(self.targetOriginalItems)
		self.drawLeftItem()
		self.drawRightItem()

	def leftLeftAction(self):
		self.current_past_item = (self.current_past_item - 1) % 10
		self.leftGLViewer.clear()
		self.drawLeftItem()

	def leftRightAction(self):
		self.current_past_item = (self.current_past_item + 1) % 10
		self.leftGLViewer.clear()
		self.drawLeftItem()

	def leftMiddleAction(self):
		#TO DO!
		if self.leftMiddleButton.isChecked():
			self.leftMiddleButton.setStyleSheet('background-color : lightblue')
			self.leftMiddleButton.setCheckable(False)
			self.current_past_item = self.last_past_item
			self.leftGLViewer.clear()
			self.drawLeftItem()
		else:
			self.leftMiddleButton.setStyleSheet('background-color : lightgrey')
			self.leftMiddleButton.setCheckable(True)
			self.last_past_item = self.current_past_item
			self.current_past_item = 0

			while True:
				self.leftGLViewer.clear()
				self.drawLeftItem()
				self.current_past_item = (self.current_past_item + 1) % 10
				#loop = QtCore.QEventLoop()
				#QtCore.QTimer.singleShot(2000, loop)

	def rightLeftAction(self):
		self.current_target_item = (self.current_target_item - 1) % 1
		self.rightGLViewer.clear()
		self.drawRightItem()

	def rightRightAction(self):
		self.current_target_item = (self.current_target_item + 1) % 1
		self.rightGLViewer.clear()
		self.drawRightItem()

	def drawLeftItem(self):
		# init GLViewer
		r = R.from_euler('xyz', [90, 0, 180], degrees=True)
		rr = r.as_matrix()
		data = self.pastItems[self.current_past_item]
		data = np.matmul(data, rr)
		self.leftGLViewer.opts['distance'] = 2
		for i in range(1, 22):
			xx = (data[i][0], data[i][1], data[i][2])
			yy = (data[parents[i]][0], data[parents[i]][1], data[parents[i]][2])
			pts = np.array([xx, yy])

			center = (data[i] + data[parents[i]]) / 2.
			length = np.linalg.norm(data[i] - data[parents[i]]) / 2.
			radius = [0.02, 0.02]
			md = gl.MeshData.cylinder(rows=40, cols=40, radius=radius, length=2*length)

			m1 = gl.GLMeshItem(meshdata=md,
							   smooth=True,
							   color=(1, 0, 0.5, 1),
							   shader="balloon",
							   glOptions="additive")

			v = data[i] - data[parents[i]]
			theta = np.arctan2(v[1], v[0])
			phi = np.arctan2(np.linalg.norm(v[:2]), v[2])

			tr = pg.Transform3D()
			tr.translate(*data[parents[i]])
			tr.rotate(theta * 180 / np.pi, 0, 0, 1)
			tr.rotate(phi * 180 / np.pi, 0, 1, 0)
			tr.scale(1, 1, 1)
			tr.translate(0, 0, 0)
			m1.setTransform(tr)

			self.leftGLViewer.addItem(m1)

			# self.lines = gl.GLLinePlotItem(
			# 	pos = pts,
			# 	color = pg.glColor((255, 0, 0)),
			# 	width=5
			# )
			# self.leftGLViewer.addItem(self.lines)

		gz = gl.GLGridItem()
		gz.translate(0, 0, -1)
		self.leftGLViewer.addItem(gz)
		self.points = gl.GLScatterPlotItem(
			pos = data,
			color = pg.glColor((0, 255, 0)),
			size=5
			)
		self.leftGLViewer.addItem(self.points)


	def drawRightItem(self):
		# init GLViewer
		r = R.from_euler('xyz', [90, 0, 180], degrees=True)
		rr = r.as_matrix()
		data = self.targetItems[self.current_target_item]
		data = np.matmul(data, rr)
		self.rightGLViewer.opts['distance'] = 2
		for i in range(1, 22):
			xx = (data[i][0], data[i][1], data[i][2])
			yy = (data[parents[i]][0], data[parents[i]][1], data[parents[i]][2])
			pts = np.array([xx, yy])

			center = (data[i] + data[parents[i]]) / 2.
			length = np.linalg.norm(data[i] - data[parents[i]]) / 2.
			radius = [0.02, 0.02]
			md = gl.MeshData.cylinder(rows=40, cols=40, radius=radius, length=2*length)

			m1 = gl.GLMeshItem(meshdata=md,
							   smooth=True,
							   color=(1, 0, 0.5, 1),
							   shader="balloon",
							   glOptions="additive")

			v = data[i] - data[parents[i]]
			theta = np.arctan2(v[1], v[0])
			phi = np.arctan2(np.linalg.norm(v[:2]), v[2])

			tr = pg.Transform3D()
			tr.translate(*data[parents[i]])
			tr.rotate(theta * 180 / np.pi, 0, 0, 1)
			tr.rotate(phi * 180 / np.pi, 0, 1, 0)
			tr.scale(1, 1, 1)
			tr.translate(0, 0, 0)
			m1.setTransform(tr)

			self.rightGLViewer.addItem(m1)

			# self.lines = gl.GLLinePlotItem(
			# 	pos = pts,
			# 	color = pg.glColor((255, 0, 0)),
			# 	width=5
			# )
			# self.rightGLViewer.addItem(self.lines)

		gz = gl.GLGridItem()
		gz.translate(0, 0, -1)
		self.rightGLViewer.addItem(gz)
		self.points = gl.GLScatterPlotItem(
			pos = data,
			color = pg.glColor((0, 255, 0)),
			size=5
			)
		self.rightGLViewer.addItem(self.points)

	def drawMiddleItem(self):
		# init GLViewer
		r = R.from_euler('xyz', [90, 0, 180], degrees=True)
		rr = r.as_matrix()
		data = np.matmul(self.data, rr)
		self.middleGLViewer.opts['distance'] = 2
		for i in range(1, 22):
			xx = (data[i][0], data[i][1], data[i][2])
			yy = (data[parents[i]][0], data[parents[i]][1], data[parents[i]][2])
			pts = np.array([xx, yy])

			center = (data[i] + data[parents[i]]) / 2.
			length = np.linalg.norm(data[i] - data[parents[i]]) / 2.
			radius = [0.02, 0.02]
			md = gl.MeshData.cylinder(rows=40, cols=40, radius=radius, length=2*length)

			m1 = gl.GLMeshItem(meshdata=md,
							   smooth=True,
							   color=(1, 0, 0.5, 1),
							   shader="balloon",
							   glOptions="additive")

			v = data[i] - data[parents[i]]
			theta = np.arctan2(v[1], v[0])
			phi = np.arctan2(np.linalg.norm(v[:2]), v[2])

			tr = pg.Transform3D()
			tr.translate(*data[parents[i]])
			tr.rotate(theta * 180 / np.pi, 0, 0, 1)
			tr.rotate(phi * 180 / np.pi, 0, 1, 0)
			tr.scale(1, 1, 1)
			tr.translate(0, 0, 0)
			m1.setTransform(tr)

			self.middleGLViewer.addItem(m1)

			# self.lines = gl.GLLinePlotItem(
			# 	pos = pts,
			# 	color = pg.glColor((255, 0, 0)),
			# 	width=5
			# )
			# self.middleGLViewer.addItem(self.lines)

		gz = gl.GLGridItem()
		gz.translate(0, 0, -1)
		self.middleGLViewer.addItem(gz)
		self.points = gl.GLScatterPlotItem(
			pos = data,
			color = pg.glColor((0, 255, 0)),
			size=5
			)
		self.middleGLViewer.addItem(self.points)

	def generate(self):
		self.pastLatentItems, self.targetLatentItems = self.sw.run_fvae(self.pastOriginalItems, self.targetOriginalItems)
		self.transitions = self.sw.run_rtn(self.pastLatentItems, self.targetLatentItems)


def main():
	app = QApplication(sys.argv)
	viewer = Viewer()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()