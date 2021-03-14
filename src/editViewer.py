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
import os

import torch
import torch.nn as nn

from test import run_gui
from options.test_options import TestOptions
from utils.calculate_3Dheatmap import calculate_3Dheatmap

parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 12, 14, 15, 16, 12, 18, 19, 20]

class Viewer(QMainWindow):
	def __init__(self):
		super(Viewer, self).__init__()
		self.initUI()

	def initUI(self):
		exitAct = QAction(QIcon('exit.png'), '&Exit', self)
		exitAct.setShortcut('Ctrl+Q')
		exitAct.setStatusTip('Exit application')
		exitAct.triggered.connect(qApp.quit)

		selectAct = QAction('Select Sequence', self)
		selectAct.setShortcut('Ctrl+S')
		selectAct.setStatusTip('Select Sequence')
		selectAct.triggered.connect(self.openFileNameDialog)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(exitAct)
		fileMenu.addAction(selectAct)

		self.resize(800, 600)
		self.center()
		self.setWindowTitle('EditViewer')

		# Top HBox
		top_vbox = QVBoxLayout()
		topbottom_hbox = QHBoxLayout()
		self.name_txt = QLineEdit()
		self.name_txt.setReadOnly(True)

		self.idx_txt = QLineEdit()
		#self.idx_txt.setReadOnly(True)
		self.idx_txt.returnPressed.connect(self.anyAction)
		self.start_txt = QLineEdit()
		self.end_txt = QLineEdit()
		select_botton = QPushButton('Select')
		select_botton.clicked.connect(self.selectAction)
		topbottom_hbox.addWidget(self.idx_txt)
		topbottom_hbox.addWidget(self.start_txt)
		topbottom_hbox.addWidget(self.end_txt)
		topbottom_hbox.addWidget(select_botton)

		top_vbox.addWidget(self.name_txt)
		top_vbox.addLayout(topbottom_hbox)

		# Middle HBox
		middle_hbox = QHBoxLayout()
		self.GLViewer = gl.GLViewWidget()
		self.GLViewer.qglColor(QtCore.Qt.white)
		self.GLViewer.renderText(0, 0, 0, 'This is a test')
		middle_hbox.addWidget(self.GLViewer)

		# Bottom HBox
		bottom_hbox = QHBoxLayout()
		# Left
		prevButton = QPushButton('<< Prev')
		prevButton.clicked.connect(self.prevAction)
		prevButton.setShortcut("Left")
		nextButton = QPushButton('Next >>')
		nextButton.clicked.connect(self.nextAction)
		nextButton.setShortcut("Right")
		deleteButton = QPushButton('Delete/Undelete')
		deleteButton.clicked.connect(self.deleteAction)
		deleteButton.setShortcut("D")
		saveButton = QPushButton('Save')
		saveButton.clicked.connect(self.saveAction)
		self.isDeleteLabel = QLabel()
		self.isDeleteLabel.setText('Not Removed')
		self.isDeleteLabel.setAlignment(QtCore.Qt.AlignCenter)
		self.isDeleteLabel.setStyleSheet("border: 2px solid black; color: black")
		bottom_hbox.addWidget(prevButton)
		bottom_hbox.addWidget(nextButton)
		bottom_hbox.addWidget(self.isDeleteLabel)
		bottom_hbox.addWidget(deleteButton)
		bottom_hbox.addWidget(saveButton)

		# Set layout
		layout = QVBoxLayout()
		layout.addLayout(top_vbox)
		layout.addLayout(middle_hbox)
		layout.addLayout(bottom_hbox)

		widget = QWidget()
		widget.setLayout(layout)
		self.setCentralWidget(widget)

		self.current_item = 0

		self.show()

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def openFileNameDialog(self):
		self.name_txt.clear()
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.file, _ = QFileDialog.getOpenFileName(self,"Select Sequence", "/home/cz/cs/PG19/dataset/lafan/test_set","All Files (*);;PKL Files (*.pkl)", options=options)
		self.name_txt.setText(self.file)
		self.getItems(self.file)

	def getItems(self, file):
		with open(file, 'rb') as f:
			data = pickle.load(f, encoding='latin1')
		self.X = data['X']
		self.Q = data['Q']
		self.rv = data['rv']
		self.X = np.asarray(self.X)
		self.Q = np.asarray(self.Q)
		self.rv = np.asarray(self.rv)
		self.size = self.X.shape[0]
		self.isDeleted = np.zeros(self.size, dtype=bool)
		self.drawItem()

	def prevAction(self):
		self.current_item = (self.current_item - 1) % self.size
		self.GLViewer.clear()
		self.drawItem()

	def nextAction(self):
		self.current_item = (self.current_item + 1) % self.size
		self.GLViewer.clear()
		self.drawItem()

	def anyAction(self):
		idx = int(self.idx_txt.text())
		self.current_item = idx
		self.GLViewer.clear()
		self.drawItem()

	def selectAction(self):
		start_idx = self.start_txt.text()
		end_idx = self.end_txt.text()

		self.isDeleted = np.zeros(self.size, dtype=bool)
		if start_idx != '':
			self.isDeleted[:int(start_idx)+1] = True
		if end_idx != '':
			self.isDeleted[int(end_idx):] = True
		self.saveAction()

	def deleteAction(self):
		if self.isDeleted[self.current_item] == True:
			self.isDeleted[self.current_item] = False
		else:
			self.isDeleted[self.current_item] = True

		self.GLViewer.clear()
		self.drawItem()

	def saveAction(self):
		isMaintained = np.invert(self.isDeleted)
		self.X = self.X[isMaintained]
		self.Q = self.Q[isMaintained]
		self.rv = self.rv[isMaintained]

		data = {
			'Q': self.Q,
			'X': self.X,
			'rv': self.rv
		}

		output_path = '/home/cz/cs/PG19/dataset/lafan/test_set_new'
		if not os.path.exists(output_path):
			os.makedirs(output_path)

		file = Path(self.file)
		file = file.parts[-1]
		with open(os.path.join(output_path, file), 'wb') as f:
			pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
		print('Success!')

	def showMessage(self):
		QMessageBox.about(self, 'Warning', 'start or end cannot be empty!')

	def drawItem(self):
		# update label
		self.idx_txt.setText(str(self.current_item))
		if self.isDeleted[self.current_item] == True:
			self.isDeleteLabel.setStyleSheet("border: 2px solid red; color: red")
			self.isDeleteLabel.setText('Removed')
		else:
			self.isDeleteLabel.setStyleSheet("border: 2px solid black; color: black")
			self.isDeleteLabel.setText('Not Removed')
		# init GLViewer
		r = R.from_euler('xyz', [90, 0, 180], degrees=True)
		rr = r.as_matrix()
		data = self.X[self.current_item].copy()
		root = np.zeros((1, 3))
		data = np.concatenate((root, data), axis=0)
		data = np.matmul(data, rr)
		self.GLViewer.opts['distance'] = 2
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

			self.GLViewer.addItem(m1)

			# self.lines = gl.GLLinePlotItem(
			# 	pos = pts,
			# 	color = pg.glColor((255, 0, 0)),
			# 	width=5
			# )
			# self.GLViewer.addItem(self.lines)

		gz = gl.GLGridItem()
		gz.translate(0, 0, -1)
		self.GLViewer.addItem(gz)
		self.points = gl.GLScatterPlotItem(
			pos = data,
			color = pg.glColor((0, 255, 0)),
			size=5
			)
		self.GLViewer.addItem(self.points)


def main():
	app = QApplication(sys.argv)
	viewer = Viewer()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()