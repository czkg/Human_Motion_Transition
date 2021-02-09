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


parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 12, 14, 15, 16, 12, 18, 19, 20]

data = [[0, 0, 0],
	   [ 9.90281498e-02,  2.86642184e-03,  2.54880054e-02],
       [ 1.09747259e-01, -4.12289731e-01,  1.47049395e-02],
       [ 1.12829961e-01, -8.08685292e-01, -6.66201461e-02],
       [ 1.09636183e-01, -8.68968918e-01,  8.71749215e-02],
       [-1.01798598e-01, -1.71968101e-03,  9.92563001e-03],
       [-1.00573482e-01, -4.17119470e-01,  1.51872611e-02],
       [-1.30742496e-01, -8.08325887e-01, -8.38104882e-02],
       [-1.44408761e-01, -8.69565217e-01,  6.90299528e-02],
       [ 2.93072297e-04,  6.64606974e-02, -2.33676185e-02],
       [-3.48237828e-03,  1.85740287e-01, -8.85014849e-03],
       [-7.28442695e-03,  3.02306588e-01,  8.28439868e-03],
       [-1.53806977e-02,  5.45524534e-01,  4.88452829e-02],
       [-4.21779361e-03,  6.56997293e-01,  5.76231476e-02],
       [ 4.49601995e-02,  4.91437176e-01,  3.16319471e-02],
       [ 1.52646153e-01,  4.91423205e-01,  2.74827088e-02],
       [ 4.65195510e-01,  4.51578528e-01,  2.04376599e-02],
       [ 7.03856108e-01,  4.61727872e-01,  4.97297167e-02],
       [-6.89572200e-02,  4.89716487e-01,  1.92219725e-02],
       [-1.73719718e-01,  4.91919711e-01, -5.94695073e-03],
       [-4.83772276e-01,  4.89057694e-01, -6.23721586e-02],
       [-7.20709563e-01,  5.27081231e-01, -4.41019709e-02]]
data = np.asarray(data)


class PathWindow(QWidget):
	def __init__(self):
		super(PathWindow, self).__init__()
		self.initUI()

	def initUI(self):
		self.resize(300, 30)
		self.center()
		self.setWindowTitle('Load Path')

		# Push Button
		fvae_btn = QPushButton('f-VAE')
		fvae_btn.clicked.connect(self.openFileNameDialogFVAE)
		rtn_btn = QPushButton('rtn')
		rtn_btn.clicked.connect(self.openFileNameDialogRTN)

		# Text Edit
		self.fvae_txt = QTextEdit()
		self.rtn_txt = QTextEdit()

		# Layout
		fvae_hbox = QHBoxLayout()
		rtn_hbox = QHBoxLayout()
		fvae_hbox.addWidget(fvae_btn)
		fvae_hbox.addWidget(self.fvae_txt)
		fvae_hbox.addStretch(1)
		rtn_hbox.addWidget(rtn_btn)
		rtn_hbox.addWidget(self.rtn_txt)
		rtn_hbox.addStretch(1)

		layout = QVBoxLayout()
		layout.addLayout(fvae_hbox)
		layout.addLayout(rtn_hbox)
		self.setLayout(layout)


	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def openFileNameDialogFVAE(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.fvae_name, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;PTH Files (*.pth)", options=options)
		self.fvae_txt.setText(self.fvae_name)

	def openFileNameDialogRTN(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.rtn_name, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;PTH Files (*.pth)", options=options)
		self.rtn_txt.setText(self.rtn_name)

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


class Viewer(QMainWindow):
	def __init__(self):
		super(Viewer, self).__init__()
		self.initUI()

	def initUI(self):
		exitAct = QAction(QIcon('exit.png'), '&Exit', self)
		exitAct.setShortcut('Ctrl+Q')
		exitAct.setStatusTip('Exit application')
		exitAct.triggered.connect(qApp.quit)

		pathAct = QAction('Path', self)
		pathAct.setShortcut('Ctrl+P')
		pathAct.setStatusTip('Load path')
		pathAct.triggered.connect(self.showPathWindow)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(exitAct)
		fileMenu.addAction(pathAct)

		self.resize(1280, 600)
		self.center()
		self.setWindowTitle('TransitionViewer')

		# top HBox
		top_hbox = QHBoxLayout()
		selectButton = QPushButton('Select Past/Target')
		selectButton.clicked.connect(self.showSelectWindow)
		transitionButton = QPushButton('Generate Transition')
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

		self.data = data

		# Bottom HBox
		bottom_hbox = QHBoxLayout()
		# Left
		leftLeftButton = QPushButton('<< Prev')
		leftLeftButton.clicked.connect(self.leftLeftAction)
		leftMiddleButton = QPushButton('Auto')
		leftRightButton = QPushButton('Next >>')
		leftRightButton.clicked.connect(self.leftRightAction)
		left_bottom_hbox = QHBoxLayout()
		left_bottom_hbox.addWidget(leftLeftButton)
		left_bottom_hbox.addWidget(leftMiddleButton)
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

	def showPathWindow(self):
		self.pathWindow = PathWindow()
		self.pathWindow.show()

	def showSelectWindow(self):
		self.selectWindow = SelectWindow()
		self.selectWindow.show()
		self.selectWindow.procDone.connect(self.getItems)

	def getItems(self, pitems, titems):
		self.pastItems = []
		self.targetItems = []
		for i in pitems:
			with open(i, 'rb') as f:
				data = pickle.load(f, encoding='latin1')
			root = np.zeros((1, 3))
			data = np.concatenate((root, data), axis=0)
			self.pastItems.append(data)
		for i in titems:
			with open(i, 'rb') as f:
				data = pickle.load(f, encoding='latin1')
			root = np.zeros((1, 3))
			data = np.concatenate((root, data), axis=0)
			self.targetItems.append(data)

		self.pastItems = np.asarray(self.pastItems)
		self.targetItems = np.asarray(self.targetItems)
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


def main():
	app = QApplication(sys.argv)
	viewer = Viewer()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()