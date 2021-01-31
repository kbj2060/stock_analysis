import rootpath
import sys
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import FinanceDataReader as fdr
from src.scraping import Scraping
from src.utils import report_error
import pandas as pd
import datetime
import os
import time


class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(320, 238)
        dialog.setWindowOpacity(2.0)
        self.update = QtWidgets.QPushButton(dialog)
        self.update.setEnabled(True)
        self.update.setGeometry(QtCore.QRect(10, 20, 101, 41))
        self.update.setCheckable(False)
        self.update.setObjectName("update")
        self.load = QtWidgets.QPushButton(dialog)
        self.load.setGeometry(QtCore.QRect(10, 100, 101, 41))
        self.load.setObjectName("load")
        self.check = QtWidgets.QPushButton(dialog)
        self.check.setGeometry(QtCore.QRect(10, 180, 101, 41))
        self.check.setObjectName("check")
        self.label = QtWidgets.QLabel(dialog)
        self.label.setGeometry(QtCore.QRect(130, 20, 171, 201))
        self.label.setObjectName("label")

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "STOCK ASISTANT"))
        self.update.setText(_translate("dialog", "DATA UPDATE"))
        self.load.setText(_translate("dialog", "LOAD TOP"))
        self.check.setText(_translate("dialog", "GRAPH"))

class Worker(QThread):
    process = pyqtSignal(str)
    finished = pyqtSignal()
    threadactive = False

    def run(self):
        while True:
            self._scraping()
            self.msleep(500)

    def _scraping(self):
        root = rootpath.detect()
        df_krx = fdr.StockListing('KRX')
        codes = list(df_krx['Symbol'])
        keywords = list(df_krx['Name'])
        for code, keyword in zip(codes, keywords):
            if self.threadactive:
                try:
                    if not os.path.exists(root + '/stock/{0}/{0}.csv'.format(keyword)):
                        continue
                    sc = Scraping(code, keyword)
                    data = pd.read_csv(root + '/stock/{0}/{0}.csv'.format(keyword))
                    res = sc.get_current_data(code, data)
                    if res.empty:
                        self.process.emit('{0} is already updated!'.format(keyword))
                        self.msleep(500)
                        continue
                    res = res.loc[~res.index.duplicated(keep='first')]
                    res.to_csv(sc.make_csv(keyword))
                    self.process.emit('{0} is completed!'.format(keyword))
                    self.msleep(500)
                except Exception as e:
                    print('SCRAPING ERROR')
                    error = '[{0}] {1} Error. {2} \n'.format(datetime.datetime.now(), keyword, e)
                    report_error(error, 'scraping.err')
                    continue


    def stop(self):
        self.threadactive = False

# 멀티 프로세싱 알아볼 것
class MyApp(QWidget, Ui_dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.root = rootpath.detect()

        self.thread = QThread()
        self.thread.start()

        self.worker = Worker()
        self.worker.process.connect(self.show_process)
        self.update.clicked.connect(self.scraping)
        self.check.clicked.connect(self.graph)
        self.load.clicked.connect(self.predict)
        self.worker.start()

    @pyqtSlot(bool)
    def scraping(self, data):
        if data == False:
           self.worker.threadactive = True

    @pyqtSlot(str)
    def show_process(self, data):
        self.label.setText(data)

    def stop_thread(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()

    def predict(self, data):
       pass

    def graph(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())