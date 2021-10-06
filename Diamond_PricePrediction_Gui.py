import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from pricePrediction_Gui_python import Ui_MainWindow #Import Qt design
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

class Diamond(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.InitWindow()

    def InitWindow(self):
        #setting the styles of the labels
        self.ui.label_preprocess.setStyleSheet("color : rgb(0,128,64);")
        self.ui.train_label.setStyleSheet("color : rgb(0,128,64);")
        self.ui.split_succes_label.setStyleSheet("color : rgb(0,128,64);")

        #signal slot connections of buttons
        self.ui.loadData_btn.clicked.connect(self.getData)
        self.ui.preprocess_btn.clicked.connect(self.dataPreprocessing)
        self.ui.split_btn.clicked.connect(self.splitDataset)
        self.ui.train_btn.clicked.connect(self.trainModel)
        self.ui.predict_btn.clicked.connect(self.predictPrice)
        self.ui.plot_btn.clicked.connect(self.dataAnalysis)
        self.ui.readData_btn.clicked.connect(self.readData)

        #set slider and spinbox
        self.ui.split_slider.valueChanged.connect(self.ui.split_spinbox.setValue)
        self.ui.split_spinbox.valueChanged.connect(self.ui.split_slider.setValue)

    def progressBar(self , sec):
        for i in range(101):
            time.sleep(sec)
            self.ui.progressBar.setValue(i)
        
    def getData(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/Semih/Desktop/NYP-GUI',
                                            'CSV files (*.csv)')
        self.dataPath = fname[0]
        self.ui.dataPath.setText(self.dataPath)

    def readData(self):
        self.data = pd.read_csv(self.dataPath) #Read data
        self.data = self.data[(self.data[['x', 'y', 'z']] != 0).all(axis=1)] #Either of length, width or dept of diamonds can not be 0.
        self.ui.dataPath.setText("diamonds.csv was successfully read !")
        self.ui.dataPath.setStyleSheet("color : rgb(0,128,64);")

    def dataPreprocessing(self):
        self.data = self.data.drop_duplicates().reset_index(drop=True) #Drop duplicate values
        dropList = ['depth', 'table', 'x', 'y', 'z']
        self.data = self.data.drop(dropList, axis=1)
        self.data['price'] = self.data.price.astype(float)

        #Converting categorical data into numeric form
        label_encoder = LabelEncoder()
        self.data["cut"] = label_encoder.fit_transform(self.data["cut"])
        self.data["clarity"] = label_encoder.fit_transform(self.data["clarity"])
        self.data["color"] = label_encoder.fit_transform(self.data["color"])

        self.x = self.data.drop(['price'], axis=1)
        self.y = self.data['price']

        self.ui.label_preprocess.setText("Successfully completed !")

    def splitDataset(self):
        #Train-Test Split
        test_size = (self.ui.split_slider.value())/100
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        self.ui.split_succes_label.setText("Successful !")

    def trainModel(self):
        #Linear Regression
        if self.ui.linear_regression.isChecked():
            self.model_LR = LinearRegression()
            self.model_LR.fit(self.X_train, self.y_train)

            predict_LR = self.model_LR.predict(self.X_test)
            self.acc_LR = r2_score(self.y_test, predict_LR)
            
            self.progressBar(0.01)
            self.ui.train_label.setText("Linear Regression - Model successfully trained !")

          	
            return self.model_LR , self.acc_LR

        #Decision Tree
        elif self.ui.decision_tree.isChecked():
            self.model_DT = DecisionTreeRegressor()
            self.model_DT.fit(self.X_train, self.y_train)
            predict_DT = self.model_DT.predict(self.X_test)
            self.acc_DT = r2_score(self.y_test, predict_DT)

            self.progressBar(0.01)
            self.ui.train_label.setText("Decision Tree - Model successfully trained !")
            return self.model_DT , self.acc_DT

        #Random Forest
        elif self.ui.random_forest.isChecked():
            self.model_RF = RandomForestRegressor(n_estimators=50)
            self.model_RF.fit(self.X_train, self.y_train)
            predict_RF = self.model_RF.predict(self.X_test)
            self.acc_RF = r2_score(self.y_test, predict_RF)

            self.progressBar(0.03)
            self.ui.train_label.setText("Random Forest - Model successfully trained !")
            return self.model_RF, self.acc_RF

        #KNN
        elif self.ui.knn.isChecked():
            self.model_KNN = KNeighborsRegressor(n_neighbors=5)
            self.model_KNN.fit(self.X_train, self.y_train)
            predict_KNN = self.model_KNN.predict(self.X_test)
            self.acc_KNN = r2_score(self.y_test, predict_KNN)

            self.progressBar(0.02)
            self.ui.train_label.setText("KNN - Model successfully trained !")
            return self.model_KNN, self.acc_KNN

        else: QMessageBox.warning(self, "Warning", "Please select model !")

    def predictPrice(self):
        self.ui.train_label.setText(" ")
        carat_value = self.ui.carat_value.text()
        cut_value = self.ui.cut_value.text()
        color_value = self.ui.color_value.text()
        clarity_value = self.ui.clarity_value.text()

        model , acc = self.trainModel()
        price = model.predict([[carat_value, int(cut_value), int(clarity_value), int(color_value)]])[0] * 0.1
        self.ui.train_label.setText("Prediction successful !")

        x = '-' * 62
        self.ui.results_area.setText(f"                ***** Results *****"
                                      f"\n{x}"
                                      f"\nCarat: {carat_value}"
                                      f"\n{x}"
                                      f"\nCut: {cut_value}"
                                      f"\n{x}"
                                      f"\nColor: {color_value}"
                                      f"\n{x}"
                                      f"\nClarity: {clarity_value}"
                                      f"\n{x}"
                                      f"\n\nPredicted Price: {round(price,3)}"
                                      f"\n{x}"
                                      f"\nModel Accuracy: {round(acc,3)}"
                                      f"\n{x}")

    def dataAnalysis(self):
        if self.ui.countplot_cut.isChecked():
            ax = sns.countplot(x="cut", data=self.data, palette="Set1")

        elif self.ui.countplot_color.isChecked():
            ax = sns.countplot(x="color", data=self.data, palette="Set1")

        elif self.ui.countplot_clarity.isChecked():
            ax = sns.countplot(x="clarity", data=self.data, palette="Set1")

        elif self.ui.catplot_cutCarat.isChecked():
            ax = sns.catplot(x="cut",y="carat",data=self.data)

        elif self.ui.distplot_carat.isChecked():
            ax = sns.distplot(self.data.carat)

        elif self.ui.distribution_carat.isChecked():
            plt.hist(self.data['carat'], bins=20, color='b')
            plt.xlabel('Carat Weight'), plt.ylabel('Frequency'), plt.title('Distribution of diamond carat weight')

        elif self.ui.distribution_price.isChecked():
            plt.hist(self.data['price'], bins=20, color='r')
            plt.xlabel('Diamond Price'), plt.ylabel('Frequency'), plt.title('Distribution of diamond price')

        elif self.ui.correlation_matrix.isChecked():
            corr = self.data.corr()
            plt.figure(figsize=(7,7))
            ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True, annot=True)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_ylim(len(corr) + 0.5, -0.5);

        else: QMessageBox.warning(self, "Warning", "Please select EDA !")

        plt.show()

app = QApplication([])
window = Diamond()
window.show()
app.exec_()