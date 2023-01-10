# Load libraries
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

'''
Source data: https://www.kaggle.com/datasets/whenamancodes/predict-diabities
'''

class MachineLearningModel(object):
    
    def __init__(self):
        pass

    '''
    Data exploration and basic statistcs
    '''
    def data_exploration(self)->pd.DataFrame:
        contains_null_values = False
        df_raw = pd.read_csv('../data/diabetes.csv')
        print()
        print('1) First rows of the dataframe')
        print(df_raw.head())
        contains_null_values = self.check_null_values(df_raw)
        print()
        print('2) Check null values on dataframe (Fix necessary if returns is True)')
        print(f'Null Values: {contains_null_values}')
        print()
        print('3) Dataframe shape')
        print(f'Shape: {df_raw.shape}')
        print()
        print('4) Datum basic statistics')
        self.basic_statistcs(df_raw)
        return df_raw

    def check_null_values(self,df)->bool:
        check_nan = df.isnull().values.any()
        return check_nan
    
    def basic_statistcs(self,df)->any:
        print(df.describe())    
    
    '''
    Model definition
    '''    
    def decision_tree_classifier(self)->object:
        print()
        print('5) Creating model object DecisionTreeClassifier')
        model_object = DecisionTreeClassifier()
        return model_object

    '''
    Model trainning and test
    '''
    def train_test_definition(self,df,model_object)->list:
        print()
        print('6) Training and testing the model')
        y = df.Outcome
        X = df.drop(['Outcome'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
        model_fit = model_object.fit(X_train,y_train) # Train Decision Tree Classifer
        y_pred = model_fit.predict(X_test) #Predict the response for test dataset
        return y_test, y_pred, model_fit

    def evaluate_model(self,y_test,y_pred)->any:
        print()
        print('7) Evaluating model test result')
        result = round(metrics.accuracy_score(y_test, y_pred),2)
        print(f'DecisionTreeClassifier Accuracy: {result*100}%')
   
    '''
    Model publish
    '''
    def publish_model(self,model_fit)->any:
        print()
        print('8) Publishing fited model')
        pickle.dump(model_fit, open('predictor.pkl', 'wb'))

    '''
    Main function
    '''
    def execute(self)->any:        
        df = self.data_exploration()
        model_object = self.decision_tree_classifier()
        y_test, y_pred, model_fit = self.train_test_definition(df,model_object)
        self.evaluate_model(y_test, y_pred)
        self.publish_model(model_fit)
        print()


if __name__ == '__main__':
    model = MachineLearningModel()
    model.execute()