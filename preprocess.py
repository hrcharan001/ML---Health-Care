import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import IsolationForest

DROP_COLS = [
    'Patient_ID', 'Name', 'Doctor', 'Hospital',
    'Age', 'Room_Number', 'Blood_Type', 'Gender',
]

NUM_COLS = [
    'Billing_Amount', 'stay_duration', 'difference_bill', 'avg_billing_per_day',
]

ONE_HOT_CAT_COLS = [
    'Medical_Condition', 'Insurance_Provider', 'Admission_Type', 'Medication',
]

ORDINAL_CAT_COLS = ['Test_Results']

class FeatureEngineering(BaseEstimator,TransformerMixin):

  def __init__(self, grp_cols = ('Medical_Condition', 'Test_Results')):
    self.grp_cols = list(grp_cols)

  def base_features(self,X):
    df = X.copy()
    df.drop(columns=DROP_COLS, inplace=True)
    df['Date_of_Admission'] = pd.to_datetime(df['Date_of_Admission'])
    df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'])
    df['stay_duration'] = (df['Discharge_Date'] - df['Date_of_Admission']).dt.days
    df.drop(columns=['Date_of_Admission', 'Discharge_Date'], inplace=True)
    df['billing_per_day'] = df['Billing_Amount'] / df['stay_duration']
    df['billing_per_day'] = df['billing_per_day'].fillna(0)
    return df


  def fit(self, X, y=None):
    df = self.base_features(X)
    self.grp_means = df.groupby(self.grp_cols)['billing_per_day'].mean()
    self.global_mean = df['billing_per_day'].mean()
    return self

  def transform(self, X, y=None):
    df = self.base_features(X)
    avg = df.set_index(self.grp_cols).index.map(self.grp_means)
    avg = pd.Series(avg, index=df.index)
    avg = avg.fillna(self.global_mean)
    df['avg_billing_per_day'] = avg
    df['difference_bill'] = df['billing_per_day'] - df['avg_billing_per_day']
    df.drop(columns=['billing_per_day'], inplace=True)
    return df

def build_pipeline():
  preprocessor = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ONE_HOT_CAT_COLS),
        ('num', 'passthrough', NUM_COLS),
        ('ordinal',
         OrdinalEncoder(categories=[['Normal', 'Inconclusive', 'Abnormal']]),
         ORDINAL_CAT_COLS),
    ])

  pipeline = Pipeline([
      ('feature_engineering', FeatureEngineering()),
      ('preprocessor', preprocessor),
      ('model', IsolationForest(
          n_estimators=200,
          contamination=0.02,
          random_state=42
      ))
  ])

  return pipeline

def train_model(csv_path, output_path):
  df = pd.read_csv(csv_path)
  pipeline = build_pipeline()
  pipeline.fit(df)
  with open(output_path, 'wb') as f:
    joblib.dump(pipeline, f)
  return pipeline
