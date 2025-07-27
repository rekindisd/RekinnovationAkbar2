import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class ProjectMLModels:
    def __init__(self, data):
        self.data = data.copy()
        self.delay_model = None
        self.failure_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.model_results = {}
        
    def prepare_features(self, df):
        """Prepare features for ML models"""
        # Select relevant columns
        feature_cols = [
            'Project', 'Open Actions', 'Total Actions', 'Images', 'Comments', 
            'Documents', 'OverDue', 'Type', 'Days_Since_Creation', 
            'Action_Completion_Rate', 'Type_Risk_Level', 'Location_Depth',
            'Project_Type_Diversity', 'Project_Location_Diversity'
        ]
        
        # Keep only existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                # Fit on all unique values including 'Unknown'
                unique_values = list(X[col].unique()) + ['Unknown']
                le.fit(unique_values)
                self.label_encoders[col] = le
            
            # Transform values, handling unseen categories
            X[col] = X[col].astype(str)
            X[col] = X[col].apply(lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown')
            X[col] = self.label_encoders[col].transform(X[col])
        
        # Convert boolean columns to int
        boolean_cols = [col for col in X.columns if X[col].dtype == 'bool']
        for col in boolean_cols:
            X[col] = X[col].astype(int)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        return X
    
    def train_models(self):
        """Train ML models for delay and failure prediction"""
        
        if len(self.data) < 10:
            return {"error": "Insufficient data for training"}
        
        # Prepare features
        X = self.prepare_features(self.data)
        self.feature_columns = X.columns.tolist()
        
        results = {}
        
        # Train delay prediction model
        if 'Is_Delayed' in self.data.columns:
            y_delay = self.data['Is_Delayed']
            
            if len(y_delay.unique()) > 1 and len(X) > 5:
                try:
                    delay_results = self._train_single_model(X, y_delay, 'delay')
                    results.update(delay_results)
                except Exception as e:
                    print(f"Error training delay model: {e}")
        
        # Train failure prediction model
        if 'Is_Potential_Failure' in self.data.columns:
            y_failure = self.data['Is_Potential_Failure']
            
            if len(y_failure.unique()) > 1 and len(X) > 5:
                try:
                    failure_results = self._train_single_model(X, y_failure, 'failure')
                    results.update(failure_results)
                except Exception as e:
                    print(f"Error training failure model: {e}")
        
        # Feature importance
        if self.delay_model is not None:
            importance = self.delay_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance
        
        self.model_results = results
        return results
    
    def _train_single_model(self, X, y, model_type):
        """Train a single ML model"""
        results = {}
        
        # Split data
        if len(X) < 10:
            # Use the entire dataset for training if too small
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division='warn'),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division='warn'),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division='warn')
        }
        
        # Store model and results
        if model_type == 'delay':
            self.delay_model = rf_model
            results['delay_metrics'] = metrics
        elif model_type == 'failure':
            self.failure_model = rf_model
            results['failure_metrics'] = metrics
        
        return results
    
    def predict(self, input_data):
        """Make predictions for new data"""
        predictions = {}
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Add missing columns with default values
        for col in self.feature_columns:
            if col not in input_df.columns:
                if col in ['Images', 'Comments', 'Documents', 'OverDue']:
                    input_df[col] = False
                elif col in ['Open Actions', 'Total Actions', 'Project']:
                    input_df[col] = 0
                else:
                    input_df[col] = 0
        
        # Prepare features
        try:
            X_input = self.prepare_features(input_df)
            X_input_scaled = self.scaler.transform(X_input)
            
            # Delay prediction
            if self.delay_model is not None:
                delay_prob = self.delay_model.predict_proba(X_input_scaled)[0]
                predictions['delay_probability'] = delay_prob[1] if len(delay_prob) > 1 else delay_prob[0]
            
            # Failure prediction
            if self.failure_model is not None:
                failure_prob = self.failure_model.predict_proba(X_input_scaled)[0]
                predictions['failure_probability'] = failure_prob[1] if len(failure_prob) > 1 else failure_prob[0]
                
        except Exception as e:
            print(f"Error making predictions: {e}")
            predictions['delay_probability'] = 0.5
            predictions['failure_probability'] = 0.3
        
        return predictions
    
    def get_model_results(self):
        """Get stored model results"""
        return self.model_results
