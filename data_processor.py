import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, data):
        self.data = data.copy()
    
    def process_data(self):
        """Process and clean the construction data"""
        df = self.data.copy()
        
        # Convert date columns
        date_columns = ['Created', 'Status Changed']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
        
        # Convert boolean columns
        boolean_columns = ['OverDue', 'Images', 'Comments', 'Documents']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
        
        # Convert numeric columns
        numeric_columns = ['Open Actions', 'Total Actions', 'Project']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _create_derived_features(self, df):
        """Create derived features for ML models"""
        
        # Time-based features
        if 'Created' in df.columns and 'Status Changed' in df.columns:
            df['Days_Since_Creation'] = (df['Status Changed'] - df['Created']).dt.days
            df['Days_Since_Creation'] = df['Days_Since_Creation'].fillna(0)
        
        # Status-based features
        if 'Status' in df.columns:
            # Create binary delay indicator
            delay_statuses = ['Open / Ongoing Works', 'Opened']
            df['Is_Delayed'] = df['Status'].isin(delay_statuses).astype(int)
            
            # Create failure indicator (forms that are stuck or problematic)
            failure_statuses = ['Open / Ongoing Works']  # Forms that remain open for too long
            df['Is_Potential_Failure'] = df['Status'].isin(failure_statuses).astype(int)
        
        # Action-based features
        if 'Open Actions' in df.columns and 'Total Actions' in df.columns:
            df['Action_Completion_Rate'] = np.where(
                df['Total Actions'] > 0,
                (df['Total Actions'] - df['Open Actions']) / df['Total Actions'],
                1.0
            )
        
        # Project complexity indicator
        project_complexity = df.groupby('Project').agg({
            'Type': 'nunique',
            'Location': 'nunique'
        }).rename(columns={'Type': 'Project_Type_Diversity', 'Location': 'Project_Location_Diversity'})
        
        df = df.merge(project_complexity, left_on='Project', right_index=True, how='left')
        
        # Form type encoding
        if 'Type' in df.columns:
            type_risk_map = {
                'Safety Forms': 3,  # High risk
                'Quality 00 General': 2,  # Medium risk
                'Quality 02 Architectural': 2,
                'Site Management': 1,  # Low risk
                'Subcontractor Inspections': 1,
                'Permits': 3  # High risk
            }
            df['Type_Risk_Level'] = df['Type'].map(type_risk_map).fillna(1)
        
        # Location complexity
        if 'Location' in df.columns:
            df['Location_Depth'] = df['Location'].str.count('>')
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill boolean columns with False
        boolean_columns = ['OverDue', 'Images', 'Comments', 'Documents']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].mode().empty:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def get_feature_columns(self):
        """Get columns suitable for ML features"""
        feature_columns = [
            'Project', 'Open Actions', 'Total Actions', 'Action_Completion_Rate',
            'Days_Since_Creation', 'Type_Risk_Level', 'Location_Depth',
            'Project_Type_Diversity', 'Project_Location_Diversity',
            'Images', 'Comments', 'Documents', 'OverDue'
        ]
        return feature_columns
    
    def get_target_columns(self):
        """Get target columns for ML models"""
        return ['Is_Delayed', 'Is_Potential_Failure']
