import os
import pandas as pd
import sqlalchemy as sql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text
from datetime import datetime
import streamlit as st

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')

Base = declarative_base()

class ConstructionProject(Base):
    __tablename__ = 'construction_projects'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ref = Column(String(100), unique=True, nullable=False)
    status = Column(String(100))
    location = Column(Text)
    name = Column(Text)
    created = Column(DateTime)
    type = Column(String(100))
    status_changed = Column(DateTime)
    open_actions = Column(Integer, default=0)
    total_actions = Column(Integer, default=0)
    association = Column(String(100))
    overdue = Column(Boolean, default=False)
    images = Column(Boolean, default=False)
    comments = Column(Boolean, default=False)
    documents = Column(Boolean, default=False)
    project = Column(Integer)
    report_forms_status = Column(String(50))
    report_forms_group = Column(String(100))
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class ProjectPrediction(Base):
    __tablename__ = 'project_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, nullable=False)
    form_type = Column(String(100))
    delay_probability = Column(Float)
    failure_probability = Column(Float)
    open_actions = Column(Integer)
    total_actions = Column(Integer)
    has_images = Column(Boolean)
    has_comments = Column(Boolean)
    has_documents = Column(Boolean)
    predicted_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.Session = None
        self.connect()
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.engine = create_engine(DATABASE_URL)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            return True
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return False
    
    def save_construction_data(self, dataframe):
        """Save construction data to database"""
        try:
            session = self.Session()
            
            # Clear existing data (optional - you might want to keep historical data)
            session.query(ConstructionProject).delete()
            
            # Convert dataframe to database records
            for _, row in dataframe.iterrows():
                project_record = ConstructionProject(
                    ref=str(row.get('Ref', '')),
                    status=str(row.get('Status', '')),
                    location=str(row.get('Location', '')),
                    name=str(row.get('Name', '')),
                    created=pd.to_datetime(row.get('Created'), errors='coerce'),
                    type=str(row.get('Type', '')),
                    status_changed=pd.to_datetime(row.get('Status Changed'), errors='coerce'),
                    open_actions=int(row.get('Open Actions', 0)) if pd.notna(row.get('Open Actions')) else 0,
                    total_actions=int(row.get('Total Actions', 0)) if pd.notna(row.get('Total Actions')) else 0,
                    association=str(row.get('Association', '')),
                    overdue=bool(row.get('OverDue', False)),
                    images=bool(row.get('Images', False)),
                    comments=bool(row.get('Comments', False)),
                    documents=bool(row.get('Documents', False)),
                    project=int(row.get('Project', 0)) if pd.notna(row.get('Project')) else 0,
                    report_forms_status=str(row.get('Report Forms Status', '')),
                    report_forms_group=str(row.get('Report Forms Group', ''))
                )
                session.add(project_record)
            
            session.commit()
            session.close()
            return True, f"Successfully saved {len(dataframe)} records to database"
            
        except Exception as e:
            session.rollback()
            session.close()
            return False, f"Error saving data: {str(e)}"
    
    def load_construction_data(self):
        """Load construction data from database"""
        try:
            session = self.Session()
            projects = session.query(ConstructionProject).all()
            session.close()
            
            if not projects:
                return None, "No data found in database"
            
            # Convert to dataframe
            data = []
            for project in projects:
                data.append({
                    'Ref': project.ref,
                    'Status': project.status,
                    'Location': project.location,
                    'Name': project.name,
                    'Created': project.created,
                    'Type': project.type,
                    'Status Changed': project.status_changed,
                    'Open Actions': project.open_actions,
                    'Total Actions': project.total_actions,
                    'Association': project.association,
                    'OverDue': project.overdue,
                    'Images': project.images,
                    'Comments': project.comments,
                    'Documents': project.documents,
                    'Project': project.project,
                    'Report Forms Status': project.report_forms_status,
                    'Report Forms Group': project.report_forms_group
                })
            
            df = pd.DataFrame(data)
            return df, f"Successfully loaded {len(df)} records from database"
            
        except Exception as e:
            return None, f"Error loading data: {str(e)}"
    
    def save_prediction(self, project_id, form_type, delay_prob, failure_prob, 
                       open_actions, total_actions, has_images, has_comments, has_documents):
        """Save prediction results to database"""
        try:
            session = self.Session()
            
            prediction = ProjectPrediction(
                project_id=project_id,
                form_type=form_type,
                delay_probability=delay_prob,
                failure_probability=failure_prob,
                open_actions=open_actions,
                total_actions=total_actions,
                has_images=has_images,
                has_comments=has_comments,
                has_documents=has_documents
            )
            
            session.add(prediction)
            session.commit()
            session.close()
            return True, "Prediction saved successfully"
            
        except Exception as e:
            session.rollback()
            session.close()
            return False, f"Error saving prediction: {str(e)}"
    
    def get_prediction_history(self, limit=50):
        """Get prediction history from database"""
        try:
            session = self.Session()
            predictions = session.query(ProjectPrediction).order_by(
                ProjectPrediction.predicted_at.desc()
            ).limit(limit).all()
            session.close()
            
            if not predictions:
                return None, "No prediction history found"
            
            # Convert to dataframe
            data = []
            for pred in predictions:
                data.append({
                    'Project ID': pred.project_id,
                    'Form Type': pred.form_type,
                    'Delay Probability': pred.delay_probability,
                    'Failure Probability': pred.failure_probability,
                    'Open Actions': pred.open_actions,
                    'Total Actions': pred.total_actions,
                    'Has Images': pred.has_images,
                    'Has Comments': pred.has_comments,
                    'Has Documents': pred.has_documents,
                    'Predicted At': pred.predicted_at
                })
            
            df = pd.DataFrame(data)
            return df, f"Successfully loaded {len(df)} prediction records"
            
        except Exception as e:
            return None, f"Error loading predictions: {str(e)}"
    
    def get_project_statistics(self):
        """Get project statistics from database"""
        try:
            session = self.Session()
            
            # Basic statistics
            total_projects = session.query(ConstructionProject.project).distinct().count()
            total_forms = session.query(ConstructionProject).count()
            open_forms = session.query(ConstructionProject).filter_by(report_forms_status='Open').count()
            closed_forms = session.query(ConstructionProject).filter_by(report_forms_status='Closed').count()
            overdue_forms = session.query(ConstructionProject).filter_by(overdue=True).count()
            
            session.close()
            
            stats = {
                'total_projects': total_projects,
                'total_forms': total_forms,
                'open_forms': open_forms,
                'closed_forms': closed_forms,
                'overdue_forms': overdue_forms,
                'completion_rate': (closed_forms / total_forms * 100) if total_forms > 0 else 0
            }
            
            return stats, "Statistics retrieved successfully"
            
        except Exception as e:
            return None, f"Error getting statistics: {str(e)}"

# Initialize database manager
@st.cache_resource
def get_database_manager():
    return DatabaseManager()
