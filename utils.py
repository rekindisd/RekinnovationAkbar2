import base64
import pandas as pd
import streamlit as st

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    
    Args:
        object_to_download: The object to be downloaded (DataFrame, string, etc.)
        download_filename: Filename and extension of file
        download_link_text: Text to display for download link
        
    Returns:
        HTML string containing the download link
    """
    
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    
    # Encode the data
    b64 = base64.b64encode(object_to_download.encode()).decode()
    
    # Create the download link
    href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    
    return href

def format_number(num):
    """Format numbers for display"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def calculate_project_health(project_data):
    """Calculate overall project health score"""
    if len(project_data) == 0:
        return 0
    
    # Factors for health calculation
    completion_rate = len(project_data[project_data['Report Forms Status'] == 'Closed']) / len(project_data)
    overdue_rate = len(project_data[project_data['OverDue'] == True]) / len(project_data)
    
    # Total actions completion
    total_actions = project_data['Total Actions'].sum()
    open_actions = project_data['Open Actions'].sum()
    action_completion = (total_actions - open_actions) / total_actions if total_actions > 0 else 1
    
    # Calculate health score (0-100)
    health_score = (completion_rate * 0.4 + action_completion * 0.4 + (1 - overdue_rate) * 0.2) * 100
    
    return min(100, max(0, health_score))

def get_risk_color(risk_level):
    """Get color based on risk level"""
    if risk_level >= 0.7:
        return "red"
    elif risk_level >= 0.4:
        return "orange"
    else:
        return "green"

def create_summary_stats(data):
    """Create summary statistics for the data"""
    stats = {
        'total_forms': len(data),
        'total_projects': data['Project'].nunique(),
        'open_forms': len(data[data['Report Forms Status'] == 'Open']),
        'closed_forms': len(data[data['Report Forms Status'] == 'Closed']),
        'overdue_forms': len(data[data['OverDue'] == True]),
        'forms_with_images': len(data[data['Images'] == True]),
        'forms_with_comments': len(data[data['Comments'] == True]),
        'forms_with_documents': len(data[data['Documents'] == True]),
        'avg_open_actions': data['Open Actions'].mean(),
        'avg_total_actions': data['Total Actions'].mean()
    }
    
    return stats

def validate_data(data):
    """Validate the uploaded data"""
    required_columns = ['Ref', 'Status', 'Project', 'Created', 'Type']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    if len(data) == 0:
        return False, "Dataset is empty"
    
    return True, "Data validation passed"
