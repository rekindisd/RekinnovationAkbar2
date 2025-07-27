import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class ProjectVisualizations:
    def __init__(self, data):
        self.data = data.copy()
        
    def create_status_distribution(self):
        """Create status distribution pie chart"""
        status_counts = self.data['Status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Project Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=400)
        
        return fig
    
    def create_type_distribution(self):
        """Create form type distribution chart"""
        type_counts = self.data['Type'].value_counts().head(10)
        
        fig = px.bar(
            x=type_counts.values,
            y=type_counts.index,
            orientation='h',
            title="Top 10 Form Types",
            labels={'x': 'Count', 'y': 'Form Type'},
            color=type_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        
        return fig
    
    def create_timeline_analysis(self):
        """Create timeline analysis of form creation"""
        if 'Created' not in self.data.columns:
            return self._create_empty_chart("Timeline data not available")
        
        # Convert to datetime if not already
        timeline_data = self.data.copy()
        timeline_data['Created'] = pd.to_datetime(timeline_data['Created'], errors='coerce')
        timeline_data = timeline_data.dropna(subset=['Created'])
        
        if len(timeline_data) == 0:
            return self._create_empty_chart("No valid date data available")
        
        # Group by date
        daily_counts = timeline_data.groupby(timeline_data['Created'].dt.date).size().reset_index()
        daily_counts.columns = ['Date', 'Count']
        
        fig = px.line(
            daily_counts,
            x='Date',
            y='Count',
            title="Daily Form Creation Timeline",
            labels={'Count': 'Number of Forms Created'}
        )
        
        fig.update_traces(mode='lines+markers')
        fig.update_layout(height=400)
        
        return fig
    
    def create_project_progress(self):
        """Create project progress visualization"""
        # Calculate completion rate by project
        project_stats = self.data.groupby('Project').agg({
            'Report Forms Status': lambda x: (x == 'Closed').sum() / len(x) * 100,
            'Status': 'count'
        }).reset_index()
        
        project_stats.columns = ['Project', 'Completion_Rate', 'Total_Forms']
        project_stats = project_stats.sort_values('Completion_Rate', ascending=True)
        
        fig = px.bar(
            project_stats.head(20),  # Show top 20 projects
            x='Completion_Rate',
            y='Project',
            orientation='h',
            title="Project Completion Rates (%)",
            labels={'Completion_Rate': 'Completion Rate (%)', 'Project': 'Project ID'},
            color='Completion_Rate',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        
        return fig
    
    def create_project_comparison(self):
        """Create project comparison chart"""
        project_summary = self.data.groupby('Project').agg({
            'Open Actions': 'sum',
            'Total Actions': 'sum',
            'OverDue': lambda x: (x == True).sum(),
            'Status': 'count'
        }).reset_index()
        
        project_summary.columns = ['Project', 'Open_Actions', 'Total_Actions', 'Overdue_Count', 'Form_Count']
        
        # Calculate metrics
        project_summary['Action_Completion_Rate'] = np.where(
            project_summary['Total_Actions'] > 0,
            (project_summary['Total_Actions'] - project_summary['Open_Actions']) / project_summary['Total_Actions'] * 100,
            100
        )
        
        project_summary['Overdue_Rate'] = project_summary['Overdue_Count'] / project_summary['Form_Count'] * 100
        
        # Filter projects with sufficient data
        project_summary = project_summary[project_summary['Form_Count'] >= 5].head(15)
        
        if len(project_summary) == 0:
            return self._create_empty_chart("Insufficient project data for comparison")
        
        fig = px.scatter(
            project_summary,
            x='Action_Completion_Rate',
            y='Overdue_Rate',
            size='Form_Count',
            hover_data=['Project', 'Form_Count'],
            title="Project Performance Matrix",
            labels={
                'Action_Completion_Rate': 'Action Completion Rate (%)',
                'Overdue_Rate': 'Overdue Rate (%)',
                'Form_Count': 'Number of Forms'
            }
        )
        
        # Add quadrant lines
        fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="High Overdue Threshold")
        fig.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="Good Completion Threshold")
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_risk_heatmap(self, risk_data):
        """Create risk assessment heatmap"""
        if risk_data is None or len(risk_data) == 0:
            return self._create_empty_chart("No risk data available")
        
        # Create risk matrix
        risk_matrix = risk_data.pivot_table(
            values='Delay_Risk',
            index='Project',
            columns='Type',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            risk_matrix,
            title="Risk Heatmap by Project and Form Type",
            labels=dict(x="Form Type", y="Project", color="Risk Level"),
            aspect="auto",
            color_continuous_scale="RdYlBu_r"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def _create_empty_chart(self, message):
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        return fig
