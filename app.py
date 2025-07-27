import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64

from data_processor import DataProcessor
from ml_models import ProjectMLModels
from visualizations import ProjectVisualizations
from utils import download_link
from database import get_database_manager

# Page configuration
st.set_page_config(
    page_title="Construction Project Analytics & ML",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = None
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = get_database_manager()

def main():
    st.title("🏗️ Construction Project Analytics & ML")
    st.markdown("### Predict project delays and analyze construction data")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Upload", "Data Analytics", "Machine Learning", "Risk Assessment", "Database", "Export Results"]
    )
    
    if page == "Data Upload":
        data_upload_page()
    elif page == "Data Analytics":
        data_analytics_page()
    elif page == "Machine Learning":
        machine_learning_page()
    elif page == "Risk Assessment":
        risk_assessment_page()
    elif page == "Database":
        database_page()
    elif page == "Export Results":
        export_results_page()

def data_upload_page():
    st.header("📁 Data Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Construction Data CSV",
        type=['csv'],
        help="Upload your construction project management data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            # Process data
            processor = DataProcessor(data)
            processed_data = processor.process_data()
            st.session_state.processed_data = processed_data
            
            # Save to database
            db_manager = st.session_state.db_manager
            success, message = db_manager.save_construction_data(data)
            
            if success:
                st.success("✅ Data uploaded, processed, and saved to database successfully!")
                st.info(f"📊 {message}")
            else:
                st.warning("⚠️ Data processed but database save failed")
                st.error(f"Database error: {message}")
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Total Projects", data['Project'].nunique())
            with col3:
                st.metric("Status Types", data['Status'].nunique())
            with col4:
                st.metric("Form Types", data['Type'].nunique())
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(10))
            
            # Show data info
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Information:**")
                buffer = io.StringIO()
                data.info(buf=buffer)
                st.text(buffer.getvalue())
            
            with col2:
                st.write("**Missing Values:**")
                missing_data = data.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                if len(missing_data) > 0:
                    st.dataframe(missing_data.to_frame(name='Missing Count'))
                else:
                    st.write("No missing values found!")
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to get started")

def data_analytics_page():
    st.header("📊 Data Analytics")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    data = st.session_state.data
    processed_data = st.session_state.processed_data
    
    # Project selector
    projects = sorted(data['Project'].unique())
    selected_projects = st.multiselect(
        "Select Projects to Analyze",
        projects,
        default=projects[:5] if len(projects) > 5 else projects
    )
    
    if not selected_projects:
        st.warning("Please select at least one project!")
        return
    
    # Filter data by selected projects
    filtered_data = data[data['Project'].isin(selected_projects)]
    
    # Create visualizations
    viz = ProjectVisualizations(filtered_data)
    
    # Overview metrics
    st.subheader("📈 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Selected Projects", len(selected_projects))
    with col2:
        st.metric("Total Forms", len(filtered_data))
    with col3:
        open_forms = len(filtered_data[filtered_data['Report Forms Status'] == 'Open'])
        st.metric("Open Forms", open_forms)
    with col4:
        closed_forms = len(filtered_data[filtered_data['Report Forms Status'] == 'Closed'])
        st.metric("Closed Forms", closed_forms)
    
    # Status distribution
    st.subheader("📊 Status Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_status = viz.create_status_distribution()
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        fig_type = viz.create_type_distribution()
        st.plotly_chart(fig_type, use_container_width=True)
    
    # Time analysis
    st.subheader("⏰ Time Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_timeline = viz.create_timeline_analysis()
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        fig_project_progress = viz.create_project_progress()
        st.plotly_chart(fig_project_progress, use_container_width=True)
    
    # Project comparison
    st.subheader("🔍 Project Comparison")
    fig_comparison = viz.create_project_comparison()
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed project analysis
    st.subheader("🎯 Detailed Project Analysis")
    selected_project = st.selectbox("Select a project for detailed analysis", selected_projects)
    
    if selected_project:
        project_data = filtered_data[filtered_data['Project'] == selected_project]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Forms", len(project_data))
        with col2:
            overdue_count = len(project_data[project_data['OverDue'] == True])
            st.metric("Overdue Forms", overdue_count)
        with col3:
            with_images = len(project_data[project_data['Images'] == True])
            st.metric("Forms with Images", with_images)
        
        # Project details table
        st.write("**Project Details:**")
        project_summary = project_data.groupby(['Status', 'Type']).size().reset_index(name='Count')
        st.dataframe(project_summary)

def machine_learning_page():
    st.header("🤖 Machine Learning")
    
    if st.session_state.processed_data is None:
        st.warning("Please upload and process data first!")
        return
    
    processed_data = st.session_state.processed_data
    
    # Initialize ML models
    if st.session_state.ml_models is None:
        with st.spinner("Training machine learning models..."):
            ml_models = ProjectMLModels(processed_data)
            results = ml_models.train_models()
            st.session_state.ml_models = ml_models
    else:
        ml_models = st.session_state.ml_models
        results = ml_models.get_model_results()
    
    # Model performance
    st.subheader("📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Delay Prediction Model:**")
        if 'delay_metrics' in results:
            metrics = results['delay_metrics']
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("Precision", f"{metrics['precision']:.3f}")
            st.metric("Recall", f"{metrics['recall']:.3f}")
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        else:
            st.warning("Insufficient data for delay prediction")
    
    with col2:
        st.write("**Failure Prediction Model:**")
        if 'failure_metrics' in results:
            metrics = results['failure_metrics']
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("Precision", f"{metrics['precision']:.3f}")
            st.metric("Recall", f"{metrics['recall']:.3f}")
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        else:
            st.warning("Insufficient data for failure prediction")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    if 'feature_importance' in results:
        importance_df = results['feature_importance']
        fig = px.bar(
            importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Model predictions
    st.subheader("🔮 Prediksi Risiko Proyek")
    
    # Input form for new predictions
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_projects = sorted(processed_data['Project'].unique())
            project_id = st.selectbox("Pilih Project ID", available_projects)
            form_type = st.selectbox("Tipe Form", processed_data['Type'].unique())
            
        with col2:
            has_images = st.checkbox("Ada Gambar")
            has_comments = st.checkbox("Ada Komentar")
            has_documents = st.checkbox("Ada Dokumen")
            
        with col3:
            open_actions = st.number_input("Aksi Terbuka", min_value=0, value=0)
            total_actions = st.number_input("Total Aksi", min_value=0, value=0)
        
        predict_button = st.form_submit_button("Prediksi Risiko")
        
        if predict_button and st.session_state.ml_models:
            # Create input data for prediction
            input_data = {
                'Project': project_id,
                'Type': form_type,
                'Open Actions': open_actions,
                'Total Actions': total_actions,
                'Images': has_images,
                'Comments': has_comments,
                'Documents': has_documents
            }
            
            # Make predictions
            predictions = ml_models.predict(input_data)
            
            # Save prediction to database
            db_manager = st.session_state.db_manager
            save_success, save_message = db_manager.save_prediction(
                project_id, form_type, 
                predictions.get('delay_probability', 0),
                predictions.get('failure_probability', 0),
                open_actions, total_actions,
                has_images, has_comments, has_documents
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if 'delay_probability' in predictions:
                    delay_prob = predictions['delay_probability']
                    st.metric("Probabilitas Delay", f"{delay_prob:.1%}")
                    if delay_prob > 0.5:
                        st.error("⚠️ Risiko tinggi terjadi delay!")
                    else:
                        st.success("✅ Risiko rendah terjadi delay")
                        
            with col2:
                if 'failure_probability' in predictions:
                    failure_prob = predictions['failure_probability']
                    st.metric("Probabilitas Gagal", f"{failure_prob:.1%}")
                    if failure_prob > 0.3:
                        st.error("⚠️ Risiko tinggi proyek gagal!")
                    else:
                        st.success("✅ Risiko rendah proyek gagal")
            
            if save_success:
                st.info("💾 Prediksi disimpan ke database")
            else:
                st.warning(f"⚠️ Gagal menyimpan prediksi: {save_message}")

def risk_assessment_page():
    st.header("⚠️ Risk Assessment Dashboard")
    
    if st.session_state.processed_data is None or st.session_state.ml_models is None:
        st.warning("Please upload data and train models first!")
        return
    
    data = st.session_state.data
    ml_models = st.session_state.ml_models
    
    # Overall risk metrics
    st.subheader("📊 Overall Risk Metrics")
    
    # Calculate risk for all open forms
    open_forms = data[data['Report Forms Status'] == 'Open']
    
    if len(open_forms) > 0:
        risk_predictions = []
        
        for _, row in open_forms.iterrows():
            input_data = {
                'Project': row['Project'],
                'Type': row['Type'],
                'Open Actions': row['Open Actions'],
                'Total Actions': row['Total Actions'],
                'Images': row['Images'],
                'Comments': row['Comments'],
                'Documents': row['Documents']
            }
            
            try:
                predictions = ml_models.predict(input_data)
                risk_predictions.append({
                    'Ref': row['Ref'],
                    'Project': row['Project'],
                    'Name': row['Name'],
                    'Type': row['Type'],
                    'Delay_Risk': predictions.get('delay_probability', 0),
                    'Failure_Risk': predictions.get('failure_probability', 0)
                })
            except:
                continue
        
        if risk_predictions:
            risk_df = pd.DataFrame(risk_predictions)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_delay_risk = len(risk_df[risk_df['Delay_Risk'] > 0.5])
                st.metric("High Delay Risk", f"{high_delay_risk}")
            
            with col2:
                high_failure_risk = len(risk_df[risk_df['Failure_Risk'] > 0.3])
                st.metric("High Failure Risk", f"{high_failure_risk}")
            
            with col3:
                avg_delay_risk = risk_df['Delay_Risk'].mean()
                st.metric("Avg Delay Risk", f"{avg_delay_risk:.1%}")
            
            with col4:
                avg_failure_risk = risk_df['Failure_Risk'].mean()
                st.metric("Avg Failure Risk", f"{avg_failure_risk:.1%}")
            
            # Risk distribution charts
            st.subheader("📈 Risk Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_delay = px.histogram(
                    risk_df,
                    x='Delay_Risk',
                    title="Delay Risk Distribution",
                    labels={'Delay_Risk': 'Delay Risk Probability'},
                    nbins=20
                )
                st.plotly_chart(fig_delay, use_container_width=True)
            
            with col2:
                fig_failure = px.histogram(
                    risk_df,
                    x='Failure_Risk',
                    title="Failure Risk Distribution",
                    labels={'Failure_Risk': 'Failure Risk Probability'},
                    nbins=20
                )
                st.plotly_chart(fig_failure, use_container_width=True)
            
            # High-risk items
            st.subheader("🚨 High-Risk Items")
            
            # Filter high-risk items
            high_risk_items = risk_df[
                (risk_df['Delay_Risk'] > 0.5) | (risk_df['Failure_Risk'] > 0.3)
            ].sort_values(['Delay_Risk', 'Failure_Risk'], ascending=False)
            
            if len(high_risk_items) > 0:
                st.dataframe(
                    high_risk_items[['Ref', 'Project', 'Name', 'Type', 'Delay_Risk', 'Failure_Risk']],
                    use_container_width=True
                )
            else:
                st.success("🎉 No high-risk items identified!")
            
            # Project-wise risk analysis
            st.subheader("🏗️ Project-wise Risk Analysis")
            
            project_risk = risk_df.groupby('Project').agg({
                'Delay_Risk': 'mean',
                'Failure_Risk': 'mean'
            }).reset_index()
            
            fig_project_risk = px.scatter(
                project_risk,
                x='Delay_Risk',
                y='Failure_Risk',
                title="Project Risk Matrix",
                labels={
                    'Delay_Risk': 'Average Delay Risk',
                    'Failure_Risk': 'Average Failure Risk'
                },
                text='Project'
            )
            fig_project_risk.add_hline(y=0.3, line_dash="dash", line_color="red")
            fig_project_risk.add_vline(x=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_project_risk, use_container_width=True)
            
            # Store risk data for export
            st.session_state.risk_data = risk_df
        
        else:
            st.warning("Unable to calculate risk predictions for the data.")
    else:
        st.info("No open forms found for risk assessment.")

def database_page():
    st.header("🗃️ Database Management")
    
    db_manager = st.session_state.db_manager
    
    # Database statistics
    st.subheader("📊 Database Statistics")
    stats, stats_message = db_manager.get_project_statistics()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Projects", stats['total_projects'])
        with col2:
            st.metric("Total Forms", stats['total_forms'])
        with col3:
            st.metric("Open Forms", stats['open_forms'])
        with col4:
            st.metric("Completion Rate", f"{stats['completion_rate']:.1f}%")
    else:
        st.warning(f"Could not load statistics: {stats_message}")
    
    # Load data from database
    st.subheader("📥 Load Data from Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Construction Data"):
            with st.spinner("Loading data from database..."):
                data, message = db_manager.load_construction_data()
                
                if data is not None:
                    st.session_state.data = data
                    
                    # Process the loaded data
                    processor = DataProcessor(data)
                    processed_data = processor.process_data()
                    st.session_state.processed_data = processed_data
                    
                    st.success(f"✅ {message}")
                    st.info("Data loaded and processed successfully")
                else:
                    st.error(f"❌ {message}")
    
    with col2:
        if st.button("Load Prediction History"):
            with st.spinner("Loading prediction history..."):
                predictions, pred_message = db_manager.get_prediction_history()
                
                if predictions is not None:
                    st.success(f"✅ {pred_message}")
                    st.subheader("🔮 Recent Predictions")
                    st.dataframe(predictions, use_container_width=True)
                else:
                    st.error(f"❌ {pred_message}")
    
    # Data preview
    if st.session_state.data is not None:
        st.subheader("👀 Current Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        st.write(f"**Total records:** {len(st.session_state.data)}")
    
    # Database management actions
    st.subheader("🔧 Database Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Database**")
        if st.button("📤 Export All Data"):
            data, message = db_manager.load_construction_data()
            if data is not None:
                csv_data = data.to_csv(index=False)
                st.download_button(
                    label="Download Database Export",
                    data=csv_data,
                    file_name=f"database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📈 Export Predictions"):
            predictions, pred_message = db_manager.get_prediction_history()
            if predictions is not None:
                pred_csv = predictions.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=pred_csv,
                    file_name=f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def export_results_page():
    st.header("📤 Export Results")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    st.subheader("Available Exports")
    
    # Original data export
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Original Data"):
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="Download Original Data CSV",
                data=csv,
                file_name="construction_data_original.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.session_state.processed_data is not None:
            if st.button("🔧 Export Processed Data"):
                csv = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data CSV",
                    data=csv,
                    file_name="construction_data_processed.csv",
                    mime="text/csv"
                )
    
    # Risk assessment export
    if hasattr(st.session_state, 'risk_data'):
        st.subheader("Risk Assessment Export")
        if st.button("⚠️ Export Risk Assessment"):
            csv = st.session_state.risk_data.to_csv(index=False)
            st.download_button(
                label="Download Risk Assessment CSV",
                data=csv,
                file_name="construction_risk_assessment.csv",
                mime="text/csv"
            )
    
    # Model results export
    if st.session_state.ml_models is not None:
        st.subheader("Model Results Export")
        if st.button("🤖 Export Model Results"):
            results = st.session_state.ml_models.get_model_results()
            
            # Create a summary report
            report = []
            report.append("Construction Project ML Model Results")
            report.append("="*50)
            report.append("")
            
            if 'delay_metrics' in results:
                report.append("Delay Prediction Model:")
                metrics = results['delay_metrics']
                report.append(f"  Accuracy: {metrics['accuracy']:.3f}")
                report.append(f"  Precision: {metrics['precision']:.3f}")
                report.append(f"  Recall: {metrics['recall']:.3f}")
                report.append(f"  F1 Score: {metrics['f1']:.3f}")
                report.append("")
            
            if 'failure_metrics' in results:
                report.append("Failure Prediction Model:")
                metrics = results['failure_metrics']
                report.append(f"  Accuracy: {metrics['accuracy']:.3f}")
                report.append(f"  Precision: {metrics['precision']:.3f}")
                report.append(f"  Recall: {metrics['recall']:.3f}")
                report.append(f"  F1 Score: {metrics['f1']:.3f}")
                report.append("")
            
            if 'feature_importance' in results:
                report.append("Top 10 Feature Importance:")
                importance_df = results['feature_importance']
                for _, row in importance_df.head(10).iterrows():
                    report.append(f"  {row['feature']}: {row['importance']:.3f}")
            
            report_text = "\n".join(report)
            
            st.download_button(
                label="Download Model Results Report",
                data=report_text,
                file_name="ml_model_results.txt",
                mime="text/plain"
            )

def generate_summary_report(data):
    """Generate a comprehensive summary report"""
    report = []
    report.append("CONSTRUCTION PROJECT ANALYTICS SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Basic statistics
    report.append("BASIC STATISTICS:")
    report.append("-" * 20)
    report.append(f"Total Records: {len(data)}")
    report.append(f"Total Projects: {data['Project'].nunique()}")
    report.append(f"Total Form Types: {data['Type'].nunique()}")
    report.append("")
    
    # Status distribution
    report.append("STATUS DISTRIBUTION:")
    report.append("-" * 20)
    status_counts = data['Status'].value_counts()
    for status, count in status_counts.items():
        percentage = (count / len(data)) * 100
        report.append(f"{status}: {count} ({percentage:.1f}%)")
    report.append("")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()
