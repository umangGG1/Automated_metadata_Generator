import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our custom classes (assuming they're in the same directory)
from automated_metadata_generation import DocumentProcessor, SemanticAnalyzer, MetadataGenerator

# Configure Streamlit page
st.set_page_config(
    page_title="Automated Metadata Generation System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    padding: 1rem 0;
    border-bottom: 2px solid #1f77b4;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #c3e6cb;
}

.error-message {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None

# Main header
st.markdown('<h1 class="main-header">📄 Automated Metadata Generation System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛠️ System Controls")
st.sidebar.markdown("---")

# File upload section
st.sidebar.subheader("📤 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a document",
    type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    help="Supported formats: PDF, DOCX, TXT, and image files (PNG, JPG, TIFF, BMP)"
)

# Processing options
st.sidebar.subheader("⚙️ Processing Options")
include_ocr = st.sidebar.checkbox("Enable OCR for images", value=True, help="Use OCR to extract text from images")
detailed_analysis = st.sidebar.checkbox("Detailed semantic analysis", value=True, help="Perform comprehensive NLP analysis")

# Initialize metadata generator
@st.cache_resource
def initialize_metadata_generator():
    """Initialize the metadata generator (cached for performance)"""
    return MetadataGenerator()

try:
    metadata_generator = initialize_metadata_generator()
    st.sidebar.success("✅ System initialized successfully!")
except Exception as e:
    st.sidebar.error(f"❌ System initialization failed: {str(e)}")
    st.error("Please check your environment and install required dependencies.")
    st.stop()

# Main content area
if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Process button
    if st.sidebar.button("🚀 Generate Metadata", type="primary"):
        with st.spinner("Processing document... This may take a few moments."):
            try:
                # Generate metadata
                metadata = metadata_generator.generate_metadata(tmp_file_path)
                st.session_state.metadata = metadata
                st.session_state.processed_file = uploaded_file.name
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                st.sidebar.markdown('<div class="success-message">✅ Metadata generated successfully!</div>', 
                                  unsafe_allow_html=True)
                
            except Exception as e:
                st.sidebar.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', 
                                  unsafe_allow_html=True)
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

# Display results if metadata is available
if st.session_state.metadata is not None:
    metadata = st.session_state.metadata
    
    # Check if processing was successful
    if 'error' in metadata:
        st.error(f"❌ Processing failed: {metadata['error']}")
    else:
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📄 File Info", "🧠 Content Analysis", 
            "📈 Visualizations", "💾 Export"
        ])
        
        with tab1:
            st.subheader("📊 Metadata Overview")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="File Size",
                    value=metadata['file_metadata']['file_size_human']
                )
            
            with col2:
                st.metric(
                    label="Word Count",
                    value=f"{metadata['content_metadata']['content_metrics']['word_count']:,}"
                )
            
            with col3:
                st.metric(
                    label="Language",
                    value=metadata['content_metadata']['language'].upper()
                )
            
            with col4:
                st.metric(
                    label="Completeness Score",
                    value=f"{metadata['quality_metrics']['completeness_score']:.1%}",
                    delta=f"{metadata['quality_metrics']['classification_confidence']:.1%}"
                )
            
            # Document type and summary
            st.subheader("📝 Document Summary")
            doc_type = metadata['content_metadata']['document_type']
            st.info(f"**Document Type:** {doc_type['predicted_type']} (Confidence: {doc_type['confidence']:.1%})")
            
            summary = metadata['content_metadata']['summary']
            if summary:
                st.write("**Summary:**")
                st.write(summary)
            else:
                st.warning("No summary available - text extraction may have failed.")
        
        with tab2:
            st.subheader("📄 File Information")
            
            # File metadata
            file_info = metadata['file_metadata']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"• **File Name:** {file_info['file_name']}")
                st.write(f"• **File Type:** {file_info['file_extension'].upper()}")
                st.write(f"• **File Size:** {file_info['file_size_human']} ({file_info['file_size_bytes']:,} bytes)")
                st.write(f"• **MD5 Hash:** `{file_info['file_hash_md5']}`")
            
            with col2:
                st.write("**Timestamps:**")
                st.write(f"• **Created:** {file_info['creation_time']}")
                st.write(f"• **Modified:** {file_info['modification_time']}")
                st.write(f"• **Processed:** {metadata['processing_metadata']['processing_timestamp']}")
                st.write(f"• **Extraction Method:** {metadata['processing_metadata']['extraction_method']}")
        
        with tab3:
            st.subheader("🧠 Content Analysis")
            
            # Content metrics
            content_metrics = metadata['content_metadata']['content_metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Text Statistics:**")
                st.write(f"• **Characters:** {content_metrics['character_count']:,}")
                st.write(f"• **Words:** {content_metrics['word_count']:,}")
                st.write(f"• **Sentences:** {content_metrics['sentence_count']:,}")
                st.write(f"• **Paragraphs:** {content_metrics['paragraph_count']:,}")
                st.write(f"• **Unique Words:** {content_metrics['unique_words']:,}")
                st.write(f"• **Lexical Diversity:** {content_metrics['lexical_diversity']:.3f}")
            
            with col2:
                st.write("**Readability Scores:**")
                readability = metadata['content_metadata']['readability_scores']
                for metric, score in readability.items():
                    st.write(f"• **{metric.replace('_', ' ').title()}:** {score:.1f}")
            
            # Key phrases
            st.subheader("🔑 Key Phrases")
            key_phrases = metadata['content_metadata']['key_phrases']
            if key_phrases:
                # Display as tags
                phrase_html = ""
                for phrase in key_phrases[:15]:
                    phrase_html += f'<span style="background-color: #002b70; padding: 0.2rem 0.5rem; margin: 0.1rem; border-radius: 0.3rem; display: inline-block;">{phrase}</span> '
                st.markdown(phrase_html, unsafe_allow_html=True)
            else:
                st.warning("No key phrases extracted.")
            
            # Topics
            st.subheader("📊 Topic Analysis")
            topics = metadata['content_metadata']['topics']
            if any(score > 0 for score in topics.values()):
                topics_df = pd.DataFrame(list(topics.items()), columns=['Topic', 'Score'])
                topics_df = topics_df[topics_df['Score'] > 0].sort_values('Score', ascending=False)
                
                if not topics_df.empty:
                    fig = px.bar(topics_df, x='Score', y='Topic', orientation='h',
                               title="Topic Relevance Scores")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No specific topics identified.")
            
            # Entities
            st.subheader("🏷️ Extracted Entities")
            entities = metadata['content_metadata']['entities']
            
            entity_cols = st.columns(len(entities))
            for i, (entity_type, entity_list) in enumerate(entities.items()):
                with entity_cols[i]:
                    st.write(f"**{entity_type.replace('_', ' ').title()}**")
                    if entity_list:
                        for entity in entity_list[:5]:  # Show first 5
                            st.write(f"• {entity}")
                        if len(entity_list) > 5:
                            st.write(f"... and {len(entity_list) - 5} more")
                    else:
                        st.write("None found")
        
        with tab4:
            st.subheader("📈 Data Visualizations")
            
            # Content metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Readability scores radar chart
                readability = metadata['content_metadata']['readability_scores']
                if readability:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=list(readability.values()),
                        theta=[name.replace('_', ' ').title() for name in readability.keys()],
                        fill='toself',
                        name='Readability Scores'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=False,
                        title="Readability Metrics"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Quality metrics pie chart
                quality_scores = {
                    'Text Extraction': metadata['quality_metrics']['text_extraction_confidence'],
                    'Classification': metadata['quality_metrics']['classification_confidence'],
                    'Completeness': metadata['quality_metrics']['completeness_score']
                }
                
                fig = px.pie(
                    values=list(quality_scores.values()),
                    names=list(quality_scores.keys()),
                    title="Quality Metrics Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Word frequency (if key phrases available)
            if metadata['content_metadata']['key_phrases']:
                st.subheader("🔤 Word Frequency Analysis")
                phrases = metadata['content_metadata']['key_phrases'][:10]
                
                # Create a simple frequency chart
                fig = px.bar(
                    x=list(range(len(phrases))),
                    y=phrases,
                    orientation='h',
                    title="Top Key Phrases"
                )
                fig.update_layout(xaxis_title="Frequency Rank", yaxis_title="Phrases")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("💾 Export Options")
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON export
                json_str = json.dumps(metadata, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"metadata_{st.session_state.processed_file}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV export (flattened metadata)
                flattened_data = {}
                
                def flatten_dict(d, parent_key='', sep='_'):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)
                
                flattened_data = flatten_dict(metadata)
                df = pd.DataFrame([flattened_data])
                csv_str = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_str,
                    file_name=f"metadata_{st.session_state.processed_file}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Summary report
                summary_report = f"""
# Metadata Report: {metadata['file_metadata']['file_name']}

## File Information
- **File Name:** {metadata['file_metadata']['file_name']}
- **File Size:** {metadata['file_metadata']['file_size_human']}
- **File Type:** {metadata['file_metadata']['file_extension']}
- **Processing Date:** {metadata['processing_metadata']['processing_timestamp']}

## Content Summary
- **Language:** {metadata['content_metadata']['language']}
- **Document Type:** {metadata['content_metadata']['document_type']['predicted_type']}
- **Word Count:** {metadata['content_metadata']['content_metrics']['word_count']:,}
- **Completeness Score:** {metadata['quality_metrics']['completeness_score']:.1%}

## Key Findings
- **Summary:** {metadata['content_metadata']['summary'][:200]}...
- **Top Topics:** {', '.join([k for k, v in sorted(metadata['content_metadata']['topics'].items(), key=lambda x: x[1], reverse=True)[:3] if v > 0])}
- **Key Phrases:** {', '.join(metadata['content_metadata']['key_phrases'][:10])}

---
Generated by Automated Metadata Generation System
"""
                
                st.download_button(
                    label="📝 Download Report",
                    data=summary_report,
                    file_name=f"report_{st.session_state.processed_file}.md",
                    mime="text/markdown"
                )
            
            # Display metadata structure
            st.subheader("🔍 Raw Metadata Structure")
            with st.expander("View complete metadata structure"):
                st.json(metadata)

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Automated Metadata Generation System! 🚀
    
    This system provides comprehensive metadata extraction and analysis for various document types.
    
    ### 🌟 Features:
    - **Multi-format Support**: PDF, DOCX, TXT, and image files (PNG, JPG, TIFF, BMP)
    - **OCR Capabilities**: Extract text from scanned documents and images
    - **Semantic Analysis**: AI-powered content understanding and classification
    - **Structured Metadata**: Generate comprehensive, structured metadata
    - **Interactive Visualizations**: Explore your document's characteristics
    - **Multiple Export Formats**: JSON, CSV, and Markdown reports
    
    ### 📋 How to Use:
    1. **Upload** a document using the sidebar
    2. **Configure** processing options if needed
    3. **Click** "Generate Metadata" to process your document
    4. **Explore** the results in different tabs
    5. **Export** your metadata in your preferred format
    
    ### 🔧 Supported File Types:
    - **PDF**: Portable Document Format files
    - **DOCX**: Microsoft Word documents
    - **TXT**: Plain text files
    - **Images**: PNG, JPG, JPEG, TIFF, BMP (with OCR)
    
    Get started by uploading a document in the sidebar! ⬅️
    """)
    
    # System status
    st.subheader("🖥️ System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ Document Processor: Ready")
    with col2:
        st.success("✅ Semantic Analyzer: Ready")
    with col3:
        st.success("✅ Metadata Generator: Ready")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 Automated Metadata Generation System | Built with Streamlit & AI</p>
    <p>For technical support or feature requests, please refer to the documentation.</p>
</div>
""", unsafe_allow_html=True)