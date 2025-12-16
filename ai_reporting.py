"""
Streamlit Application for AI Medical Reporting
Interfaces with API for medical image analysis
"""

import streamlit as st
import requests
from PIL import Image
import io
import base64
from datetime import datetime
import markdown
import tempfile
import os
import pydicom
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import markdown2
from classification_client import ClassificationClient

# Import configuration
try:
    from config import *
except ImportError:
    # Fallback to default values if config.py is not found
    API_URL = "https://a5b9c23d9648.ngrok-free.app"
    DEFAULT_ROLE_INSTRUCTION = "You are an expert radiologist analyzing medical images. Provide detailed, professional medical assessments."
    MAX_TOKENS_DEFAULT = 2048
    MAX_TOKENS_MIN = 512
    MAX_TOKENS_MAX = 4096
    DEFAULT_PROMPT_PNG = "Please provide a detailed medical assessment of the image(s), including findings, differential diagnosis, and recommendations."
    DEFAULT_PROMPT_DICOM = "Please analyze this medical imaging study and provide detailed findings, including anatomical observations, pathological findings, and clinical recommendations."
    PAGE_TITLE = "AI Medical Reporting"
    PAGE_ICON = "🏥"
    FOOTER_TEXT = "<div style='text-align: center; color: #666; padding: 2rem;'><p>AI Medical Reporting System</p></div>"

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, beautiful UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #374151;
    }
    
    /* Main header with gradient */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Sidebar text - white for gradient background */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stExpanderHeader {
        color: white !important;
    }

    [data-testid="stSidebar"] .stTextInput>div>div>input {
        background-color: rgba(255,255,255,0.1);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Button styling - Modern gradient buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25);
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Upload section with glassmorphism */
    .upload-section {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        border: 2px dashed #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    /* Upload section text - dark for semi-transparent background */
    .upload-section * {
        color: #374151 !important;
    }
    
    /* Result section with card design */
    .result-section {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        border: none;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }

    /* Result section text - dark for white background */
    .result-section * {
        color: #374151 !important;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Error message */
    .error-message {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #d6ebff 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Image preview with modern styling */
    .image-preview {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 0.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease;
    }
    
    .image-preview:hover {
        transform: scale(1.05);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: 2px solid #e2e8f0;
        transition: all 0.3s;
    }

    /* Tab text - dark for white background */
    .stTabs [data-baseweb="tab"] {
        color: #374151 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8fafc;
        border-color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: transparent !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
        position: relative;
        color: #374151 !important;
        cursor: pointer;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
    }

    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.2);
        color: #1e293b !important;
    }

    /* Hide default expander icons to prevent text overlap */
    .streamlit-expanderHeader svg,
    .streamlit-expanderHeader::before {
        display: none !important;
    }

    /* Custom expander icon using CSS */
    .streamlit-expanderHeader {
        padding-right: 2rem;
    }

    .streamlit-expanderHeader::after {
        content: "▶";
        position: absolute;
        right: 1rem;
        top: 50%;
        transform: translateY(-50%);
        color: #667eea;
        font-size: 0.8rem;
        font-weight: bold;
        transition: transform 0.3s ease;
    }

    /* Rotate icon when expanded */
    [aria-expanded="true"] .streamlit-expanderHeader::after {
        transform: translateY(-50%) rotate(90deg);
    }

    /* Ensure expander content has proper spacing */
    .streamlit-expanderContent {
        margin-top: 0.5rem;
        padding-left: 1rem;
        border-left: 3px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px dashed #cbd5e1;
        transition: all 0.3s;
    }

    /* File uploader text - dark for white background */
    [data-testid="stFileUploader"] * {
        color: #374151 !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f8fafc;
    }
    
    /* Text input fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s;
        background: white !important;
        color: #374151 !important;
    }

    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Ensure text area labels are dark */
    .stTextArea label {
        color: #374151 !important;
    }

    /* Select boxes - ensure dark text */
    .stSelectbox > div > div {
        background: white !important;
        color: #374151 !important;
    }

    .stSelectbox label {
        color: #374151 !important;
    }

    /* Number input - ensure dark text */
    .stNumberInput > div > div > input {
        background: white !important;
        color: #374151 !important;
    }

    .stNumberInput label {
        color: #374151 !important;
    }

    /* Slider labels - ensure dark text */
    .stSlider label {
        color: #374151 !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }

    /* Radio button text - dark for white background */
    .stRadio > div * {
        color: #374151 !important;
    }

    /* Checkbox */
    .stCheckbox {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
    }

    /* Checkbox text - dark for white background */
    .stCheckbox * {
        color: #374151 !important;
    }

    /* Ensure radio button labels are dark */
    .stRadio label {
        color: #374151 !important;
    }

    /* Ensure checkbox labels are dark */
    .stCheckbox label {
        color: #374151 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Success/Info/Warning/Error boxes */
    .element-container div[data-testid="stNotification"] {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    }
    
    /* Card effect for containers */
    .element-container {
        transition: transform 0.3s ease;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'api_url' not in st.session_state:
    st.session_state.api_url = API_URL
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'dicom_metadata' not in st.session_state:
    st.session_state.dicom_metadata = {}
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []
if 'classification_api_url' not in st.session_state:
    st.session_state.classification_api_url = "http://localhost:5002"
if 'enable_classification' not in st.session_state:
    st.session_state.enable_classification = True
if 'classification_model_type' not in st.session_state:
    st.session_state.classification_model_type = 'auto'

def dicom_to_pil(dicom_file):
    """Convert DICOM file to PIL Image"""
    try:
        dicom_file.seek(0)
        ds = pydicom.dcmread(dicom_file)
        pixel_array = ds.pixel_array
        
        # Normalize to 0-255
        pixel_array = pixel_array.astype(float)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        # Convert to PIL Image
        if len(pixel_array.shape) == 2:
            img = Image.fromarray(pixel_array, mode='L').convert('RGB')
        else:
            img = Image.fromarray(pixel_array)
        
        return img, ds
    except Exception as e:
        st.error(f"Error converting DICOM: {str(e)}")
        return None, None

def extract_dicom_metadata(dicom_dataset):
    """Extract important DICOM metadata"""
    metadata = {}
    try:
        metadata['PatientID'] = str(getattr(dicom_dataset, 'PatientID', 'Unknown'))
        metadata['PatientName'] = str(getattr(dicom_dataset, 'PatientName', 'Unknown'))
        metadata['PatientAge'] = str(getattr(dicom_dataset, 'PatientAge', 'Unknown'))
        metadata['PatientSex'] = str(getattr(dicom_dataset, 'PatientSex', 'Unknown'))
        metadata['StudyDate'] = str(getattr(dicom_dataset, 'StudyDate', 'Unknown'))
        metadata['StudyDescription'] = str(getattr(dicom_dataset, 'StudyDescription', 'Unknown'))
        metadata['Modality'] = str(getattr(dicom_dataset, 'Modality', 'Unknown'))
        metadata['BodyPartExamined'] = str(getattr(dicom_dataset, 'BodyPartExamined', 'Unknown'))
        metadata['SeriesDescription'] = str(getattr(dicom_dataset, 'SeriesDescription', 'Unknown'))
        metadata['Manufacturer'] = str(getattr(dicom_dataset, 'Manufacturer', 'Unknown'))
        metadata['InstitutionName'] = str(getattr(dicom_dataset, 'InstitutionName', 'Unknown'))
    except Exception as e:
        st.warning(f"Could not extract all DICOM metadata: {str(e)}")
    return metadata

def create_dicom_context_prompt(metadata, base_prompt):
    """Create enhanced prompt with DICOM metadata context"""
    context = f"""
**DICOM Image Context:**
- Modality: {metadata.get('Modality', 'Unknown')}
- Body Part: {metadata.get('BodyPartExamined', 'Unknown')}
- Study Description: {metadata.get('StudyDescription', 'Unknown')}
- Series Description: {metadata.get('SeriesDescription', 'Unknown')}
- Patient Age: {metadata.get('PatientAge', 'Unknown')}
- Patient Sex: {metadata.get('PatientSex', 'Unknown')}

{base_prompt}
"""
    return context

def check_api_health(api_url):
    """Check if the API is available"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_loaded', False), data
        return False, None
    except:
        return False, None

def analyze_png_images(api_url, images, prompt, role_instruction, max_tokens):
    """Send PNG images to the API for analysis"""
    try:
        files = []
        for idx, img in enumerate(images):
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            files.append(('images', (f'image_{idx}.png', img_byte_arr, 'image/png')))
        
        data = {
            'prompt': prompt,
            'role_instruction': role_instruction,
            'max_new_tokens': max_tokens
        }
        
        response = requests.post(
            f"{api_url}/api/analyze-with-images",
            files=files,
            data=data,
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'success': False, 'error': f'API returned status code {response.status_code}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_dicom_files(api_url, dicom_files, prompt, role_instruction, max_tokens, is_series=False):
    """Send DICOM files to the API for analysis"""
    try:
        if is_series:
            # Multiple DICOM files (series)
            files = []
            for idx, dcm_file in enumerate(dicom_files):
                dcm_file.seek(0)
                files.append(('dicom_files', (dcm_file.name, dcm_file, 'application/dicom')))
            
            data = {
                'prompt': prompt,
                'role_instruction': role_instruction,
                'max_new_tokens': max_tokens,
                'generate_summary': True
            }
            
            response = requests.post(
                f"{api_url}/api/analyze-dicom-series",
                files=files,
                data=data,
                timeout=600
            )
        else:
            # Single DICOM file
            dcm_file = dicom_files[0]
            dcm_file.seek(0)
            files = [('dicom_file', (dcm_file.name, dcm_file, 'application/dicom'))]
            
            data = {
                'prompt': prompt,
                'role_instruction': role_instruction,
                'max_new_tokens': max_tokens
            }
            
            response = requests.post(
                f"{api_url}/api/analyze-dicom",
                files=files,
                data=data,
                timeout=300
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'success': False, 'error': f'API returned status code {response.status_code}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def create_pdf_from_markdown(markdown_text, patient_name="Unknown", study_date=None, images=None, metadata=None):
    """Convert markdown report to PDF with images using reportlab"""
    if study_date is None:
        study_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create PDF file
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    doc = SimpleDocTemplate(pdf_file.name, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for PDF elements
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("🏥 AI Medical Imaging Report", title_style))
    story.append(Spacer(1, 12))
    
    # Patient info
    info_data = [
        ['Patient:', patient_name],
        ['Date:', study_date],
        ['Generated by:', 'AI Reporting System']
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))
    
    # Add DICOM metadata if available
    if metadata:
        story.append(Paragraph("DICOM Information", heading_style))
        meta_data = [
            ['Modality:', metadata.get('Modality', 'N/A')],
            ['Body Part:', metadata.get('BodyPartExamined', 'N/A')],
            ['Study Description:', metadata.get('StudyDescription', 'N/A')],
        ]
        meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 20))
    
    # Convert markdown to paragraphs
    lines = markdown_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        
        if line.startswith('# '):
            story.append(Paragraph(line[2:], heading_style))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['Heading3']))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['Heading4']))
        elif line.startswith('**') and line.endswith('**'):
            story.append(Paragraph(f"<b>{line[2:-2]}</b>", styles['Normal']))
        elif line.startswith('- ') or line.startswith('* '):
            story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
        else:
            story.append(Paragraph(line, styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Add images if provided
    if images and len(images) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Analyzed Images", heading_style))
        story.append(Spacer(1, 12))
        
        for idx, img in enumerate(images):
            try:
                # Save image to temporary file
                img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                img.save(img_temp.name, 'PNG')
                
                # Add to PDF with max width
                img_width = 4*inch
                img_height = img.height * (img_width / img.width)
                if img_height > 5*inch:
                    img_height = 5*inch
                    img_width = img.width * (img_height / img.height)
                
                story.append(Paragraph(f"Image {idx + 1}", styles['Heading4']))
                story.append(Spacer(1, 6))
                story.append(RLImage(img_temp.name, width=img_width, height=img_height))
                story.append(Spacer(1, 12))
                
                # Clean up temp file
                os.unlink(img_temp.name)
            except Exception as e:
                story.append(Paragraph(f"Error adding image {idx + 1}: {str(e)}", styles['Normal']))
    
    # Disclaimer
    story.append(PageBreak())
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#856404'),
        leftIndent=20,
        rightIndent=20
    )
    story.append(Paragraph("<b>⚠️ Disclaimer:</b>", styles['Heading4']))
    story.append(Paragraph(
        "This report is generated by an AI system and is for informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)
    
    return pdf_file.name

# Main App Header with modern design
st.markdown('''
<div style="text-align: center; padding: 2rem 0;">
    <div class="main-header">🏥 AI Medical Reporting System</div>
    <div class="sub-header">Powered by Advanced AI Classification & Reporting Models</div>
</div>
''', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL Configuration
    st.subheader("🤖 Reporting AI API")
    api_url = st.text_input(
        "Reporting API URL",
        value=st.session_state.api_url,
        help="Enter the URL of your AI Reporting API"
    )
    st.session_state.api_url = api_url
    
    # Check API Health
    if st.button("🔍 Check Reporting API Status"):
        with st.spinner("Checking API..."):
            is_healthy, health_data = check_api_health(api_url)
            if is_healthy:
                st.success("✅ Reporting API is online and ready!")
                if health_data:
                    st.json(health_data)
            else:
                st.error("❌ Reporting API is not available. Please check the URL and ensure the API is running.")
    
    st.divider()
    
    # Classification API Configuration
    st.subheader("🔬 Classification API")
    
    enable_classification = st.checkbox(
        "Enable Pre-Classification",
        value=st.session_state.enable_classification,
        help="Run classification model before AI reporting"
    )
    st.session_state.enable_classification = enable_classification
    
    classification_api_url = st.text_input(
        "Classification API URL",
        value=st.session_state.classification_api_url,
        help="Enter the URL of your Classification API (MAZIK)",
        disabled=not enable_classification
    )
    st.session_state.classification_api_url = classification_api_url
    
    if enable_classification:
        # Model selection for PNG images
        model_type = st.radio(
            "Classification Model",
            options=["Auto-Detect", "Brain Tumor", "Mammography"],
            help="For PNG images: choose model type. For DICOM: automatically detected from metadata.",
            horizontal=True
        )
        
        # Map to internal values
        model_type_map = {
            "Auto-Detect": "auto",
            "Brain Tumor": "brain",
            "Mammography": "mammography"
        }
        st.session_state.classification_model_type = model_type_map[model_type]
        
        if st.button("🔍 Check Classification API Status"):
            with st.spinner("Checking Classification API..."):
                try:
                    classifier = ClassificationClient(classification_api_url)
                    is_healthy, health_data = classifier.check_health()
                    if is_healthy:
                        st.success("✅ Classification API is online!")
                        st.json(health_data)
                    else:
                        st.error("❌ Classification API is not available.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        role_instruction = st.text_area(
            "Role Instruction",
            value=DEFAULT_ROLE_INSTRUCTION,
            help="Customize the AI's role and behavior"
        )
        
        max_tokens = st.slider(
            "Max Response Tokens",
            min_value=MAX_TOKENS_MIN,
            max_value=MAX_TOKENS_MAX,
            value=MAX_TOKENS_DEFAULT,
            step=256,
            help="Maximum length of the AI response"
        )
    
    st.divider()
    
    # Information
    st.info("""
    **How to use:**
    1. Select image type (PNG or DICOM)
    2. Upload your medical images
    3. Enter your analysis prompt
    4. Click Analyze
    5. View and download results
    """)

# Main Content Area
tab1, tab2 = st.tabs(["📊 Analysis", "📄 Results"])

with tab1:
    # Image Type Selection
    st.markdown("### 📁 Select Image Type")
    image_type = st.radio(
        "Choose the type of medical images you want to analyze:",
        options=["PNG/JPEG Images", "DICOM Files"],
        horizontal=True,
        help="Select PNG/JPEG for standard images or DICOM for medical imaging files"
    )
    
    st.divider()
    
    if image_type == "PNG/JPEG Images":
        st.markdown("### 🖼️ Upload PNG/JPEG Images")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload one or more medical images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="You can upload multiple images for analysis"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
            
            # Store images in session state
            st.session_state.uploaded_images = []
            for uploaded_file in uploaded_files:
                uploaded_file.seek(0)
                img = Image.open(uploaded_file)
                st.session_state.uploaded_images.append(img)
            
            # Display image previews
            cols = st.columns(min(len(uploaded_files), 4))
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 4]:
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prompt Input
        st.markdown("### 💬 Analysis Prompt")
        prompt = st.text_area(
            "Enter your question or analysis request:",
            value=DEFAULT_PROMPT_PNG,
            height=150,
            help="Describe what you want the AI to analyze"
        )
        
        # Analyze Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔬 Analyze Images", use_container_width=True)
        
        if analyze_button:
            if not uploaded_files:
                st.error("❌ Please upload at least one image before analyzing.")
            elif not prompt.strip():
                st.error("❌ Please enter an analysis prompt.")
            else:
                # Check API health first
                is_healthy, _ = check_api_health(api_url)
                if not is_healthy:
                    st.error("❌ API is not available. Please check the API URL and ensure it's running.")
                else:
                    # Load images
                    images = [Image.open(f) for f in uploaded_files]
                    
                    # Step 1: Classification (if enabled)
                    classification_results = []
                    if st.session_state.enable_classification:
                        with st.spinner("🔬 Step 1/2: Running classification model..."):
                            try:
                                classifier = ClassificationClient(st.session_state.classification_api_url)
                                
                                # Classify each image with selected model type
                                for idx, img in enumerate(images):
                                    st.write(f"Classifying image {idx+1}/{len(images)}...")
                                    class_result = classifier.auto_classify(
                                        img, 
                                        image_type=st.session_state.classification_model_type
                                    )
                                    classification_results.append(class_result)
                                
                                # Store classification results
                                st.session_state.classification_results = classification_results
                                
                                # Show classification summary
                                st.success(f"✅ Classification complete for {len(images)} image(s)")
                                with st.expander("📊 Classification Results", expanded=True):
                                    for idx, class_res in enumerate(classification_results):
                                        if class_res['success']:
                                            pred = class_res['prediction']
                                            st.write(f"**Image {idx+1}:**")
                                            st.write(f"- Class: {pred['predicted_class'].replace('_', ' ')}")
                                            st.write(f"- Confidence: {pred['confidence']*100:.1f}%")
                                        else:
                                            st.write(f"**Image {idx+1}:** Classification failed - {class_res.get('error', 'Unknown error')}")
                            except Exception as e:
                                st.warning(f"⚠️ Classification failed: {str(e)}")
                                st.info("Proceeding with AI reporting without classification...")
                    
                    # Step 2: AI Reporting
                    with st.spinner("🤖 Step 2/2: Generating AI medical report..."):
                        # Enhance prompt with classification results
                        enhanced_prompt = prompt
                        if classification_results and st.session_state.enable_classification:
                            classification_context = "\n\n**Classification Pre-Analysis:**\n"
                            for idx, class_res in enumerate(classification_results):
                                if class_res['success']:
                                    classifier_client = ClassificationClient(st.session_state.classification_api_url)
                                    classification_context += f"\nImage {idx+1}:\n"
                                    classification_context += classifier_client.format_classification_for_prompt(class_res)
                            
                            enhanced_prompt = classification_context + "\n" + prompt
                        
                        # Call Reporting API
                        result = analyze_png_images(
                            api_url,
                            images,
                            enhanced_prompt,
                            role_instruction,
                            max_tokens
                        )
                        
                        # Store result
                        st.session_state.analysis_result = result
                        
                        if result.get('success'):
                            st.success("✅ Analysis completed successfully!")
                            st.info("👉 **Switch to the 'Results' tab above to view the detailed analysis report.**")
                            
                            # Display a preview of the results here
                            with st.expander("📊 Preview Results (Click to expand)", expanded=True):
                                answer = result.get('answer') or result.get('response', '')
                                if answer:
                                    st.markdown(answer[:500] + "..." if len(answer) > 500 else answer)
                                    st.info("👆 This is a preview. See the 'Results' tab for the complete report.")
                            
                            st.balloons()
                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    else:  # DICOM Files
        st.markdown("### 🏥 Upload DICOM Files")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        # DICOM type selection
        dicom_type = st.radio(
            "DICOM Analysis Type:",
            options=["Single DICOM File", "DICOM Series (Multiple Files)"],
            horizontal=True,
            help="Choose single file for one image or series for multiple related DICOM files"
        )
        
        uploaded_dicom = st.file_uploader(
            "Upload DICOM file(s)",
            type=['dcm', 'dicom'],
            accept_multiple_files=(dicom_type == "DICOM Series (Multiple Files)"),
            help="Upload .dcm or .dicom files"
        )
        
        if uploaded_dicom:
            if isinstance(uploaded_dicom, list):
                st.success(f"✅ {len(uploaded_dicom)} DICOM file(s) uploaded successfully!")
                
                # Convert and store DICOM images
                st.session_state.uploaded_images = []
                dicom_files_list = uploaded_dicom
                
                # Extract metadata from first file
                first_img, first_ds = dicom_to_pil(dicom_files_list[0])
                if first_ds:
                    st.session_state.dicom_metadata = extract_dicom_metadata(first_ds)
                    
                    # Show metadata
                    with st.expander("📋 DICOM Metadata", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Modality:** {st.session_state.dicom_metadata.get('Modality', 'N/A')}")
                            st.write(f"**Body Part:** {st.session_state.dicom_metadata.get('BodyPartExamined', 'N/A')}")
                            st.write(f"**Study Date:** {st.session_state.dicom_metadata.get('StudyDate', 'N/A')}")
                        with col2:
                            st.write(f"**Patient Age:** {st.session_state.dicom_metadata.get('PatientAge', 'N/A')}")
                            st.write(f"**Patient Sex:** {st.session_state.dicom_metadata.get('PatientSex', 'N/A')}")
                            st.write(f"**Series:** {st.session_state.dicom_metadata.get('SeriesDescription', 'N/A')}")
                
                # Show previews
                st.markdown("**Preview of DICOM Images:**")
                cols = st.columns(min(len(dicom_files_list), 4))
                for idx, dcm_file in enumerate(dicom_files_list):
                    img, ds = dicom_to_pil(dcm_file)
                    if img:
                        st.session_state.uploaded_images.append(img)
                        with cols[idx % 4]:
                            st.image(img, caption=f"Slice {idx+1}", use_container_width=True)
            else:
                st.success(f"✅ DICOM file uploaded: {uploaded_dicom.name}")
                
                # Convert and store single DICOM
                img, ds = dicom_to_pil(uploaded_dicom)
                if img and ds:
                    st.session_state.uploaded_images = [img]
                    st.session_state.dicom_metadata = extract_dicom_metadata(ds)
                    
                    # Show metadata
                    with st.expander("📋 DICOM Metadata", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Modality:** {st.session_state.dicom_metadata.get('Modality', 'N/A')}")
                            st.write(f"**Body Part:** {st.session_state.dicom_metadata.get('BodyPartExamined', 'N/A')}")
                            st.write(f"**Study Date:** {st.session_state.dicom_metadata.get('StudyDate', 'N/A')}")
                        with col2:
                            st.write(f"**Patient Age:** {st.session_state.dicom_metadata.get('PatientAge', 'N/A')}")
                            st.write(f"**Patient Sex:** {st.session_state.dicom_metadata.get('PatientSex', 'N/A')}")
                            st.write(f"**Series:** {st.session_state.dicom_metadata.get('SeriesDescription', 'N/A')}")
                    
                    # Show preview
                    st.markdown("**Preview:**")
                    st.image(img, caption="DICOM Image", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prompt Input
        st.markdown("### 💬 Analysis Prompt")
        prompt = st.text_area(
            "Enter your question or analysis request:",
            value=DEFAULT_PROMPT_DICOM,
            height=150,
            help="Describe what you want the AI to analyze"
        )
        
        # Analyze Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔬 Analyze DICOM", use_container_width=True)
        
        if analyze_button:
            if not uploaded_dicom:
                st.error("❌ Please upload DICOM file(s) before analyzing.")
            elif not prompt.strip():
                st.error("❌ Please enter an analysis prompt.")
            else:
                # Check API health first
                is_healthy, _ = check_api_health(api_url)
                if not is_healthy:
                    st.error("❌ API is not available. Please check the API URL and ensure it's running.")
                else:
                    # Prepare files
                    if isinstance(uploaded_dicom, list):
                        dicom_files = uploaded_dicom
                        is_series = True
                    else:
                        dicom_files = [uploaded_dicom]
                        is_series = (dicom_type == "DICOM Series (Multiple Files)")
                    
                    # Step 1: Classification (if enabled and images available)
                    classification_results = []
                    if st.session_state.enable_classification and st.session_state.uploaded_images:
                        with st.spinner("🔬 Step 1/2: Running classification on DICOM images..."):
                            try:
                                classifier = ClassificationClient(st.session_state.classification_api_url)
                                
                                # Classify each DICOM image using intelligent metadata-based selection
                                for idx, img in enumerate(st.session_state.uploaded_images):
                                    st.write(f"Classifying slice {idx+1}/{len(st.session_state.uploaded_images)}...")
                                    # Use metadata from first file for all slices (series should be same modality)
                                    if st.session_state.dicom_metadata:
                                        class_result = classifier.intelligent_classify_from_metadata(
                                            img, 
                                            st.session_state.dicom_metadata
                                        )
                                    else:
                                        # Fallback to auto if no metadata
                                        class_result = classifier.auto_classify(img, image_type='auto')
                                    classification_results.append(class_result)
                                
                                # Store classification results
                                st.session_state.classification_results = classification_results
                                
                                # Show classification summary
                                st.success(f"✅ Classification complete for {len(st.session_state.uploaded_images)} slice(s)")
                                with st.expander("📊 Classification Results", expanded=True):
                                    for idx, class_res in enumerate(classification_results):
                                        if class_res['success']:
                                            pred = class_res['prediction']
                                            st.write(f"**Slice {idx+1}:**")
                                            st.write(f"- Class: {pred['predicted_class'].replace('_', ' ')}")
                                            st.write(f"- Confidence: {pred['confidence']*100:.1f}%")
                                        else:
                                            st.write(f"**Slice {idx+1}:** Classification failed")
                            except Exception as e:
                                st.warning(f"⚠️ Classification failed: {str(e)}")
                                st.info("Proceeding with AI reporting without classification...")
                    
                    # Step 2: AI Reporting
                    with st.spinner("🤖 Step 2/2: Generating AI medical report..."):
                        # Start with DICOM metadata context
                        enhanced_prompt = prompt
                        if st.session_state.dicom_metadata:
                            enhanced_prompt = create_dicom_context_prompt(
                                st.session_state.dicom_metadata,
                                prompt
                            )
                        
                        # Add classification results
                        if classification_results and st.session_state.enable_classification:
                            classification_context = "\n\n**Classification Pre-Analysis:**\n"
                            for idx, class_res in enumerate(classification_results):
                                if class_res['success']:
                                    classifier_client = ClassificationClient(st.session_state.classification_api_url)
                                    classification_context += f"\nSlice {idx+1}:\n"
                                    classification_context += classifier_client.format_classification_for_prompt(class_res)
                            
                            enhanced_prompt = enhanced_prompt + "\n" + classification_context
                        
                        # Call Reporting API
                        result = analyze_dicom_files(
                            api_url,
                            dicom_files,
                            enhanced_prompt,
                            role_instruction,
                            max_tokens,
                            is_series
                        )
                        
                        # Store result
                        st.session_state.analysis_result = result
                        
                        if result.get('success'):
                            st.success("✅ Analysis completed successfully!")
                            st.info("👉 **Switch to the 'Results' tab above to view the detailed analysis report.**")
                            
                            # Display a preview of the results here
                            with st.expander("📊 Preview Results (Click to expand)", expanded=True):
                                if 'slice_results' in result:
                                    st.markdown(f"**DICOM Series Analysis Complete**")
                                    st.markdown(f"- Total Slices: {result.get('total_slices', 0)}")
                                    st.markdown(f"- Analyzed: {result.get('analyzed_slices', 0)}")
                                    if result.get('summary'):
                                        st.markdown("**Summary Preview:**")
                                        summary = result['summary']
                                        st.markdown(summary[:500] + "..." if len(summary) > 500 else summary)
                                else:
                                    answer = result.get('answer') or result.get('response', '')
                                    if answer:
                                        st.markdown(answer[:500] + "..." if len(answer) > 500 else answer)
                                st.info("👆 This is a preview. See the 'Results' tab for the complete report.")
                            
                            st.balloons()
                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

with tab2:
    st.markdown("### 📊 Analysis Results")
    
    if st.session_state.analysis_result is None:
        # st.info("👈 Please perform an analysis first to see results here.")
        pass
    else:
        result = st.session_state.analysis_result
        
        if result.get('success'):
            # Display Classification Results First (if available)
            if st.session_state.classification_results and st.session_state.enable_classification:
                st.markdown("#### 🔬 Classification Results")
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                
                # Summary statistics
                total_classified = len(st.session_state.classification_results)
                successful = sum(1 for r in st.session_state.classification_results if r.get('success'))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", total_classified)
                with col2:
                    st.metric("Successfully Classified", successful)
                with col3:
                    avg_conf = np.mean([r['prediction']['confidence'] for r in st.session_state.classification_results if r.get('success')]) * 100 if successful > 0 else 0
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                
                # Detailed results
                st.markdown("---")
                for idx, class_res in enumerate(st.session_state.classification_results):
                    if class_res.get('success'):
                        pred = class_res['prediction']
                        model_type = class_res.get('model_type', 'Unknown')
                        
                        with st.expander(f"🖼️ Image {idx+1} - {pred['predicted_class'].replace('_', ' ')}", expanded=False):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if idx < len(st.session_state.uploaded_images):
                                    st.image(st.session_state.uploaded_images[idx], use_container_width=True)
                            
                            with col2:
                                st.markdown(f"**Model Type:** {model_type.replace('_', ' ').title()}")
                                st.markdown(f"**Predicted Class:** {pred['predicted_class'].replace('_', ' ')}")
                                st.markdown(f"**Confidence:** {pred['confidence']*100:.1f}%")
                                st.markdown(f"**Is Confident:** {'Yes ✅' if pred.get('is_confident', False) else 'No ⚠️'}")
                                
                                # Show probability distribution
                                st.markdown("**Class Probabilities:**")
                                probs = pred.get('class_probabilities', {})
                                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                                    st.progress(prob, text=f"{class_name.replace('_', ' ')}: {prob*100:.1f}%")
                    else:
                        with st.expander(f"🖼️ Image {idx+1} - Classification Failed", expanded=False):
                            st.error(f"Error: {class_res.get('error', 'Unknown error')}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.divider()
            
            # Display AI Reporting Results
            st.markdown("#### 🤖 AI Medical Report")
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            
            # Check if it's a DICOM series result
            if 'slice_results' in result:
                # DICOM Series Result
                st.markdown("#### 📋 DICOM Series Analysis")
                
                # Patient Info
                if result.get('patient_info'):
                    with st.expander("👤 Patient Information", expanded=True):
                        patient_info = result['patient_info']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Patient ID:** {patient_info.get('patient_id', 'N/A')}")
                            st.write(f"**Patient Name:** {patient_info.get('patient_name', 'N/A')}")
                        with col2:
                            st.write(f"**Study Date:** {patient_info.get('study_date', 'N/A')}")
                            st.write(f"**Modality:** {patient_info.get('modality', 'N/A')}")
                        st.write(f"**Study Description:** {patient_info.get('study_description', 'N/A')}")
                
                # Summary
                if result.get('summary'):
                    st.markdown("#### 📝 Overall Summary")
                    st.markdown(result['summary'])
                    st.divider()
                
                # Individual Slice Results
                st.markdown(f"#### 🔬 Individual Slice Analyses ({result.get('analyzed_slices', 0)}/{result.get('total_slices', 0)} slices)")
                
                for slice_result in result.get('slice_results', []):
                    with st.expander(f"Slice {slice_result['slice_number']} (Instance: {slice_result.get('instance_number', 'N/A')})", expanded=False):
                        st.markdown(slice_result['analysis'])
                
                # Prepare markdown for PDF
                markdown_content = f"# DICOM Series Analysis Report\n\n"
                if result.get('patient_info'):
                    pi = result['patient_info']
                    markdown_content += f"**Patient ID:** {pi.get('patient_id', 'N/A')}  \n"
                    markdown_content += f"**Patient Name:** {pi.get('patient_name', 'N/A')}  \n"
                    markdown_content += f"**Study Date:** {pi.get('study_date', 'N/A')}  \n"
                    markdown_content += f"**Modality:** {pi.get('modality', 'N/A')}  \n"
                    markdown_content += f"**Study Description:** {pi.get('study_description', 'N/A')}  \n\n"
                
                if result.get('summary'):
                    markdown_content += f"## Overall Summary\n\n{result['summary']}\n\n"
                
                markdown_content += f"## Individual Slice Analyses\n\n"
                for slice_result in result.get('slice_results', []):
                    markdown_content += f"### Slice {slice_result['slice_number']} (Instance: {slice_result.get('instance_number', 'N/A')})\n\n"
                    markdown_content += f"{slice_result['analysis']}\n\n"
                
                patient_name = result.get('patient_info', {}).get('patient_name', 'Unknown')
                study_date = result.get('patient_info', {}).get('study_date', None)
                
            else:
                # Single Image Result
                st.markdown("#### 📝 Analysis Report")
                
                # Display the answer/response
                answer = result.get('answer') or result.get('response', '')
                
                if answer:
                    # Display in a nice formatted box
                    st.markdown("---")
                    st.markdown(answer)
                    st.markdown("---")
                else:
                    st.warning("⚠️ No analysis text found in the response.")
                    st.json(result)  # Show raw result for debugging
                
                # Show thought process if available
                if result.get('thought'):
                    with st.expander("🧠 AI Thought Process", expanded=False):
                        st.markdown(result['thought'])
                
                # Show analyzed images
                if st.session_state.uploaded_images and len(st.session_state.uploaded_images) > 0:
                    st.markdown("---")
                    st.markdown("#### 🖼️ Analyzed Images")
                    cols = st.columns(min(len(st.session_state.uploaded_images), 3))
                    for idx, img in enumerate(st.session_state.uploaded_images):
                        with cols[idx % 3]:
                            st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                
                # Prepare markdown for PDF
                markdown_content = f"# Medical Image Analysis Report\n\n"
                markdown_content += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # Add DICOM metadata if available
                if st.session_state.dicom_metadata:
                    markdown_content += f"## DICOM Information\n\n"
                    markdown_content += f"- **Modality:** {st.session_state.dicom_metadata.get('Modality', 'N/A')}\n"
                    markdown_content += f"- **Body Part:** {st.session_state.dicom_metadata.get('BodyPartExamined', 'N/A')}\n"
                    markdown_content += f"- **Study Description:** {st.session_state.dicom_metadata.get('StudyDescription', 'N/A')}\n\n"
                
                markdown_content += f"## Findings\n\n{answer}\n\n"
                
                if result.get('thought'):
                    markdown_content += f"## AI Reasoning Process\n\n{result['thought']}\n\n"
                
                patient_name = "Patient"
                study_date = None
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download Options
            st.divider()
            st.markdown("### 💾 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as Markdown
                st.download_button(
                    label="📄 Download as Markdown",
                    data=markdown_content,
                    file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                # Download as PDF with images
                if st.button("📑 Generate & Download PDF", use_container_width=True):
                    with st.spinner("Generating PDF with images..."):
                        try:
                            # Get images and metadata
                            images_for_pdf = st.session_state.uploaded_images if st.session_state.uploaded_images else None
                            metadata_for_pdf = st.session_state.dicom_metadata if st.session_state.dicom_metadata else None
                            
                            pdf_path = create_pdf_from_markdown(
                                markdown_content, 
                                patient_name, 
                                study_date,
                                images=images_for_pdf,
                                metadata=metadata_for_pdf
                            )
                            
                            with open(pdf_path, 'rb') as pdf_file:
                                pdf_bytes = pdf_file.read()
                            
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_bytes,
                                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            # Clean up temp file
                            os.unlink(pdf_path)
                            
                            st.success("✅ PDF generated successfully with images!")
                        except Exception as e:
                            st.error(f"❌ Error generating PDF: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
        
        else:
            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            st.info("Please check your API connection and try again.")

# Footer
st.divider()
st.markdown(FOOTER_TEXT, unsafe_allow_html=True)

