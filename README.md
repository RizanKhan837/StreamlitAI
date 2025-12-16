# AI Medical Reporting - Streamlit Application

A beautiful and intuitive Streamlit application for medical image analysis using the MedGemma AI model.

## Features

✨ **Beautiful Modern UI**
- Clean, professional interface with custom styling
- Responsive design with intuitive navigation
- Real-time status indicators and progress feedback

🖼️ **Multiple Image Format Support**
- PNG/JPEG images for standard medical images
- DICOM files for medical imaging (single or series)
- Multiple file upload support
- Image preview functionality

🔬 **Comprehensive Analysis**
- Detailed medical image analysis
- Support for single images or complete DICOM series
- Patient information extraction from DICOM metadata
- Summary generation for DICOM series

📄 **Professional Reporting**
- Beautiful markdown-formatted results
- Download reports as Markdown (.md)
- Generate professional PDF reports with:
  - Custom headers and footers
  - Patient information
  - Formatted medical findings
  - Disclaimer sections

⚙️ **Flexible Configuration**
- Configurable API endpoint
- Adjustable AI parameters (role instruction, max tokens)
- API health monitoring
- Advanced settings panel

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Install WeasyPrint Dependencies

**Windows:**
```bash
# Download and install GTK3 runtime from:
# https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3-pip python3-cffi python3-brotli libpango-1.0-0 libpangoft2-1.0-0
```

**macOS:**
```bash
brew install python3 cairo pango gdk-pixbuf libffi
```

## Usage

### 1. Start Your API Server

First, ensure your MedGemma API is running:

```bash
python api_app_latest.py
```

Note the API URL (e.g., `http://localhost:8000` or the ngrok URL).

### 2. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### 3. Configure API Connection

1. In the sidebar, enter your API URL
2. Click "Check API Status" to verify connection
3. Adjust advanced settings if needed (role instruction, max tokens)

### 4. Analyze Images

#### For PNG/JPEG Images:

1. Select "PNG/JPEG Images" option
2. Upload one or more image files
3. Enter your analysis prompt (or use the default)
4. Click "Analyze Images"
5. View results in the "Results" tab

#### For DICOM Files:

1. Select "DICOM Files" option
2. Choose "Single DICOM File" or "DICOM Series"
3. Upload your DICOM file(s)
4. Enter your analysis prompt
5. Click "Analyze DICOM"
6. View results with patient info and slice-by-slice analysis

### 5. Download Reports

After analysis:
1. Go to the "Results" tab
2. Review the formatted report
3. Download as Markdown or generate PDF
4. PDF includes professional formatting with headers/footers

## API Endpoints Used

The application interfaces with the following API endpoints:

- `/health` - Check API status
- `/api/analyze-with-images` - Analyze PNG/JPEG images
- `/api/analyze-dicom` - Analyze single DICOM file
- `/api/analyze-dicom-series` - Analyze DICOM series

## Configuration Options

### Sidebar Settings:

- **API URL**: Your MedGemma API endpoint
- **Role Instruction**: Customize the AI's behavior and expertise
- **Max Response Tokens**: Control response length (512-4096)

### Default Values:

```python
API_URL = "http://localhost:8000"
ROLE_INSTRUCTION = "You are an expert radiologist analyzing medical images..."
MAX_TOKENS = 2048
```

## Features in Detail

### Image Upload
- Drag and drop support
- Multiple file selection
- Image preview thumbnails
- Format validation

### Analysis Results
- Markdown-formatted medical reports
- Expandable sections for detailed findings
- Patient information display (DICOM)
- Slice-by-slice analysis (DICOM series)
- Overall summary generation

### PDF Generation
- Professional medical report layout
- Custom header with logo and patient info
- Page numbers and footer
- Formatted markdown content
- Medical disclaimer section
- A4 page size with proper margins

### Error Handling
- API connection validation
- File format verification
- Timeout management
- User-friendly error messages
- Graceful degradation

## Troubleshooting

### API Connection Issues

**Problem:** "API is not available" error

**Solutions:**
1. Verify API is running: `curl http://localhost:8000/health`
2. Check firewall settings
3. Ensure correct API URL in sidebar
4. For ngrok URLs, use the HTTPS version

### PDF Generation Issues

**Problem:** WeasyPrint errors

**Solutions:**
1. Ensure GTK3 runtime is installed (Windows)
2. Install system dependencies (Linux/macOS)
3. Check Python version compatibility (3.8+)
4. Try reinstalling: `pip install --force-reinstall weasyprint`

### Image Upload Issues

**Problem:** Images not uploading

**Solutions:**
1. Check file format (PNG, JPEG, DICOM only)
2. Verify file size (default limit: 200MB)
3. Ensure sufficient disk space
4. Check file permissions

### DICOM Issues

**Problem:** DICOM files not processing

**Solutions:**
1. Verify files are valid DICOM format
2. Check file extensions (.dcm or .dicom)
3. Ensure files contain pixel data
4. For series, upload all related files together

## Example Prompts

### For X-rays:
```
Please analyze this chest X-ray and provide:
1. Overall impression
2. Specific findings
3. Differential diagnosis
4. Recommendations for further evaluation
```

### For CT/MRI Series:
```
Please analyze this imaging series and provide:
1. Anatomical structures visible
2. Pathological findings
3. Progression across slices
4. Clinical significance
5. Recommendations
```

### For General Medical Images:
```
Provide a detailed medical assessment including:
- Visual findings
- Possible conditions
- Severity assessment
- Recommended next steps
```

## Security Notes

⚠️ **Important Security Considerations:**

1. **Protected Health Information (PHI)**: This application may process sensitive medical data. Ensure compliance with:
   - HIPAA (US)
   - GDPR (EU)
   - Local healthcare data regulations

2. **Network Security**: 
   - Use HTTPS for production deployments
   - Implement authentication/authorization
   - Secure API endpoints
   - Use VPN for remote access

3. **Data Storage**:
   - Images are processed in memory
   - Temporary files are cleaned up
   - No persistent storage by default
   - Consider encryption for any stored data

4. **Access Control**:
   - Implement user authentication
   - Log access and usage
   - Regular security audits
   - Role-based access control

## Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Production Deployment

**Using Streamlit Cloud:**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Configure secrets for API URL
4. Deploy

**Using Docker:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 libpangoft2-1.0-0 \
    && pip install -r requirements_streamlit.txt
COPY streamlit_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

**Using Nginx (Reverse Proxy):**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Performance Tips

1. **Image Optimization**: Resize large images before upload
2. **Batch Processing**: Analyze multiple images in one request
3. **API Timeout**: Adjust timeout for large DICOM series
4. **Caching**: Enable Streamlit caching for repeated analyses
5. **Resource Limits**: Monitor memory usage with large files

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational and research purposes. Ensure compliance with all applicable medical device regulations and data protection laws before clinical use.

## Support

For issues, questions, or feature requests:
1. Check this README
2. Review API documentation
3. Check application logs
4. Contact support team

## Changelog

### Version 1.0.0
- Initial release
- PNG/JPEG image support
- DICOM file support
- DICOM series analysis
- PDF report generation
- Beautiful UI with custom styling
- API health monitoring
- Configurable settings

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [MedGemma](https://huggingface.co/google/medgemma-4b-it)
- PDF generation by [WeasyPrint](https://weasyprint.org/)
- Medical imaging support via [pydicom](https://pydicom.github.io/)

---

**⚠️ Medical Disclaimer**: This application is an AI-powered tool designed to assist in medical image analysis. It is NOT a substitute for professional medical advice, diagnosis, or treatment. All results should be reviewed and validated by qualified healthcare professionals before any clinical decisions are made.

