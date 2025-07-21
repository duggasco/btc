/**
 * Data Upload Functionality
 */

// Global variables
let selectedFile = null;
let uploadInProgress = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadHandlers();
    setupDragAndDrop();
});

function initializeUploadHandlers() {
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const previewBtn = document.getElementById('preview-btn');
    const uploadBtn = document.getElementById('upload-btn');
    
    // File input change handler
    fileInput.addEventListener('change', handleFileSelect);
    
    // Preview button handler
    previewBtn.addEventListener('click', previewFile);
    
    // Form submit handler
    uploadForm.addEventListener('submit', handleUpload);
    
    // Data type change handler
    document.getElementById('data-type').addEventListener('change', function() {
        updateSymbolPlaceholder(this.value);
    });
}

function setupDragAndDrop() {
    const uploadArea = document.querySelector('.file-upload-area');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('drag-over'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('drag-over'), false);
    });
    
    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        document.getElementById('file-input').files = files;
        handleFileSelect({ target: { files: files } });
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    
    if (!file) {
        selectedFile = null;
        updateUIState(false);
        return;
    }
    
    // Validate file
    const validExtensions = ['csv', 'xlsx'];
    const fileExtension = file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
        showMessage('Invalid file type. Please select a CSV or XLSX file.', 'error');
        e.target.value = '';
        selectedFile = null;
        updateUIState(false);
        return;
    }
    
    // Check file size (100MB limit)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
        showMessage('File too large. Maximum size is 100MB.', 'error');
        e.target.value = '';
        selectedFile = null;
        updateUIState(false);
        return;
    }
    
    selectedFile = file;
    updateUIState(true);
    
    // Update placeholder text
    const placeholder = document.querySelector('.file-upload-placeholder p');
    placeholder.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
}

function updateUIState(fileSelected) {
    const previewBtn = document.getElementById('preview-btn');
    const uploadBtn = document.getElementById('upload-btn');
    
    previewBtn.disabled = !fileSelected;
    uploadBtn.disabled = !fileSelected;
}

function updateSymbolPlaceholder(dataType) {
    const symbolInput = document.getElementById('symbol');
    const helpText = symbolInput.nextElementSibling;
    
    switch(dataType) {
        case 'ohlcv':
            symbolInput.placeholder = 'BTC, GLD, VIX, etc.';
            helpText.textContent = 'For OHLCV data: BTC, GLD, VIX, TNX (10-year), etc.';
            break;
        case 'onchain':
            symbolInput.value = 'BTC';
            symbolInput.placeholder = 'BTC';
            helpText.textContent = 'On-chain data is typically for BTC';
            break;
        case 'sentiment':
            symbolInput.value = 'BTC';
            symbolInput.placeholder = 'BTC';
            helpText.textContent = 'Sentiment data symbol (usually BTC)';
            break;
        case 'macro':
            symbolInput.placeholder = 'DXY, GLD, etc.';
            helpText.textContent = 'Macro indicator symbol';
            break;
    }
}

async function previewFile() {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        showMessage('Loading preview...', 'info');
        
        const response = await fetch('/data/api/preview', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            displayPreview(result.preview);
            showMessage('Preview loaded successfully', 'success');
        } else {
            showMessage(result.error || 'Failed to preview file', 'error');
        }
    } catch (error) {
        console.error('Preview error:', error);
        showMessage('Error previewing file', 'error');
    }
}

function displayPreview(previewData) {
    const previewSection = document.getElementById('preview-section');
    const previewContent = document.getElementById('preview-content');
    
    let html = `
        <div class="preview-info">
            <p><strong>Columns:</strong> ${previewData.columns.join(', ')}</p>
            <p><strong>Total rows in preview:</strong> ${previewData.total_rows}</p>
        </div>
        <div class="table-responsive">
            <table class="table table-sm">
                <thead>
                    <tr>
                        ${previewData.columns.map(col => `<th>${col}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
    `;
    
    previewData.rows.forEach(row => {
        html += '<tr>';
        previewData.columns.forEach(col => {
            html += `<td>${row[col] || ''}</td>`;
        });
        html += '</tr>';
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    previewContent.innerHTML = html;
    previewSection.style.display = 'block';
}

async function handleUpload(e) {
    e.preventDefault();
    
    if (!selectedFile || uploadInProgress) return;
    
    uploadInProgress = true;
    const uploadBtn = document.getElementById('upload-btn');
    const uploadSpinner = document.getElementById('upload-spinner');
    const progressDiv = document.getElementById('upload-progress');
    const progressBar = progressDiv.querySelector('.progress-bar');
    const progressText = progressDiv.querySelector('.progress-text');
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('data_type', document.getElementById('data-type').value);
    formData.append('symbol', document.getElementById('symbol').value);
    formData.append('source', document.getElementById('source').value);
    
    // Update UI
    uploadBtn.disabled = true;
    uploadSpinner.style.display = 'inline-block';
    progressDiv.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = 'Uploading...';
    
    try {
        // Simulate progress for user feedback
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += 10;
                progressBar.style.width = progress + '%';
            }
        }, 200);
        
        const response = await fetch('/data/api/upload', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            progressText.textContent = 'Upload complete!';
            showMessage(result.message || 'Data uploaded successfully', 'success');
            
            // Reset form
            setTimeout(() => {
                document.getElementById('upload-form').reset();
                selectedFile = null;
                updateUIState(false);
                progressDiv.style.display = 'none';
                document.querySelector('.file-upload-placeholder p').textContent = 'Drag and drop a file here or click to browse';
                document.getElementById('preview-section').style.display = 'none';
            }, 2000);
        } else {
            progressText.textContent = 'Upload failed';
            showMessage(result.error || 'Upload failed', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showMessage('Error uploading file', 'error');
        progressText.textContent = 'Upload error';
    } finally {
        uploadInProgress = false;
        uploadBtn.disabled = false;
        uploadSpinner.style.display = 'none';
    }
}

function showMessage(message, type) {
    const messagesDiv = document.getElementById('upload-messages');
    const alertClass = type === 'error' ? 'alert-danger' : 
                      type === 'success' ? 'alert-success' : 
                      'alert-info';
    
    messagesDiv.innerHTML = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = messagesDiv.querySelector('.alert');
        if (alert) {
            alert.classList.remove('show');
            setTimeout(() => alert.remove(), 150);
        }
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}