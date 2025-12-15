// Main JavaScript for Fair-or-Foul Web Application

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupDragAndDrop();
    loadUploadedFiles();
    setupEventListeners();
    setupSmoothScrolling();
}

// Drag and Drop Functionality
function setupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('file');

    if (!dropZone || !fileInput) return;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    // Handle click to browse
    dropZone.addEventListener('click', () => fileInput.click());
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    document.getElementById('dropZone').classList.add('dragover');
}

function unhighlight(e) {
    document.getElementById('dropZone').classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        document.getElementById('file').files = files;
        updateFileDisplay(files[0]);
    }
}

function updateFileDisplay(file) {
    const dropZone = document.getElementById('dropZone');
    const content = dropZone.querySelector('.drop-zone-content');
    
    content.innerHTML = `
        <i class="fas fa-file fa-3x text-primary mb-3"></i>
        <p class="mb-2 fw-bold">${file.name}</p>
        <p class="text-muted small">Size: ${formatFileSize(file.size)}</p>
    `;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// File Management
function loadUploadedFiles() {
    fetch('/get_uploaded_files')
        .then(response => response.json())
        .then(files => {
            displayUploadedFiles(files);
            updateFileSelectors(files);
        })
        .catch(error => {
            console.error('Error loading uploaded files:', error);
        });
}

function displayUploadedFiles(files) {
    const container = document.getElementById('uploadedFiles');
    if (!container) return;

    if (files.length === 0) {
        container.innerHTML = '<div class="col-12 text-center text-muted"><p>No files uploaded yet</p></div>';
        return;
    }

    container.innerHTML = files.map(file => createFileCard(file)).join('');
}

function createFileCard(file) {
    const fileType = getFileType(file.name);
    const iconClass = getFileIcon(fileType);
    const iconBgClass = getFileIconBg(fileType);
    
    return `
        <div class="col-md-6 col-lg-4">
            <div class="file-card">
                <div class="d-flex align-items-center">
                    <div class="file-icon ${iconBgClass} me-3">
                        <i class="${iconClass}"></i>
                    </div>
                    <div class="flex-grow-1">
                        <h6 class="mb-1 text-truncate">${file.name}</h6>
                        <p class="mb-1 small text-muted">${formatFileSize(file.size)}</p>
                        <span class="badge bg-secondary">${fileType.toUpperCase()}</span>
                    </div>
                    <div class="ms-2">
                        <button class="btn btn-sm btn-outline-primary" onclick="downloadFile('${file.name}')">
                            <i class="fas fa-download"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function getFileType(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    if (['mp4', 'avi', 'mov', 'mkv'].includes(ext)) return 'video';
    if (ext === 'csv') return 'csv';
    return 'unknown';
}

function getFileIcon(fileType) {
    switch (fileType) {
        case 'csv': return 'fas fa-file-csv';
        case 'video': return 'fas fa-video';
        default: return 'fas fa-file';
    }
}

function getFileIconBg(fileType) {
    switch (fileType) {
        case 'csv': return 'csv';
        case 'video': return 'video';
        default: return 'unknown';
    }
}

function updateFileSelectors(files) {
    const csvFiles = files.filter(f => f.type === 'csv');
    
    // Update team rates file selector
    const teamRatesSelect = document.getElementById('teamRatesFile');
    if (teamRatesSelect) {
        updateFileSelector(teamRatesSelect, csvFiles);
    }
    
    // Update county alignment file selector
    const countyAlignmentSelect = document.getElementById('countyAlignmentFile');
    if (countyAlignmentSelect) {
        updateFileSelector(countyAlignmentSelect, csvFiles);
    }
}

function updateFileSelector(selectElement, files) {
    const currentValue = selectElement.value;
    selectElement.innerHTML = '<option value="">Choose a file...</option>';
    
    files.forEach(file => {
        const option = document.createElement('option');
        option.value = file.path;
        option.textContent = file.name;
        selectElement.appendChild(option);
    });
    
    if (currentValue) {
        selectElement.value = currentValue;
    }
}

// Analysis Functions
function runAnalysis(analysisType) {
    let filePath;
    
    if (analysisType === 'team_rates') {
        filePath = document.getElementById('teamRatesFile').value;
    } else if (analysisType === 'county_alignment') {
        filePath = document.getElementById('countyAlignmentFile').value;
    }
    
    if (!filePath) {
        showAlert('Please select a CSV file first', 'warning');
        return;
    }
    
    // Show loading state
    const button = event.target;
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
    button.disabled = true;
    
    // Make API call
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            csv_path: filePath,
            analysis_type: analysisType
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayAnalysisResults(data.data, analysisType);
            showAlert('Analysis completed successfully!', 'success');
        } else {
            showAlert('Analysis failed: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('An error occurred during analysis', 'danger');
    })
    .finally(() => {
        // Restore button state
        button.innerHTML = originalText;
        button.disabled = false;
    });
}

function displayAnalysisResults(data, analysisType) {
    const resultsDiv = document.getElementById('analysisResults');
    const resultsTable = document.getElementById('resultsTable');
    
    if (!resultsDiv || !resultsTable) return;
    
    // Create table
    let tableHTML = '<div class="table-responsive results-table">';
    tableHTML += '<table class="table table-striped">';
    
    // Table headers
    if (data.length > 0) {
        const headers = Object.keys(data[0]);
        tableHTML += '<thead><tr>';
        headers.forEach(header => {
            tableHTML += `<th>${formatHeader(header)}</th>`;
        });
        tableHTML += '</tr></thead>';
        
        // Table body
        tableHTML += '<tbody>';
        data.forEach(row => {
            tableHTML += '<tr>';
            headers.forEach(header => {
                tableHTML += `<td>${row[header]}</td>`;
            });
            tableHTML += '</tr>';
        });
        tableHTML += '</tbody>';
    }
    
    tableHTML += '</table></div>';
    
    resultsTable.innerHTML = tableHTML;
    resultsDiv.style.display = 'block';
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function formatHeader(header) {
    return header
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Utility Functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the page
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

function downloadFile(filename) {
    window.open(`/download/${filename}`, '_blank');
}

function downloadResults() {
    // This would download the processed results
    showAlert('Download functionality coming soon!', 'info');
}

// Event Listeners
function setupEventListeners() {
    // File input change
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                updateFileDisplay(e.target.files[0]);
            }
        });
    }
    
    // Form submission
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file');
            const sportSelect = document.getElementById('sport');
            
            if (!fileInput.files.length) {
                e.preventDefault();
                showAlert('Please select a file to upload', 'warning');
                return;
            }
            
            if (!sportSelect.value) {
                e.preventDefault();
                showAlert('Please select a sport type', 'warning');
                return;
            }
        });
    }

    // Safe binding for pose detection controls (avoid relying on inline onclick)
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resetBtn = document.getElementById('resetBtn');
    const calibrateBtn = document.getElementById('calibrateBtn');
    const resetCalBtn = document.getElementById('resetCalBtn');
    const overlaySelect = document.getElementById('overlayModeSelect');

    if (startBtn) startBtn.addEventListener('click', () => {
        if (window.startPoseDetection) return startPoseDetection();
        if (window.poseDetector && typeof window.poseDetector.start === 'function') return window.poseDetector.start();
        console.error('Pose detector not ready');
    });

    if (stopBtn) stopBtn.addEventListener('click', () => {
        if (window.stopPoseDetection) return stopPoseDetection();
        if (window.poseDetector && typeof window.poseDetector.stop === 'function') return window.poseDetector.stop();
        console.error('Pose detector not ready');
    });

    if (resetBtn) resetBtn.addEventListener('click', () => {
        if (window.resetDetection) return resetDetection();
        if (window.poseDetector && typeof window.poseDetector.reset === 'function') return window.poseDetector.reset();
        console.error('Pose detector not ready');
    });

    if (calibrateBtn) calibrateBtn.addEventListener('click', () => {
        if (window.poseDetector && typeof window.poseDetector.startCalibration === 'function') {
            return window.poseDetector.startCalibration();
        }
        console.error('Pose detector not ready for calibration');
    });

    if (resetCalBtn) resetCalBtn.addEventListener('click', () => {
        if (window.poseDetector && typeof window.poseDetector.resetCalibration === 'function') {
            return window.poseDetector.resetCalibration();
        }
        console.error('Pose detector not ready for calibration reset');
    });

    if (overlaySelect) overlaySelect.addEventListener('change', (e) => {
        const v = e.target.value;
        if (window.setOverlayMode) return setOverlayMode(v);
        if (window.poseDetector && typeof window.poseDetector.setOverlayMode === 'function') return window.poseDetector.setOverlayMode(v);
    });
}

// Smooth Scrolling
function setupSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Navigation Active State
function updateNavigation() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (window.pageYOffset >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
}

// Update navigation on scroll
window.addEventListener('scroll', updateNavigation);

// Refresh file list after successful upload
function refreshFileList() {
    setTimeout(() => {
        loadUploadedFiles();
    }, 1000);
}

// Auto-refresh file list every 30 seconds
setInterval(loadUploadedFiles, 30000);
