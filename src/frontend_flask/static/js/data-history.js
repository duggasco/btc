/**
 * Upload History Management
 */

// Global variables
let uploadHistory = [];
let currentPage = 1;
let itemsPerPage = 10;
let currentFilters = {};
let selectedUploadId = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeHistoryHandlers();
    loadUploadHistory();
});

function initializeHistoryHandlers() {
    // Filter controls
    document.getElementById('apply-filters-btn').addEventListener('click', applyFilters);
    document.getElementById('clear-filters-btn').addEventListener('click', clearFilters);
    
    // Refresh button
    document.getElementById('refresh-history-btn').addEventListener('click', loadUploadHistory);
    
    // Rollback confirmation
    document.getElementById('confirm-rollback-btn').addEventListener('click', confirmRollback);
    
    // Set default dates for filter
    const today = new Date();
    const lastMonth = new Date(today.getFullYear(), today.getMonth() - 1, today.getDate());
    
    document.getElementById('filter-date-from').value = lastMonth.toISOString().split('T')[0];
    document.getElementById('filter-date-to').value = today.toISOString().split('T')[0];
}

async function loadUploadHistory(page = 1) {
    try {
        currentPage = page;
        
        // Build query parameters
        const params = new URLSearchParams({
            page: page,
            per_page: itemsPerPage,
            ...currentFilters
        });
        
        const response = await fetch(`/data/api/upload-history?${params}`);
        const data = await response.json();
        
        if (response.ok) {
            uploadHistory = data.uploads || [];
            displayUploadHistory(uploadHistory);
            updatePagination(data.total || 0);
        } else {
            console.error('Failed to load upload history:', data);
            displayEmptyHistory();
        }
    } catch (error) {
        console.error('Error loading upload history:', error);
        displayEmptyHistory();
    }
}

function displayUploadHistory(uploads) {
    const tbody = document.querySelector('#upload-history-table tbody');
    tbody.innerHTML = '';
    
    if (uploads.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center">No upload history found</td></tr>';
        return;
    }
    
    uploads.forEach(upload => {
        const row = document.createElement('tr');
        const statusClass = upload.status === 'completed' ? 'badge-success' : 
                          upload.status === 'failed' ? 'badge-danger' : 
                          'badge-warning';
        
        row.innerHTML = `
            <td><code>${upload.id || 'N/A'}</code></td>
            <td>${formatDateTime(upload.upload_time)}</td>
            <td>${upload.data_type}</td>
            <td>${upload.symbol}</td>
            <td>${upload.source}</td>
            <td>${formatNumber(upload.records_inserted || 0)}</td>
            <td><span class="badge ${statusClass}">${upload.status}</span></td>
            <td>
                <button class="btn btn-sm btn-outline-info" onclick="viewUploadDetails('${upload.id}')">
                    <i class="fas fa-eye"></i>
                </button>
                ${upload.status === 'completed' ? `
                <button class="btn btn-sm btn-outline-danger" onclick="initiateRollback('${upload.id}')">
                    <i class="fas fa-undo"></i>
                </button>
                ` : ''}
            </td>
        `;
        tbody.appendChild(row);
    });
}

function displayEmptyHistory() {
    const tbody = document.querySelector('#upload-history-table tbody');
    tbody.innerHTML = '<tr><td colspan="8" class="text-center">No upload history available</td></tr>';
    updatePagination(0);
}

function updatePagination(totalItems) {
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    const pagination = document.getElementById('history-pagination');
    pagination.innerHTML = '';
    
    if (totalPages <= 1) return;
    
    // Previous button
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `<a class="page-link" href="#" onclick="loadUploadHistory(${currentPage - 1})">Previous</a>`;
    pagination.appendChild(prevLi);
    
    // Page numbers
    for (let i = 1; i <= Math.min(totalPages, 5); i++) {
        const li = document.createElement('li');
        li.className = `page-item ${i === currentPage ? 'active' : ''}`;
        li.innerHTML = `<a class="page-link" href="#" onclick="loadUploadHistory(${i})">${i}</a>`;
        pagination.appendChild(li);
    }
    
    // Next button
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `<a class="page-link" href="#" onclick="loadUploadHistory(${currentPage + 1})">Next</a>`;
    pagination.appendChild(nextLi);
}

function applyFilters() {
    currentFilters = {
        data_type: document.getElementById('filter-type').value,
        symbol: document.getElementById('filter-symbol').value,
        date_from: document.getElementById('filter-date-from').value,
        date_to: document.getElementById('filter-date-to').value
    };
    
    // Remove empty filters
    Object.keys(currentFilters).forEach(key => {
        if (!currentFilters[key]) delete currentFilters[key];
    });
    
    loadUploadHistory(1);
}

function clearFilters() {
    document.getElementById('filter-type').value = '';
    document.getElementById('filter-symbol').value = '';
    document.getElementById('filter-date-from').value = '';
    document.getElementById('filter-date-to').value = '';
    
    currentFilters = {};
    loadUploadHistory(1);
}

async function viewUploadDetails(uploadId) {
    const upload = uploadHistory.find(u => u.id === uploadId);
    if (!upload) return;
    
    const modalBody = document.getElementById('upload-details-content');
    modalBody.innerHTML = `
        <div class="upload-details">
            <h6>Upload Information</h6>
            <table class="table table-sm">
                <tr><td><strong>Upload ID:</strong></td><td><code>${upload.id}</code></td></tr>
                <tr><td><strong>Date/Time:</strong></td><td>${formatDateTime(upload.upload_time)}</td></tr>
                <tr><td><strong>Data Type:</strong></td><td>${upload.data_type}</td></tr>
                <tr><td><strong>Symbol:</strong></td><td>${upload.symbol}</td></tr>
                <tr><td><strong>Source:</strong></td><td>${upload.source}</td></tr>
                <tr><td><strong>File Name:</strong></td><td>${upload.filename || 'N/A'}</td></tr>
                <tr><td><strong>Records Inserted:</strong></td><td>${formatNumber(upload.records_inserted || 0)}</td></tr>
                <tr><td><strong>Status:</strong></td><td>${upload.status}</td></tr>
            </table>
            
            ${upload.metadata ? `
            <h6 class="mt-3">Additional Details</h6>
            <pre class="bg-dark p-2">${JSON.stringify(upload.metadata, null, 2)}</pre>
            ` : ''}
            
            ${upload.errors && upload.errors.length > 0 ? `
            <h6 class="mt-3">Errors</h6>
            <div class="alert alert-danger">
                ${upload.errors.join('<br>')}
            </div>
            ` : ''}
        </div>
    `;
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('uploadDetailsModal'));
    modal.show();
}

function initiateRollback(uploadId) {
    selectedUploadId = uploadId;
    const upload = uploadHistory.find(u => u.id === uploadId);
    
    if (!upload) return;
    
    const rollbackDetails = document.getElementById('rollback-details');
    rollbackDetails.innerHTML = `
        <table class="table table-sm">
            <tr><td><strong>Upload ID:</strong></td><td><code>${upload.id}</code></td></tr>
            <tr><td><strong>Data Type:</strong></td><td>${upload.data_type}</td></tr>
            <tr><td><strong>Symbol:</strong></td><td>${upload.symbol}</td></tr>
            <tr><td><strong>Records:</strong></td><td>${formatNumber(upload.records_inserted || 0)}</td></tr>
            <tr><td><strong>Upload Date:</strong></td><td>${formatDateTime(upload.upload_time)}</td></tr>
        </table>
    `;
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('rollbackModal'));
    modal.show();
}

async function confirmRollback() {
    if (!selectedUploadId) return;
    
    const confirmBtn = document.getElementById('confirm-rollback-btn');
    confirmBtn.disabled = true;
    confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Rolling back...';
    
    try {
        const response = await fetch(`/data/api/rollback/${selectedUploadId}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('rollbackModal'));
            modal.hide();
            
            // Show success message
            alert('Upload rolled back successfully');
            
            // Reload history
            loadUploadHistory(currentPage);
        } else {
            alert(result.error || 'Failed to rollback upload');
        }
    } catch (error) {
        console.error('Rollback error:', error);
        alert('Error rolling back upload');
    } finally {
        confirmBtn.disabled = false;
        confirmBtn.innerHTML = '<i class="fas fa-undo"></i> Rollback Upload';
        selectedUploadId = null;
    }
}

// Helper functions
function formatDateTime(dateStr) {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}