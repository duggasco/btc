// Data Management JavaScript
let availableData = [];
let selectedData = null;

document.addEventListener('DOMContentLoaded', function() {
    loadAvailableData();
    setupEventListeners();
});

function setupEventListeners() {
    // Delete confirmation text input
    const confirmInput = document.getElementById('delete-confirmation-text');
    confirmInput.addEventListener('input', function() {
        const confirmBtn = document.getElementById('confirm-delete-btn');
        confirmBtn.disabled = this.value.toUpperCase() !== 'DELETE';
    });
    
    // Confirm delete button
    document.getElementById('confirm-delete-btn').addEventListener('click', performDeletion);
    
    // Advanced deletion preview
    document.getElementById('preview-delete-btn').addEventListener('click', previewAdvancedDeletion);
    
    // Advanced deletion button
    document.getElementById('advanced-delete-btn').addEventListener('click', performAdvancedDeletion);
    
    // Reset confirmation when modal closes
    const deleteModal = document.getElementById('deleteModal');
    deleteModal.addEventListener('hidden.bs.modal', function() {
        document.getElementById('delete-confirmation-text').value = '';
        document.getElementById('confirm-delete-btn').disabled = true;
        selectedData = null;
    });
}

async function loadAvailableData() {
    try {
        const response = await fetch('/data/api/available');
        const result = await response.json();
        
        if (result.success && result.data) {
            availableData = result.data;
            displayAvailableData(result.data);
        } else {
            showMessage('Failed to load available data', 'error');
        }
    } catch (error) {
        console.error('Error loading data:', error);
        showMessage('Error loading available data', 'error');
    } finally {
        document.getElementById('loading-spinner').style.display = 'none';
        document.getElementById('data-table-container').style.display = 'block';
    }
}

function displayAvailableData(data) {
    const tbody = document.getElementById('data-table-body');
    tbody.innerHTML = '';
    
    if (data.length === 0) {
        document.getElementById('available-data-table').style.display = 'none';
        document.getElementById('no-data-message').style.display = 'block';
        return;
    }
    
    document.getElementById('available-data-table').style.display = 'table';
    document.getElementById('no-data-message').style.display = 'none';
    
    data.forEach((item, index) => {
        const row = document.createElement('tr');
        
        // Format dates
        const startDate = item.start_date ? new Date(item.start_date).toLocaleDateString() : 'N/A';
        const endDate = item.end_date ? new Date(item.end_date).toLocaleDateString() : 'N/A';
        
        row.innerHTML = `
            <td><span class="badge bg-info">${item.data_type.toUpperCase()}</span></td>
            <td>${item.symbol}</td>
            <td>${item.source}</td>
            <td>${startDate} - ${endDate}</td>
            <td>${item.record_count.toLocaleString()}</td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="viewStats(${index})">
                    <i class="fas fa-chart-bar"></i> Stats
                </button>
                <button class="btn btn-sm btn-danger" onclick="confirmDelete(${index})">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

async function viewStats(index) {
    const data = availableData[index];
    const modal = new bootstrap.Modal(document.getElementById('statsModal'));
    modal.show();
    
    document.getElementById('stats-loading').style.display = 'block';
    document.getElementById('stats-content').style.display = 'none';
    
    try {
        const response = await fetch(`/data/api/stats/${data.data_type}?source=${data.source}&symbol=${data.symbol}`);
        const result = await response.json();
        
        if (result.success && result.stats) {
            displayStats(result.stats);
        } else {
            document.getElementById('stats-content').innerHTML = '<p class="text-danger">Failed to load statistics</p>';
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('stats-content').innerHTML = '<p class="text-danger">Error loading statistics</p>';
    } finally {
        document.getElementById('stats-loading').style.display = 'none';
        document.getElementById('stats-content').style.display = 'block';
    }
}

function displayStats(stats) {
    let html = `
        <div class="row">
            <div class="col-md-6">
                <h6>General Information</h6>
                <table class="table table-sm">
                    <tr><th>Data Type:</th><td>${stats.data_type.toUpperCase()}</td></tr>
                    <tr><th>Source:</th><td>${stats.source}</td></tr>
                    <tr><th>Symbol:</th><td>${stats.symbol}</td></tr>
                    <tr><th>Total Records:</th><td>${stats.total_records.toLocaleString()}</td></tr>
                    <tr><th>Date Range:</th><td>${new Date(stats.earliest_date).toLocaleDateString()} - ${new Date(stats.latest_date).toLocaleDateString()}</td></tr>
                </table>
            </div>
    `;
    
    // Add OHLCV-specific stats if available
    if (stats.avg_price !== undefined) {
        html += `
            <div class="col-md-6">
                <h6>Price Statistics</h6>
                <table class="table table-sm">
                    <tr><th>Average Price:</th><td>$${stats.avg_price.toFixed(2)}</td></tr>
                    <tr><th>Min Price:</th><td>$${stats.min_price.toFixed(2)}</td></tr>
                    <tr><th>Max Price:</th><td>$${stats.max_price.toFixed(2)}</td></tr>
                    <tr><th>Total Volume:</th><td>${stats.total_volume.toLocaleString()}</td></tr>
                </table>
            </div>
        `;
    }
    
    html += '</div>';
    document.getElementById('stats-content').innerHTML = html;
}

function confirmDelete(index) {
    selectedData = availableData[index];
    
    const details = `
        <table class="table table-sm">
            <tr><th>Data Type:</th><td>${selectedData.data_type.toUpperCase()}</td></tr>
            <tr><th>Source:</th><td>${selectedData.source}</td></tr>
            <tr><th>Symbol:</th><td>${selectedData.symbol}</td></tr>
            <tr><th>Date Range:</th><td>${new Date(selectedData.start_date).toLocaleDateString()} - ${new Date(selectedData.end_date).toLocaleDateString()}</td></tr>
            <tr><th>Records to Delete:</th><td class="text-danger"><strong>${selectedData.record_count.toLocaleString()}</strong></td></tr>
        </table>
    `;
    
    document.getElementById('deletion-details').innerHTML = details;
    
    const modal = new bootstrap.Modal(document.getElementById('deleteModal'));
    modal.show();
}

async function performDeletion() {
    if (!selectedData) return;
    
    const confirmBtn = document.getElementById('confirm-delete-btn');
    confirmBtn.disabled = true;
    confirmBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Deleting...';
    
    try {
        const params = new URLSearchParams({
            data_type: selectedData.data_type,
            source: selectedData.source,
            symbol: selectedData.symbol,
            confirm: 'true'
        });
        
        const response = await fetch(`/data/api/delete?${params}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showMessage(`Successfully deleted ${result.deleted_count} records`, 'success');
            bootstrap.Modal.getInstance(document.getElementById('deleteModal')).hide();
            loadAvailableData(); // Reload the data
        } else {
            showMessage(result.error || 'Deletion failed', 'error');
        }
    } catch (error) {
        console.error('Error deleting data:', error);
        showMessage('Error deleting data', 'error');
    } finally {
        confirmBtn.innerHTML = '<i class="fas fa-trash"></i> Delete Data';
        confirmBtn.disabled = false;
    }
}

async function previewAdvancedDeletion() {
    const dataType = document.getElementById('adv-data-type').value;
    const source = document.getElementById('adv-source').value;
    const symbol = document.getElementById('adv-symbol').value;
    
    if (!dataType || !source) {
        showMessage('Please select data type and source', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/data/api/stats/${dataType}?source=${source}&symbol=${symbol}`);
        const result = await response.json();
        
        if (result.success && result.stats) {
            const startDate = document.getElementById('adv-start-date').value;
            const endDate = document.getElementById('adv-end-date').value;
            
            let message = `Preview: This will delete ${result.stats.total_records.toLocaleString()} records`;
            if (startDate || endDate) {
                message += ' (Note: Date filtering will be applied during deletion)';
            }
            
            showMessage(message, 'info');
            document.getElementById('advanced-delete-btn').disabled = false;
        } else {
            showMessage('No data found matching the criteria', 'warning');
            document.getElementById('advanced-delete-btn').disabled = true;
        }
    } catch (error) {
        console.error('Error previewing deletion:', error);
        showMessage('Error previewing deletion', 'error');
    }
}

async function performAdvancedDeletion() {
    const dataType = document.getElementById('adv-data-type').value;
    const source = document.getElementById('adv-source').value;
    const symbol = document.getElementById('adv-symbol').value;
    const startDate = document.getElementById('adv-start-date').value;
    const endDate = document.getElementById('adv-end-date').value;
    
    if (!confirm('Are you sure you want to delete this data? This action cannot be undone.')) {
        return;
    }
    
    const advDeleteBtn = document.getElementById('advanced-delete-btn');
    advDeleteBtn.disabled = true;
    advDeleteBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Deleting...';
    
    try {
        const params = new URLSearchParams({
            data_type: dataType,
            source: source,
            symbol: symbol,
            confirm: 'true'
        });
        
        if (startDate) params.append('start_date', startDate + 'T00:00:00');
        if (endDate) params.append('end_date', endDate + 'T23:59:59');
        
        const response = await fetch(`/data/api/delete?${params}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showMessage(`Successfully deleted ${result.deleted_count} records`, 'success');
            // Reset form
            document.getElementById('advanced-delete-form').reset();
            document.getElementById('advanced-delete-btn').disabled = true;
            loadAvailableData(); // Reload the data
        } else {
            showMessage(result.error || 'Deletion failed', 'error');
        }
    } catch (error) {
        console.error('Error deleting data:', error);
        showMessage('Error deleting data', 'error');
    } finally {
        advDeleteBtn.innerHTML = '<i class="fas fa-trash"></i> Delete with Criteria';
        advDeleteBtn.disabled = false;
    }
}

function showMessage(message, type) {
    const messagesDiv = document.getElementById('messages');
    const alertClass = type === 'error' ? 'alert-danger' : 
                      type === 'success' ? 'alert-success' : 
                      type === 'warning' ? 'alert-warning' :
                      'alert-info';
    
    const alert = document.createElement('div');
    alert.className = `alert ${alertClass} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    messagesDiv.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alert.remove(), 150);
    }, 5000);
}