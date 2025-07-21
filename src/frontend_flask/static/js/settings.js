/**
 * Settings & Configuration page functionality
 */

let currentConfig = {};
let hasUnsavedChanges = false;

function initializeSettings() {
    // Set up tab navigation
    setupTabs();
    
    // Load current configuration
    loadCurrentConfig();
    
    // Set up form handlers
    setupFormHandlers();
    
    // Set up maintenance buttons
    setupMaintenanceButtons();
    
    // Set up system health updates
    startSystemHealthUpdates();
    
    // Set up confirmation dialog
    setupConfirmationDialog();
    
    // Track form changes
    trackFormChanges();
}

function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    const panes = document.querySelectorAll('.tab-pane');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Check for unsaved changes
            if (hasUnsavedChanges) {
                showConfirmation(
                    'Unsaved Changes',
                    'You have unsaved changes. Do you want to discard them?',
                    () => {
                        hasUnsavedChanges = false;
                        switchTab(tab, tabs, panes);
                    }
                );
            } else {
                switchTab(tab, tabs, panes);
            }
        });
    });
}

function switchTab(tab, tabs, panes) {
    // Remove active class from all tabs and panes
    tabs.forEach(t => t.classList.remove('active'));
    panes.forEach(p => p.classList.remove('active'));
    
    // Add active class to clicked tab and corresponding pane
    tab.classList.add('active');
    const targetPane = document.querySelector(tab.getAttribute('href'));
    if (targetPane) {
        targetPane.classList.add('active');
    }
}

async function loadCurrentConfig() {
    try {
        const response = await window.apiClient.request('/settings/api/config/current');
        currentConfig = response;
        populateAllForms(response);
    } catch (error) {
        console.error('Failed to load configuration:', error);
        showNotification('Failed to load current configuration', 'error');
    }
}

function populateAllForms(config) {
    // Trading Rules
    if (config.trading_rules) {
        populateForm('trading-rules-form', config.trading_rules);
        updateConfidenceDisplay();
    }
    
    // API Configuration
    if (config.api_config) {
        populateForm('api-config-form', config.api_config);
    }
    
    // Notifications
    if (config.notifications) {
        populateForm('notifications-form', config.notifications);
    }
    
    // System Maintenance
    if (config.system) {
        document.getElementById('active-model').value = config.system.active_model;
        document.getElementById('auto-retrain').checked = config.system.auto_retrain;
        document.getElementById('retrain-interval').value = config.system.retrain_interval;
        document.getElementById('database-size').textContent = config.system.database_size;
        document.getElementById('cache-size').textContent = config.system.cache_size;
        document.getElementById('log-size').textContent = config.system.log_files_size;
    }
    
    // Load backups
    loadBackups();
}

function populateForm(formId, data) {
    const form = document.getElementById(formId);
    if (!form) return;
    
    Object.keys(data).forEach(key => {
        const element = form.elements[key] || document.getElementById(key.replace(/_/g, '-'));
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = data[key];
            } else {
                element.value = data[key];
            }
        }
    });
}

function setupFormHandlers() {
    // Trading Rules Form
    const tradingForm = document.getElementById('trading-rules-form');
    if (tradingForm) {
        tradingForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveConfiguration('trading_rules', getFormData(tradingForm));
        });
    }
    
    // API Configuration Form
    const apiForm = document.getElementById('api-config-form');
    if (apiForm) {
        apiForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveConfiguration('api_config', getFormData(apiForm));
        });
    }
    
    // Notifications Form
    const notificationsForm = document.getElementById('notifications-form');
    if (notificationsForm) {
        notificationsForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveConfiguration('notifications', getFormData(notificationsForm));
        });
    }
    
    // Test Notification Button
    const testNotificationBtn = document.getElementById('test-notification');
    if (testNotificationBtn) {
        testNotificationBtn.addEventListener('click', testNotification);
    }
    
    // Confidence Slider
    const confidenceSlider = document.getElementById('min-signal-confidence');
    if (confidenceSlider) {
        confidenceSlider.addEventListener('input', updateConfidenceDisplay);
    }
}

function getFormData(form) {
    const formData = new FormData(form);
    const data = {};
    
    // Process all form elements
    for (let [key, value] of formData.entries()) {
        // Convert underscores to match backend expectations
        const fieldKey = key.replace(/-/g, '_');
        
        // Handle different input types
        const element = form.elements[key];
        if (element.type === 'checkbox') {
            data[fieldKey] = element.checked;
        } else if (element.type === 'number') {
            data[fieldKey] = parseFloat(value);
        } else {
            data[fieldKey] = value;
        }
    }
    
    // Handle checkboxes that weren't included (unchecked)
    Array.from(form.elements).forEach(element => {
        if (element.type === 'checkbox' && element.name) {
            const fieldKey = element.name.replace(/-/g, '_');
            if (!(fieldKey in data)) {
                data[fieldKey] = false;
            }
        }
    });
    
    return data;
}

async function saveConfiguration(section, settings) {
    try {
        const response = await window.apiClient.request('/settings/api/config/update', {
            method: 'POST',
            body: JSON.stringify({
                section: section,
                settings: settings
            })
        });
        
        if (response.success) {
            showNotification(response.message, 'success');
            hasUnsavedChanges = false;
            
            // Update current config
            currentConfig[section] = { ...currentConfig[section], ...settings };
        }
    } catch (error) {
        console.error('Failed to save configuration:', error);
        showNotification('Failed to save configuration', 'error');
    }
}

async function testNotification() {
    const webhookUrl = document.getElementById('discord-webhook').value;
    
    if (!webhookUrl) {
        showNotification('Please enter a Discord webhook URL', 'warning');
        return;
    }
    
    try {
        const response = await window.apiClient.request('/settings/api/notifications/test', {
            method: 'POST',
            body: JSON.stringify({
                webhook_url: webhookUrl
            })
        });
        
        if (response.success) {
            showNotification(response.message, 'success');
        }
    } catch (error) {
        console.error('Failed to test notification:', error);
        showNotification('Failed to send test notification', 'error');
    }
}

function updateConfidenceDisplay() {
    const slider = document.getElementById('min-signal-confidence');
    const display = document.getElementById('confidence-value');
    if (slider && display) {
        display.textContent = `${slider.value}%`;
    }
}

function setupMaintenanceButtons() {
    // Compact Database
    const compactBtn = document.getElementById('compact-database');
    if (compactBtn) {
        compactBtn.addEventListener('click', () => {
            performMaintenance('compact', 'Compact Database', 'This will optimize the database. Continue?');
        });
    }
    
    // Clear Cache
    const clearCacheBtn = document.getElementById('clear-cache');
    if (clearCacheBtn) {
        clearCacheBtn.addEventListener('click', () => {
            performMaintenance('clear_cache', 'Clear Cache', 'This will clear all cached data. Continue?');
        });
    }
    
    // Clean Logs
    const cleanLogsBtn = document.getElementById('clean-logs');
    if (cleanLogsBtn) {
        cleanLogsBtn.addEventListener('click', () => {
            performMaintenance('clean_logs', 'Clean Logs', 'This will remove old log files. Continue?');
        });
    }
    
    // Retrain Model
    const retrainBtn = document.getElementById('retrain-now');
    if (retrainBtn) {
        retrainBtn.addEventListener('click', retrainModel);
    }
    
    // Backup & Restore
    const createBackupBtn = document.getElementById('create-backup');
    if (createBackupBtn) {
        createBackupBtn.addEventListener('click', createBackup);
    }
    
    const backupSelect = document.getElementById('backup-select');
    if (backupSelect) {
        backupSelect.addEventListener('change', (e) => {
            const restoreBtn = document.getElementById('restore-backup');
            restoreBtn.disabled = !e.target.value;
        });
    }
    
    const restoreBtn = document.getElementById('restore-backup');
    if (restoreBtn) {
        restoreBtn.addEventListener('click', () => {
            const selectedBackup = document.getElementById('backup-select').value;
            if (selectedBackup) {
                showConfirmation(
                    'Restore Backup',
                    'This will restore the system to the selected backup. All current data will be replaced. Continue?',
                    () => restoreBackup(selectedBackup)
                );
            }
        });
    }
}

function performMaintenance(operation, title, message) {
    showConfirmation(title, message, async () => {
        try {
            const response = await window.apiClient.request('/settings/api/maintenance/database', {
                method: 'POST',
                body: JSON.stringify({ operation: operation })
            });
            
            if (response.success) {
                showNotification(response.message, 'success');
                // Reload config to get updated sizes
                loadCurrentConfig();
            }
        } catch (error) {
            console.error('Maintenance operation failed:', error);
            showNotification('Operation failed', 'error');
        }
    });
}

async function retrainModel() {
    const modelType = document.getElementById('active-model').value;
    
    showConfirmation(
        'Retrain Model',
        `This will start retraining the ${modelType} model. This process may take several minutes. Continue?`,
        async () => {
            try {
                // Show loading state
                const btn = document.getElementById('retrain-now');
                const originalText = btn.textContent;
                btn.textContent = 'Retraining...';
                btn.disabled = true;
                
                const response = await window.apiClient.request('/settings/api/model/retrain', {
                    method: 'POST',
                    body: JSON.stringify({ model_type: modelType })
                });
                
                if (response.success) {
                    showNotification(response.message, 'success');
                }
                
                // Reset button
                btn.textContent = originalText;
                btn.disabled = false;
            } catch (error) {
                console.error('Model retraining failed:', error);
                showNotification('Failed to start model retraining', 'error');
                
                // Reset button
                const btn = document.getElementById('retrain-now');
                btn.textContent = 'Retrain Now';
                btn.disabled = false;
            }
        }
    );
}

async function createBackup() {
    try {
        const response = await window.apiClient.request('/settings/api/backup', {
            method: 'POST',
            body: JSON.stringify({ action: 'create' })
        });
        
        if (response.success) {
            showNotification(response.message, 'success');
            loadBackups();
        }
    } catch (error) {
        console.error('Backup creation failed:', error);
        showNotification('Failed to create backup', 'error');
    }
}

async function loadBackups() {
    try {
        const response = await window.apiClient.request('/settings/api/backup');
        const select = document.getElementById('backup-select');
        
        // Clear existing options
        select.innerHTML = '<option value="">Select a backup to restore...</option>';
        
        // Add backup options
        response.backups.forEach(backup => {
            const option = document.createElement('option');
            option.value = backup.id;
            option.textContent = `${backup.date} (${backup.size})`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load backups:', error);
    }
}

async function restoreBackup(backupId) {
    try {
        const response = await window.apiClient.request('/settings/api/backup', {
            method: 'POST',
            body: JSON.stringify({
                action: 'restore',
                backup_id: backupId
            })
        });
        
        if (response.success) {
            showNotification(response.message, 'success');
            // Reload the page after restore
            setTimeout(() => window.location.reload(), 2000);
        }
    } catch (error) {
        console.error('Backup restoration failed:', error);
        showNotification('Failed to restore backup', 'error');
    }
}

function startSystemHealthUpdates() {
    // Initial update
    updateSystemHealth();
    
    // Update every 30 seconds
    setInterval(updateSystemHealth, 30000);
}

async function updateSystemHealth() {
    try {
        const response = await window.apiClient.request('/settings/api/system/status');
        
        // Update health metrics
        updateHealthMetric('cpu-usage', response.cpu_usage, '%');
        updateHealthMetric('memory-usage', response.memory_usage, '%');
        updateHealthMetric('disk-usage', response.disk_usage, '%');
        updateHealthMetric('api-latency', response.api_latency, 'ms');
    } catch (error) {
        console.error('Failed to update system health:', error);
    }
}

function updateHealthMetric(elementId, value, suffix) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = `${value.toFixed(1)}${suffix}`;
        
        // Add color coding based on thresholds
        element.classList.remove('good', 'warning', 'danger');
        if (elementId.includes('usage')) {
            if (value < 50) element.classList.add('good');
            else if (value < 80) element.classList.add('warning');
            else element.classList.add('danger');
        } else if (elementId === 'api-latency') {
            if (value < 100) element.classList.add('good');
            else if (value < 500) element.classList.add('warning');
            else element.classList.add('danger');
        }
    }
}

function trackFormChanges() {
    const forms = document.querySelectorAll('.settings-form');
    forms.forEach(form => {
        form.addEventListener('change', () => {
            hasUnsavedChanges = true;
        });
    });
}

function setupConfirmationDialog() {
    const cancelBtn = document.getElementById('confirm-cancel');
    const dialog = document.getElementById('confirmation-dialog');
    
    if (cancelBtn && dialog) {
        cancelBtn.addEventListener('click', () => {
            dialog.style.display = 'none';
        });
        
        // Close on outside click
        dialog.addEventListener('click', (e) => {
            if (e.target === dialog) {
                dialog.style.display = 'none';
            }
        });
    }
}

function showConfirmation(title, message, onConfirm) {
    const dialog = document.getElementById('confirmation-dialog');
    const titleEl = document.getElementById('confirmation-title');
    const messageEl = document.getElementById('confirmation-message');
    const proceedBtn = document.getElementById('confirm-proceed');
    
    titleEl.textContent = title;
    messageEl.textContent = message;
    
    // Remove old event listener
    const newProceedBtn = proceedBtn.cloneNode(true);
    proceedBtn.parentNode.replaceChild(newProceedBtn, proceedBtn);
    
    // Add new event listener
    newProceedBtn.addEventListener('click', () => {
        dialog.style.display = 'none';
        onConfirm();
    });
    
    dialog.style.display = 'flex';
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}