// Main JavaScript for Fake News Detection System - FIXED VERSION

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Wait for DOM to fully load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing event listeners...');
    
    // Get DOM elements
    const form = document.getElementById('predictionForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const sampleBtn = document.getElementById('sampleBtn');
    const newsText = document.getElementById('newsText');
    
    // Check if elements exist
    if (!form) console.error('Form not found!');
    if (!analyzeBtn) console.error('Analyze button not found!');
    if (!clearBtn) console.error('Clear button not found!');
    if (!sampleBtn) console.error('Sample button not found!');
    
    // Add event listeners
    if (form) {
        form.addEventListener('submit', handleSubmit);
        console.log('Form submit handler attached');
    }
    
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Analyze button clicked');
            handleSubmit(e);
        });
        console.log('Analyze button handler attached');
    }
    
    if (clearBtn) {
        clearBtn.addEventListener('click', handleClear);
        console.log('Clear button handler attached');
    }
    
    if (sampleBtn) {
        sampleBtn.addEventListener('click', handleSample);
        console.log('Sample button handler attached');
    }
    
    // Check model status on load
    checkModelStatus();
});

// Handle Form Submission
async function handleSubmit(e) {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('handleSubmit called');
    
    const newsText = document.getElementById('newsText');
    const text = newsText ? newsText.value.trim() : '';
    
    console.log('Text length:', text.length);
    
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    if (text.length < 10) {
        showError('Please enter at least 10 characters for accurate analysis.');
        return;
    }
    
    await analyzeText(text);
}

// Analyze Text - Send to Flask Backend
async function analyzeText(text) {
    console.log('analyzeText called with:', text.substring(0, 50) + '...');
    
    showLoading(true);
    hideResults();
    hideError();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        console.log('Response status:', response.status);
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.success) {
            displayResults(data.prediction);
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Network error: ' + error.message + '. Make sure the Flask server is running on port 5000.');
    } finally {
        showLoading(false);
    }
}

// Check Model Status
async function checkModelStatus() {
    const statusBadge = document.getElementById('modelStatus');
    if (!statusBadge) return;
    
    statusBadge.innerHTML = '<i class="fas fa-circle me-1" style="font-size: 8px;"></i> Checking...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        console.log('Health check response:', data);
        
        if (data.model_loaded) {
            statusBadge.className = 'badge bg-success model-status';
            statusBadge.innerHTML = '<i class="fas fa-check-circle me-1"></i> Model Ready';
        } else {
            statusBadge.className = 'badge bg-warning model-status';
            statusBadge.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i> Model Loading...';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        statusBadge.className = 'badge bg-danger model-status';
        statusBadge.innerHTML = '<i class="fas fa-times-circle me-1"></i> Server Offline';
        showError('Cannot connect to server at ' + API_BASE_URL + '. Make sure Flask is running.');
    }
}

// Display Results
function displayResults(prediction) {
    console.log('Displaying results:', prediction);
    
    const isFake = prediction.label === 'FAKE';
    const badgeColor = isFake ? 'danger' : 'success';
    const icon = isFake ? 'fa-exclamation-triangle' : 'fa-check-circle';
    
    // Prediction Badge
    const predictionBadge = document.getElementById('predictionBadge');
    if (predictionBadge) {
        predictionBadge.innerHTML = `
            <div class="prediction-badge badge-${isFake ? 'fake' : 'real'} animate-pulse">
                <i class="fas ${icon} me-2"></i>
                ${prediction.label} NEWS
                ${prediction.is_high_confidence ? '<i class="fas fa-star ms-2"></i>' : ''}
            </div>
        `;
    }
    
    // Confidence Bar
    const confidencePercent = (prediction.confidence * 100).toFixed(1);
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    
    if (confidenceBar) {
        confidenceBar.style.width = `${confidencePercent}%`;
    }
    if (confidenceText) {
        confidenceText.textContent = `${confidencePercent}%`;
    }
    
    // Probabilities
    const fakeProbElem = document.getElementById('fakeProb');
    const realProbElem = document.getElementById('realProb');
    
    if (fakeProbElem) {
        fakeProbElem.textContent = `${(prediction.fake_probability * 100).toFixed(1)}%`;
    }
    if (realProbElem) {
        realProbElem.textContent = `${(prediction.real_probability * 100).toFixed(1)}%`;
    }
    
    // Detailed Analysis
    const detailedAnalysis = document.getElementById('detailedAnalysis');
    if (detailedAnalysis) {
        detailedAnalysis.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <p><strong><i class="fas fa-chart-line me-1"></i> Confidence:</strong> 
                        <span class="badge bg-primary">${(prediction.confidence * 100).toFixed(1)}%</span>
                    </p>
                    <p><strong><i class="fas fa-random me-1"></i> Method:</strong> 
                        <span class="badge bg-info">${prediction.method || 'ML'}</span>
                    </p>
                </div>
                <div class="col-md-6">
                    <p><strong><i class="fas fa-chart-simple me-1"></i> Risk Level:</strong> 
                        <span class="badge bg-${prediction.risk_color}">${prediction.risk_level}</span>
                    </p>
                    <p><strong><i class="fas fa-clock me-1"></i> Analyzed:</strong> 
                        <span class="badge bg-secondary">${new Date(prediction.timestamp).toLocaleTimeString()}</span>
                    </p>
                </div>
            </div>
        `;
    }
    
    // Analyzed Text
    const analyzedTextElem = document.getElementById('analyzedText');
    if (analyzedTextElem) {
        analyzedTextElem.innerHTML = `
            <i class="fas fa-quote-left text-muted me-1"></i>
            ${escapeHtml(prediction.text)}
        `;
    }
    
    // Show results
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Handle Clear
function handleClear(e) {
    if (e) e.preventDefault();
    const newsText = document.getElementById('newsText');
    if (newsText) {
        newsText.value = '';
        newsText.focus();
    }
    hideResults();
    hideError();
}

// Handle Sample
function handleSample(e) {
    if (e) e.preventDefault();
    const newsText = document.getElementById('newsText');
    if (newsText) {
        newsText.value = "SHOCKING: Government hiding alien evidence from public! An anonymous Area 51 whistleblower has leaked classified documents proving that the government has recovered multiple extraterrestrial spacecraft. NASA officials have been lying about this for over 50 years! Why won't mainstream media report this? The truth about alien contact is finally being exposed! Wake up America!";
        newsText.focus();
    }
    hideError();
}

// Show/Hide Functions
function showLoading(show) {
    const loadingSpinner = document.getElementById('loadingSpinner');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (loadingSpinner) {
        loadingSpinner.style.display = show ? 'block' : 'none';
    }
    if (analyzeBtn) {
        analyzeBtn.disabled = show;
        analyzeBtn.innerHTML = show ? '<i class="fas fa-spinner fa-spin me-2"></i> Analyzing...' : '<i class="fas fa-search me-2"></i> Analyze News';
    }
}

function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
}

function showError(message) {
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    
    if (errorMessage) {
        errorMessage.textContent = message;
    }
    if (errorAlert) {
        errorAlert.style.display = 'block';
        setTimeout(() => {
            errorAlert.style.display = 'none';
        }, 5000);
    }
}

function hideError() {
    const errorAlert = document.getElementById('errorAlert');
    if (errorAlert) {
        errorAlert.style.display = 'none';
    }
}

// Utility: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}