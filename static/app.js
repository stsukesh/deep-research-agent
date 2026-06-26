/**
 * Enterprise Research Agent — Frontend Logic
 * ===========================================
 * WHAT: Handles user interactions, API calls, polling, and UI updates.
 * HOW:  Vanilla JavaScript with fetch() for API calls and DOM manipulation.
 * WHY:  No framework needed for a demo UI — keeps it lightweight and portable.
 *
 * FLOW:
 *   1. User types query → clicks "Start Research"
 *   2. POST /api/research → get job_id
 *   3. Poll GET /api/status/{job_id} every 3 seconds
 *   4. When status = "awaiting_approval" → show findings for review
 *   5. User clicks Approve/Reject → POST /api/approve/{job_id}
 *   6. Poll again until status = "completed"
 *   7. GET /api/report/{job_id} → render the markdown report
 *   8. GET /api/metrics/{job_id} → show evaluation metrics
 */

// ===== STATE =====
let currentJobId = null;
let pollInterval = null;
let pollStartTime = null;
let rawReportMarkdown = null;

// ===== API BASE =====
const API_BASE = '/api';

// ===== HELPER: Set Query from Hint Chips =====
function setQuery(query) {
    document.getElementById('query-input').value = query;
    document.getElementById('query-input').focus();
}

// ===== START RESEARCH =====
async function startResearch() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();

    if (!query) {
        input.focus();
        return;
    }

    // Disable button
    const btn = document.getElementById('submit-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Starting...';

    try {
        const response = await fetch(`${API_BASE}/research`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        currentJobId = data.job_id;
        localStorage.setItem('current_research_job_id', currentJobId);
        pollStartTime = Date.now();

        // Show status card
        showSection('pipeline-section');
        updateAgentStatus('Planning Research', 'The Planner Agent is creating a structured research plan for your query...', 'loading');

        // Start polling
        startPolling();

    } catch (error) {
        console.error('Error starting research:', error);
        alert('Failed to start research. Make sure the server is running.');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/>
            </svg>
            Start Research
        `;
    }
}

// ===== POLLING =====
function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        if (!currentJobId) return;

        try {
            const response = await fetch(`${API_BASE}/status/${currentJobId}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const errMsg = errorData.detail || `Server returned ${response.status}`;
                updateAgentStatus(
                    'Connection Lost / Job Not Found',
                    `The server returned an error: ${errMsg}. Please start a new research session.`,
                    'error'
                );
                stopPolling();
                return;
            }
            const data = await response.json();

            handleStatusUpdate(data);

        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 3000); // Poll every 3 seconds
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// ===== STATUS HANDLER =====
function handleStatusUpdate(data) {
    const status = data.status;

    switch (status) {
        case 'pending':
            updateAgentStatus(
                'Planning Research',
                'The Planner Agent is creating a structured research plan for your query...',
                'loading'
            );
            break;

        case 'researching':
            // Determine which sub-step based on elapsed time
            const elapsed = (Date.now() - pollStartTime) / 1000;
            if (elapsed < 10) {
                updateAgentStatus(
                    'Searching Knowledge Sources',
                    'The Researcher Agent is querying Tavily, DuckDuckGo, Wikipedia, and Arxiv...',
                    'loading'
                );
            } else if (elapsed < 25) {
                updateAgentStatus(
                    'Extracting Facts & Findings',
                    'The Extractor Agent is cleaning raw search results and verifying sources...',
                    'loading'
                );
            } else {
                updateAgentStatus(
                    'Processing Results',
                    'Analyzing findings and preparing data...',
                    'loading'
                );
            }
            break;

        case 'awaiting_approval':
            updateAgentStatus(
                'Awaiting Approval',
                'Please review the extracted findings below to proceed.',
                'paused'
            );
            // Show approval UI
            showApprovalSection(data.review_data);
            stopPolling();
            break;

        case 'writing':
            updateAgentStatus(
                'Writing Report',
                'The Writer Agent is drafting a professional report in markdown...',
                'loading'
            );
            break;

        case 'completed':
            updateAgentStatus(
                'Research Completed',
                'Final report and performance metrics are ready.',
                'success'
            );
            stopPolling();
            loadReport();
            loadMetrics();
            break;

        case 'failed':
            updateAgentStatus(
                'Research Failed',
                'An error occurred during execution: ' + (data.error || 'Unknown error'),
                'error'
            );
            stopPolling();
            alert('Research failed: ' + (data.error || 'Unknown error'));
            break;
    }
}

// ===== AGENT STATUS UI =====
function updateAgentStatus(currentAction, subAction, mode) {
    const actionEl = document.getElementById('agent-current-action');
    const subActionEl = document.getElementById('agent-sub-action');
    const spinner = document.getElementById('status-spinner');
    const iconContainer = document.getElementById('status-icon');

    if (actionEl) actionEl.textContent = currentAction;
    if (subActionEl) subActionEl.textContent = subAction;

    if (mode === 'loading' || mode === 'pending') {
        if (spinner) spinner.classList.remove('hidden');
        if (iconContainer) iconContainer.classList.add('hidden');
    } else {
        if (spinner) spinner.classList.add('hidden');
        if (iconContainer) iconContainer.classList.remove('hidden');
        if (mode === 'success') {
            if (iconContainer) iconContainer.textContent = '✅';
        } else if (mode === 'error') {
            if (iconContainer) iconContainer.textContent = '❌';
        } else if (mode === 'paused') {
            if (iconContainer) iconContainer.textContent = '👤';
        }
    }
}

// ===== APPROVAL UI =====
function showApprovalSection(reviewData) {
    showSection('approval-section');

    // Stats
    const statsEl = document.getElementById('approval-stats');
    if (reviewData) {
        statsEl.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${reviewData.total_findings || '—'}</div>
                <div class="stat-label">Total Findings</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(reviewData.average_confidence * 100).toFixed(0) || '—'}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(reviewData.topics_covered || []).length}</div>
                <div class="stat-label">Topics Covered</div>
            </div>
        `;

        // Findings
        const findingsEl = document.getElementById('approval-findings');
        const findings = reviewData.top_findings || [];
        findingsEl.innerHTML = findings.map(f => {
            // Parse confidence from the finding string
            const match = f.match(/\[(\d+)%\]/);
            const conf = match ? parseInt(match[1]) : 50;
            const confClass = conf >= 80 ? 'confidence-high' : conf >= 50 ? 'confidence-mid' : 'confidence-low';

            return `<div class="finding-item">
                <span class="confidence-badge ${confClass}">${conf}%</span>
                ${f.replace(/\[\d+%\]\s*/, '')}
            </div>`;
        }).join('');
    }
}

// ===== APPROVE/REJECT =====
async function approveResearch(approved) {
    if (!currentJobId) return;

    const approveBtn = document.querySelector('.btn-approve');
    const rejectBtn = document.querySelector('.btn-reject');

    approveBtn.disabled = true;
    rejectBtn.disabled = true;

    let feedback = "";
    if (!approved) {
        const feedbackEl = document.getElementById('rejection-feedback');
        feedback = feedbackEl ? feedbackEl.value.trim() : "";
        if (!feedback) {
            alert('Please provide feedback explaining what you specifically want the researcher to focus on next.');
            approveBtn.disabled = false;
            rejectBtn.disabled = false;
            if (feedbackEl) {
                feedbackEl.focus();
                feedbackEl.style.borderColor = 'var(--danger)';
            }
            return;
        }
    }

    if (approved) {
        approveBtn.innerHTML = '<span class="spinner"></span> Generating Report...';
    }

    try {
        const response = await fetch(`${API_BASE}/approve/${currentJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ approved, feedback }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server returned ${response.status}`);
        }

        const data = await response.json();

        if (approved) {
            // Hide approval, update status
            hideSection('approval-section');
            updateAgentStatus('Writing Report', 'The Writer Agent is drafting a professional report in markdown...', 'loading');

            // Check if report is already done (might complete quickly)
            if (data.status === 'completed') {
                updateAgentStatus('Research Completed', 'Final report and performance metrics are ready.', 'success');
                loadReport();
                loadMetrics();
            } else {
                // Poll for completion
                startPolling();
            }
        } else {
            // Rejected — hide approval, start polling for re-research
            hideSection('approval-section');
            
            // Clear textarea for next time
            const feedbackEl = document.getElementById('rejection-feedback');
            if (feedbackEl) {
                feedbackEl.value = '';
                feedbackEl.style.borderColor = '';
            }
            
            pollStartTime = Date.now(); // reset progress timer
            updateAgentStatus('Searching Knowledge Sources', 'The Researcher Agent is querying Tavily, DuckDuckGo, Wikipedia, and Arxiv...', 'loading');
            
            // Start polling
            startPolling();
        }

    } catch (error) {
        console.error('Approval error:', error);
        alert('Failed to submit approval: ' + error.message + '. Please make sure the server is running and try again.');
        approveBtn.disabled = false;
        rejectBtn.disabled = false;
        if (approved) {
            approveBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
                Approve & Generate Report
            `;
        }
    }
}

// ===== LOAD REPORT =====
async function loadReport() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`${API_BASE}/report/${currentJobId}`);
        if (!response.ok) return;

        const data = await response.json();

        // Store raw markdown for copy/download
        rawReportMarkdown = data.report;

        // Render markdown as HTML (basic conversion)
        const reportHtml = markdownToHtml(data.report);

        document.getElementById('report-content').innerHTML = reportHtml;

        // Meta tags
        document.getElementById('report-meta').innerHTML = `
            <span class="meta-tag">📊 Confidence: ${(data.confidence_score * 100).toFixed(0)}%</span>
            <span class="meta-tag">🔄 Revisions: ${data.revision_count}</span>
            <span class="meta-tag">📝 Findings: ${data.findings_count}</span>
        `;

        showSection('report-section');

    } catch (error) {
        console.error('Error loading report:', error);
    }
}

// ===== LOAD METRICS =====
async function loadMetrics() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`${API_BASE}/metrics/${currentJobId}`);
        if (!response.ok) return;

        const data = await response.json();
        const m = data.metrics;

        document.getElementById('metrics-grid').innerHTML = `
            <div class="metric-card">
                <div class="metric-value">${m.research_time_seconds}s</div>
                <div class="metric-label">Research Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.tool_calls_count}</div>
                <div class="metric-label">Tool Calls</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.citations_count}</div>
                <div class="metric-label">Citations</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(m.confidence_average * 100).toFixed(0)}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.revision_count}</div>
                <div class="metric-label">Revisions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.findings_count}</div>
                <div class="metric-label">Findings</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${m.topics_researched}</div>
                <div class="metric-label">Topics</div>
            </div>
        `;

        showSection('metrics-section');

    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

// ===== MARKDOWN TO HTML (Basic) =====
function markdownToHtml(md) {
    if (!md) return '<p>No report content.</p>';

    let html = md
        // Headers
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Unordered lists
        .replace(/^[-*] (.*$)/gm, '<li>$1</li>')
        // Numbered lists
        .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
        // Paragraphs (double newlines)
        .replace(/\n\n/g, '</p><p>')
        // Single newlines in context
        .replace(/\n/g, '<br>');

    // Wrap consecutive <li> in <ul>
    html = html.replace(/(<li>.*?<\/li>)(?:<br>)?/g, '$1');
    html = html.replace(/((?:<li>.*?<\/li>)+)/g, '<ul>$1</ul>');

    return `<p>${html}</p>`;
}

// ===== SECTION VISIBILITY =====
function showSection(id) {
    document.getElementById(id).classList.remove('hidden');
    // Smooth scroll to the section
    document.getElementById(id).scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideSection(id) {
    document.getElementById(id).classList.add('hidden');
}

// ===== COPY REPORT =====
async function copyReport() {
    if (!rawReportMarkdown) return;

    const btn = document.getElementById('btn-copy-report');
    try {
        await navigator.clipboard.writeText(rawReportMarkdown);
        btn.classList.add('copied');
        btn.querySelector('span').textContent = 'Copied!';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.querySelector('span').textContent = 'Copy';
        }, 2000);
    } catch (err) {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = rawReportMarkdown;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        btn.classList.add('copied');
        btn.querySelector('span').textContent = 'Copied!';
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.querySelector('span').textContent = 'Copy';
        }, 2000);
    }
}

// ===== DOWNLOAD PDF =====
function downloadPdf() {
    const reportEl = document.getElementById('report-content');
    if (!reportEl) return;

    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Research Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            color: #1a1a2e;
            background: #fff;
            padding: 48px 56px;
            line-height: 1.75;
            font-size: 14px;
        }
        h1 {
            font-size: 24px; font-weight: 800;
            color: #6c5ce7;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 3px solid #6c5ce7;
        }
        h2 {
            font-size: 18px; font-weight: 700;
            color: #2d3436;
            margin-top: 28px; margin-bottom: 10px;
        }
        h3 {
            font-size: 15px; font-weight: 600;
            margin-top: 20px; margin-bottom: 8px;
        }
        p { margin-bottom: 12px; color: #2d3436; }
        ul, ol { margin-bottom: 12px; padding-left: 24px; }
        li { margin-bottom: 4px; color: #2d3436; }
        strong { color: #1a1a2e; font-weight: 600; }
        .footer {
            margin-top: 40px;
            padding-top: 16px;
            border-top: 1px solid #ddd;
            font-size: 11px;
            color: #999;
            text-align: center;
        }
        @media print {
            body { padding: 24px 32px; }
            @page { margin: 1cm; }
        }
    </style>
</head>
<body>
    ${reportEl.innerHTML}
    <div class="footer">Generated by Enterprise Research Agent &bull; ${new Date().toLocaleDateString()}</div>
    <script>window.onload = function() { window.print(); }<\/script>
</body>
</html>`);
    printWindow.document.close();
}

// ===== KEYBOARD SHORTCUT =====
document.addEventListener('DOMContentLoaded', async () => {
    const input = document.getElementById('query-input');

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            startResearch();
        }
    });

    // Restore previous session if any
    const savedJobId = localStorage.getItem('current_research_job_id');
    if (savedJobId) {
        currentJobId = savedJobId;
        pollStartTime = Date.now();
        try {
            const response = await fetch(`${API_BASE}/status/${currentJobId}`);
            if (response.ok) {
                const data = await response.json();
                showSection('pipeline-section');
                handleStatusUpdate(data);
            } else {
                localStorage.removeItem('current_research_job_id');
                currentJobId = null;
            }
        } catch (error) {
            console.error('Error restoring session:', error);
        }
    }
});
