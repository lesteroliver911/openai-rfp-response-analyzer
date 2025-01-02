document.addEventListener('DOMContentLoaded', function() {
    const rfpFileInput = document.getElementById('rfpFile');
    const responseFileInput = document.getElementById('responseFile');
    const processBtn = document.getElementById('processBtn');
    const processingStatus = document.getElementById('processingStatus');
    const generateReportBtn = document.getElementById('generateReportBtn');
    const reportDiv = document.getElementById('report');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send');
    const chatMessages = document.getElementById('chat-messages');

    function updateFileLabel(input) {
        const label = input.nextElementSibling;
        label.textContent = input.files[0] ? input.files[0].name : 'Choose file';
    }

    rfpFileInput.addEventListener('change', () => updateFileLabel(rfpFileInput));
    responseFileInput.addEventListener('change', () => updateFileLabel(responseFileInput));

    processBtn.addEventListener('click', processDocuments);
    generateReportBtn.addEventListener('click', generateReport);
    chatSendBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });

    function processDocuments() {
        if (!rfpFileInput.files[0] || !responseFileInput.files[0]) {
            alert('Please upload both RFP and Response files.');
            return;
        }

        processingStatus.classList.remove('d-none');
        processingStatus.textContent = 'Processing documents...';
        processBtn.disabled = true;

        // Simulating document processing
        setTimeout(() => {
            processingStatus.textContent = 'Documents processed successfully!';
            processingStatus.classList.remove('alert-info');
            processingStatus.classList.add('alert-success');
            processBtn.disabled = false;
        }, 3000);
    }

    function generateReport() {
        reportDiv.innerHTML = '<p>Generating report...</p>';

        // Simulating report generation
        setTimeout(() => {
            reportDiv.innerHTML = `
                <h4>Analysis Report</h4>
                <p>RFP Document: 20 pages</p>
                <p>Response Document: 15 pages</p>
                <p>Key Findings:</p>
                <ul>
                    <li>90% requirements addressed</li>
                    <li>5 potential areas for improvement identified</li>
                    <li>3 unique selling points highlighted</li>
                </ul>
            `;
        }, 2000);
    }

    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (message) {
            addChatMessage('user', message);
            chatInput.value = '';

            // Simulating AI response
            setTimeout(() => {
                const aiResponse = "Thank you for your question. I'm analyzing the documents and will provide an answer shortly.";
                addChatMessage('ai', aiResponse);
            }, 1000);
        }
    }

    function addChatMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', sender === 'user' ? 'user-message' : 'ai-message');
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
