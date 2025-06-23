// Initialize chat context from localStorage for persistence
let chatContext = JSON.parse(localStorage.getItem('chatContext') || '{}');

document.addEventListener('DOMContentLoaded', function() {

    // Use the inner chat-history container so the initial welcome message
    // remains intact and new messages append correctly
    const chatMessages = document.querySelector('#chat-messages .chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-button');
    const logColumn = document.getElementById('log-column');

    
    console.log('Starting with context:', chatContext);
    
    // Initialize Socket.IO
    const socket = io();
    
    // Clear chat messages on page load to avoid duplication
    while (chatMessages.firstChild) {
        if (chatMessages.firstChild.classList && 
            chatMessages.firstChild.classList.contains('message') &&
            chatMessages.firstChild.classList.contains('bot') &&
            chatMessages.firstChild.querySelector('.message-content').textContent.includes('Welcome to Authematic')) {
            // Keep only the first welcome message
            break;
        }
        chatMessages.removeChild(chatMessages.firstChild);
    }
    
    // Handle connection
    socket.on('connect', function() {
        console.log('Connected to server');
        clearChatContext(false);               // ensure fresh state for new sessions without bot message
    });
    
    // Handle disconnection
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        appendMessage('bot', 'Connection lost. Please refresh the page.');
    });
    
    // Handle received messages
    socket.on('receive_message', function(data) {
        appendMessage(data.sender, data.message);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });

    // Listener for the new progress log

    const progressLogContainer = document.getElementById('progress-log-container');
    const progressLog = document.getElementById('progress-log');
    // The surrounding card body becomes scrollable when the log grows, so keep
    // a reference to it for auto-scrolling
    const logCardBody = document.querySelector('#log-column .card-body');
    let logShown = false;

    socket.on('progress_update', function(data) {
        // Progress updates now appear in the main chat instead of the log
        appendMessage('bot', data.message);
    });

    // Listen for server log lines (stdout from the backend)
    socket.on('server_log', function(data) {
        if (!logShown && logColumn) {
            logColumn.classList.remove('d-none');
            document.getElementById('main-column').className = 'col-md-9';
            logShown = true;
        }
        if (progressLogContainer.style.display === 'none') {
            progressLogContainer.style.display = 'block';
        }

        const p = document.createElement('p');
        p.className = 'mb-1 font-monospace';
        p.textContent = data.message;
        progressLog.appendChild(p);
        // Ensure both the log itself and its scrollable container stay
        // scrolled to the bottom so the latest output is visible without
        // manual scrolling
        progressLog.scrollTop = progressLog.scrollHeight;
        if (logCardBody) {
            logCardBody.scrollTop = logCardBody.scrollHeight;
        }
    });

    
    // Handle context updates
    socket.on('set_context', function(data) {
        // Merge new context data with existing context
        chatContext = {...chatContext, ...data};
        console.log('Context updated:', chatContext);
        
        // Store in localStorage for persistence
        localStorage.setItem('chatContext', JSON.stringify(chatContext));
    });
    
    // Send message on button click
    sendButton.addEventListener('click', function() {
        sendMessage();
    });
    
    // Reset on button click
    if (resetButton) {
        resetButton.addEventListener('click', function() {
            clearChatContext();
            socket.emit('send_message', {
                message: 'reset',
                context: chatContext
            });
            showTypingIndicator(chatMessages);
            userInput.value = '';
            userInput.focus();
        });
    }
    
    // Send message on Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Function to send message
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Reset logic goes through the original handler
        if (message.toLowerCase() === 'reset' || message.toLowerCase() === 'restart') {
            clearChatContext();
            socket.emit('send_message', {
                message: message,
                context: chatContext
            });
            userInput.value = '';
            showTypingIndicator(chatMessages);
            return;
        }

        const activePane = document.querySelector('#results-tab-content .tab-pane.active');
        const usingTabs = activePane && chatContext.search_complete;

        if (usingTabs) {
            let targetHistory;
            let context = {};

            if (activePane.id === 'pane-general') {

                targetHistory = activePane.querySelector('.chat-history');

                context = { type: 'general' };
            } else {
                targetHistory = activePane.querySelector('.chat-history');
                context = { type: 'specific', doi: activePane.dataset.doi };
            }

            appendMessage('user', message, targetHistory);

            socket.emit('sidebar_chat_message', {
                query: message,
                context: context
            });

            showTypingIndicator(targetHistory);
        } else {
            appendMessage('user', message, chatMessages);
            socket.emit('send_message', {
                message: message,
                context: chatContext
            });
            showTypingIndicator(chatMessages);
        }

        userInput.value = '';
    }

    // Function to append message to a chat container
    function appendMessage(sender, message, container = chatMessages) {
        const typingIndicator = container.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Support for simple markdown-like formatting
        message = message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
            
        contentDiv.innerHTML = message;
        messageDiv.appendChild(contentDiv);
        
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }

    // Function to show typing indicator in a specific container
    function showTypingIndicator(container = chatMessages) {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        container.appendChild(indicator);
        container.scrollTop = container.scrollHeight;
    }
    
    // Function to clear chat context
    function clearChatContext(showMessage = true) {
        chatContext = {};
        localStorage.removeItem('chatContext');
        console.log('Context cleared');

        // Reset UI layout (hide tabs and event log, restore main column)
        const tabsCol = document.getElementById('tabs-column');
        const tabsContainer = document.getElementById('results-tabs');
        const contentContainer = document.getElementById('results-tab-content');
        const generalPane = document.getElementById('pane-general');

        if (tabsCol) {
            tabsCol.classList.add('d-none');
        }
        if (tabsContainer) {
            tabsContainer.innerHTML = '';
        }
        if (contentContainer && generalPane) {
            contentContainer.innerHTML = '';
            contentContainer.appendChild(generalPane);
            generalPane.classList.add('show', 'active');
        }

        if (logColumn) {
            logColumn.classList.add('d-none');
            progressLogContainer.style.display = 'none';
            logShown = false;
        }
        document.getElementById('main-column').className = 'col-md-12';

        // Clear chat messages except welcome
        while (chatMessages.childNodes.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        
        if (showMessage) {
            // Add reset message
            appendMessage('bot', 'Conversation has been reset. You can start a new search by providing a research title.');
        }
    }
    
    // Focus on input field on load
    userInput.focus();

    // Listen for the final results from the backend

    socket.on('results_ready', function(data) {
        const papers = data.papers;
        const history = data.initial_chat_history || [];
        const introMessage = data.intro_message;
        if (!papers || papers.length === 0) {
            // Handle case where no papers were found
            appendMessage('bot', "My search is complete, but unfortunately, I couldn't find any relevant papers matching your criteria.");
            return;
        }

        // Get the containers for the new UI
        const tabsContainer = document.getElementById('results-tabs');
        const contentContainer = document.getElementById('results-tab-content');

        // Clear previous results but keep the general chat pane
        tabsContainer.innerHTML = '';
        const generalPane = document.getElementById('pane-general');
        contentContainer.innerHTML = '';
        if (generalPane) {
            contentContainer.appendChild(generalPane);
            generalPane.classList.add('show', 'active');
        }

        // Build the tab skeleton with collapsible groups
        tabsContainer.innerHTML = `
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="tab-general" data-bs-toggle="tab" data-bs-target="#pane-general" type="button" role="tab" aria-controls="pane-general" aria-selected="true">General Chat</button>
            </li>
            <li class="nav-item">
                <button class="nav-link" type="button" data-bs-toggle="collapse" data-bs-target="#focused-list" aria-expanded="false">Focused Papers</button>
                <ul class="nav flex-column collapse ms-3" id="focused-list"></ul>
            </li>
            <li class="nav-item">
                <button class="nav-link" type="button" data-bs-toggle="collapse" data-bs-target="#exploratory-list" aria-expanded="false">Exploratory Papers</button>
                <ul class="nav flex-column collapse ms-3" id="exploratory-list"></ul>
            </li>
        `;

        const generalInfo = document.createElement('div');
        generalInfo.className = 'mb-3 text-muted';
        generalInfo.innerHTML = `<p class="small">Use the General Chat tab to ask questions about all ${papers.length} papers.</p>`;
        const paneGeneral = document.getElementById('pane-general');
        if (paneGeneral && !paneGeneral.querySelector('p.small')) {
            paneGeneral.prepend(generalInfo);
        }

        const focusedList = document.getElementById('focused-list');
        const exploratoryList = document.getElementById('exploratory-list');

        // --- Create a Tab and Content Pane for Each Paper ---
        papers.forEach((paper, index) => {
            const tabId = `tab-paper-${index}`;
            const paneId = `pane-paper-${index}`;

            const targetList = paper.category === 'focused' ? focusedList : exploratoryList;
            if (targetList) {
                targetList.innerHTML += `
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="${tabId}" data-bs-toggle="tab" data-bs-target="#${paneId}" type="button" role="tab" aria-controls="${paneId}" aria-selected="false">Paper ${index + 1}</button>
                    </li>
                `;
            }

            // Create the content pane for the tab
            contentContainer.innerHTML += `
                <div class="tab-pane fade" id="${paneId}" role="tabpanel" aria-labelledby="${tabId}" data-doi="${paper.doi}">
                    <div class="p-3" style="height: 500px; overflow-y: auto;">
                        <h6>${paper.title || 'No Title'}</h6>
                        <p class="small text-muted"><strong>Authors:</strong> ${paper.authors ? paper.authors.join(', ') : 'N/A'} (${paper.year || 'N/A'})</p>
                        <p class="small"><strong>DOI:</strong> <a href="${paper.doi}" target="_blank">${paper.doi}</a></p>
                        <hr>
                        <h6>AI Insights</h6>
                        <p class="small"><strong>Summary:</strong> ${paper.insights_summary || 'Not available.'}</p>
                        <p class="small"><strong>Relevance:</strong> ${paper.insights_relevance || 'Not available.'}</p>
                        <hr>
                        <h6>APA Citation</h6>
                        <p class="small font-monospace">${paper.formatted_citation || 'Not available.'}</p>
                        <hr>
                        <h6>Chat About This Paper</h6>
                        <div class="chat-history" style="height: 200px; overflow-y: auto;"></div>
                    </div>
                </div>
            `;
        });

        // --- Finally, reveal the new layout ---
        if (logColumn) {
            logColumn.classList.add('d-none');
            progressLogContainer.style.display = 'none';
            logShown = false;
        }
        document.getElementById('main-column').className = 'col-md-9';
        const tabsCol = document.getElementById('tabs-column');
        if (tabsCol) {
            tabsCol.classList.remove('d-none'); // Show tabbed sidebar on the left
        }

        // Populate general chat with earlier conversation
        const generalHistoryDiv = document.querySelector('#pane-general .chat-history');
        if (generalHistoryDiv) {
            history.forEach(item => {
                const div = document.createElement('div');
                div.className = `message ${item.sender}`;
                const c = document.createElement('div');
                c.className = 'message-content';
                c.innerHTML = item.message.replace(/\n/g, '<br>');
                div.appendChild(c);
                generalHistoryDiv.appendChild(div);
            });
            if (introMessage) {
                const introDiv = document.createElement('div');
                introDiv.className = 'message bot';
                introDiv.innerHTML = `<div class="message-content">${introMessage}</div>`;
                generalHistoryDiv.appendChild(introDiv);
            }
            generalHistoryDiv.scrollTop = generalHistoryDiv.scrollHeight;
        }

        // Ensure tabs behave as a single group
        const tabLinks = tabsContainer.querySelectorAll('button.nav-link[data-bs-toggle="tab"]');
        tabLinks.forEach(link => {
            link.addEventListener('shown.bs.tab', function(e) {
                const targetPane = document.querySelector(e.target.dataset.bsTarget);
                document.querySelectorAll('#results-tab-content .tab-pane').forEach(pane => {
                    if (pane !== targetPane) {
                        pane.classList.remove('show', 'active');
                    }
                });
                tabLinks.forEach(btn => {
                    if (btn !== e.target) {
                        btn.classList.remove('active');
                    }
                });
            });
        });
    });

    // Receive responses for chats after results are ready
    socket.on('sidebar_chat_response', function(data) {
        const activePane = document.querySelector('#results-tab-content .tab-pane.active');
        const targetHistory = activePane ? activePane.querySelector('.chat-history') : chatMessages;
        appendMessage('bot', data.message, targetHistory);
    });

    // Hide the event log if the pipeline fails without delivering results
    socket.on('pipeline_failed', function() {
        if (logColumn) {
            logColumn.classList.add('d-none');
            progressLogContainer.style.display = 'none';
            logShown = false;
        }
        const tabsCol = document.getElementById('tabs-column');
        if (tabsCol) {
            tabsCol.classList.add('d-none');
        }
        document.getElementById('main-column').className = 'col-md-12';
    });

    // Hide the event log when the pipeline is cancelled
    socket.on('pipeline_cancelled', function() {
        if (logColumn) {
            logColumn.classList.add('d-none');
            progressLogContainer.style.display = 'none';
            logShown = false;
        }
    });


});
