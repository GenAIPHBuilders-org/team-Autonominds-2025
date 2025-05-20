// Initialize chat context from localStorage for persistence
let chatContext = JSON.parse(localStorage.getItem('chatContext') || '{}');

document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-button');
    
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
        if (message) {
            // Special handling for reset/restart
            if (message.toLowerCase() === 'reset' || message.toLowerCase() === 'restart') {
                clearChatContext();
                socket.emit('send_message', {
                    message: message,
                    context: chatContext  // Send empty context
                });
            } else {
                // Display user message
                appendMessage('user', message);
                
                // Send to server with the FULL context
                socket.emit('send_message', {
                    message: message,
                    context: chatContext  // Send the entire context object
                });
            }
            
            // Clear input
            userInput.value = '';
            
            // Show typing indicator
            showTypingIndicator();
        }
    }
    
    // Function to append message to chat
    function appendMessage(sender, message) {
        // Remove typing indicator if exists
        const typingIndicator = document.querySelector('.typing-indicator');
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
        
        // Add to chat
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to clear chat context
    function clearChatContext() {
        chatContext = {};
        localStorage.removeItem('chatContext');
        console.log('Context cleared');
        
        // Clear chat messages except welcome
        while (chatMessages.childNodes.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        
        // Add reset message
        appendMessage('bot', 'Conversation has been reset. You can start a new search by providing a research title.');
    }
    
    // Focus on input field on load
    userInput.focus();
});