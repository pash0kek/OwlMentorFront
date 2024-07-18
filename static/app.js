class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbot__container'),
            sendButton: document.querySelector('.chatbot__send__btn')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "user", message: text1 }
        this.messages.push(msg1);
        this.updateChatText(chatbox);
        textField.value = '';

        this.showTypingIndicator(chatbox);

        fetch('http://127.0.0.1:5000/chat', {
            method: 'POST',
            credentials: 'include',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            this.removeTypingIndicator(chatbox);

            let msg2 = { name: "assistant", message: r.response };
            this.messages.push(msg2);
            this.updateChatText(chatbox);

        }).catch((error) => {
            console.error('Error:', error);
            this.removeTypingIndicator(chatbox);
        });
    }

    showTypingIndicator(chatbox) {
        let typingIndicator = { name: "assistant", message: "Thinking..." };
        this.messages.push(typingIndicator);
        this.updateChatText(chatbox);
    }

    removeTypingIndicator(chatbox) {
        this.messages = this.messages.filter(msg => msg.message !== "Thinking...");
        this.updateChatText(chatbox);
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "assistant")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();