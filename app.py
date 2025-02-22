import json
import gradio as gr
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./my_llm"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Load JSON dataset (contains question-answer pairs)
def load_json_data():
    with open("data.json", "r") as f:
        return json.load(f)

data = load_json_data()

# Convert dataset into a dictionary for quick lookup
qa_dict = {item["question"]: item["answer"] for item in data}

# ✅ Function to clean text (remove punctuation, lowercase)
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# ✅ Function to show chatbot introduction in a popup
def chatbot_intro():
    return (
        "**🤖 Welcome to My Custom LLM Chatbot! 🚀**\n\n"
        "This chatbot is designed to answer coding-related and company-specific questions.\n"
        "It uses a fine-tuned LLM model and a predefined knowledge base to provide accurate responses.\n\n"
        "**How to Use:**\n"
        "- Type your question in the input box.\n"
        "- If your question matches our database, you’ll get an instant answer.\n"
        "- Otherwise, the chatbot will notify you that the topic is under development.\n\n"
        "💡 Try asking coding-related questions like: `How do I read a JSON file in Python?`"
    )

# ✅ Function to get response with chat history
def chat_response(user_input, chat_history):
    user_input_clean = clean_text(user_input)  # Normalize user input

    # ✅ Check if any known question is a **substring** in the user input
    for question, answer in qa_dict.items():
        if clean_text(question) in user_input_clean:  # Compare cleaned versions
            response = answer
            break
    else:
        response = "I'm sorry, but I don't have an answer for that yet. This topic is still under development."

    chat_history.append(("User", user_input))
    chat_history.append(("Bot", response))
    return "", chat_history  # Reset input and update history

# ✅ Create ChatGPT-style UI
with gr.Blocks() as iface:
    # ✅ Show Introduction Popup
    with gr.Row():
        gr.Markdown(chatbot_intro(), elem_id="intro-popup")

    # ✅ Chatbot UI Components
    chatbot = gr.Chatbot(label="Chat with My LLM")
    user_input = gr.Textbox(placeholder="Ask a coding or company-related question...")
    submit_btn = gr.Button("Submit", interactive=False)  # Disabled initially

    # ✅ Enable button only if input is not empty
    def toggle_button(text):
        return gr.update(interactive=bool(text.strip()))

    user_input.change(toggle_button, user_input, submit_btn)

    # ✅ Submit button action with chat history
    submit_btn.click(chat_response, [user_input, chatbot], [user_input, chatbot])

# Run the UI
iface.launch(pwa=True)
