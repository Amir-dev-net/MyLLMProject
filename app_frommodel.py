import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_path = "./my_llm"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Function to generate a response
def generate_response(question):
    inputs = tokenizer(question, return_tensors="pt")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Create Gradio UI
chat_ui = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="Custom LLM Chatbot",
    description="Ask me anything from my fine-tuned dataset!"
)

chat_ui.launch()
