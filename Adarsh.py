import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

MODEL = "grok-3"

def get_ai_response(message, chat_history, current_code, extra_instruction):
    context = f"Current code file:\n```python\n{current_code}\n```\n\n"
    if extra_instruction:
        context += f"User instruction: {extra_instruction}\n\n"
    
    messages = [{"role": "system", "content": "You are an expert Python coding assistant. Provide helpful, accurate code suggestions, explanations, refactors, or bug fixes. Use markdown and code blocks."}]
    
    for human, ai in chat_history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})
    
    full_query = context + message
    messages.append({"role": "user", "content": full_query})
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}\n\nMake sure your XAI_API_KEY is set correctly."

def extract_code_from_response(current_code, ai_text):
    if "```python" in ai_text:
        code = ai_text.split("```python")[-1].split("```")[0].strip()
        return code
    return current_code

with gr.Blocks(title="My Python Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 My Python Assistant\nPersonal AI Coding Tool")
    
    with gr.Tab("Editor + Chat"):
        with gr.Row():
            with gr.Column(scale=2):
                code_editor = gr.Code(
                    label="Code Editor (Python)",
                    language="python",
                    value="# Welcome to my coding assistant!\n\ndef hello():\n    print('Hello from Adarsh!')\n\nhello()",
                    lines=30
                )
                apply_btn = gr.Button("Apply Edit to Code", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Chat")
                chatbot = gr.Chatbot(height=400, label="Chat")
                msg = gr.Textbox(placeholder="Ask anything: explain, refactor, add feature, fix bugs...", label="Message")
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    send_btn = gr.Button("Send", variant="primary")
                
                instruction = gr.Textbox(
                    label="Extra Instruction (optional)",
                    placeholder="e.g. Refactor for performance"
                )
    
    with gr.Tab("Settings"):
        model_dropdown = gr.Dropdown(["grok-3", "grok-build-0.1"], value=MODEL, label="Model")
        gr.Markdown("**Tip**: Change model if needed.")

    def respond(message, history, code, instr):
        if not message:
            return history, code
        bot_response = get_ai_response(message, history, code, instr)
        history.append((message, bot_response))
        return history, code

    send_btn.click(
        respond,
        inputs=[msg, chatbot, code_editor, instruction],
        outputs=[chatbot, code_editor]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, code_editor, instruction],
        outputs=[chatbot, code_editor]
    )
    
    def apply_edit(history, current_code):
        if not history:
            return current_code
        last_reply = history[-1][1]
        new_code = extract_code_from_response(current_code, last_reply)
        return new_code
    
    apply_btn.click(
        apply_edit,
        inputs=[chatbot, code_editor],
        outputs=[code_editor]
    )
    
    clear_btn.click(lambda: [], outputs=[chatbot])

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)