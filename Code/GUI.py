import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from model import OurCNN  
import numpy as np
from predict import predictor

dinfo = {
    "Black Rot": {
        "description": "Grape black rot is a fungal disease caused by an ascomycetous fungus, Guignardia bidwellii...",
        "wiki": "https://en.wikipedia.org/wiki/Black_rot_(grape_disease)"
    },
    "ESCA": {
        "description": "Esca is a grape disease of mature grapevines...",
        "wiki": "https://en.wikipedia.org/wiki/Esca_(disease)"
    },
    "Healty": {
        "description": "No signs of disease. Leaf appears green, normal and HEALTY.",
        "wiki": "https://en.wikipedia.org/wiki/Vineyard"
    },
    "Leaf Blight": {
        "description": "Causes irregular brown and yellow lesions on leaves, often leading to defoliation.",
        "wiki": "https://en.wikipedia.org/wiki/Leaf_blight"
    }
}

tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

labels = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

def show_disease_info(disease_name):
    info = dinfo[disease_name]
    link = f"<a href='{info['wiki']}' target='_blank' style='color:#6a0dad;'>🔗 Learn more on Wikipedia</a>"
    return f"**{disease_name}**\n\n{info['description']}\n\n{link}"

with gr.Blocks() as ui:
    gr.HTML("""
    <style>
        /* Import font globally */
        body {
            font-family: 'Raleway', sans-serif;
            background: linear-gradient(to bottom right, #fef6ff, #e6d1f2);
            margin: 0;
            padding: 0;
        }

        h1 {
            font-weight: 700;
            font-size: 5rem;
            text-align: center;
            color: #311432;
            text-shadow: 2px 2px 4px #4b0082;
            margin-bottom: 0.5em;
        }

        .gr-button {
            background-color: #6a0dad !important;
            color: #f9f9f9 !important;
            font-weight: bold;
            border-radius: 12px;
            padding: 10px 25px;
            transition: background-color 0.3s ease;
            border: none !important;
        }

        .gr-button:hover {
            background-color: #53128d !important;
        }

        .gr-textbox {
            border: 2px solid #6a0dad !important;
            color: #4b0082 !important;
            font-weight: 600;
            border-radius: 10px;
        }

        .gr-radio input[type="radio"]:checked + label {
            color: #6a0dad !important;
            font-weight: 700;
        }

        .gr-markdown {
            color: #6a0dad !important;
        }
    </style>

    <link href="https://fonts.googleapis.com/css2?family=Raleway:wght@700&display=swap" rel="stylesheet">
    <h1>🍇 Pernosphere 🍇</h1>
    """)

    with gr.Tabs():
        with gr.Tab("🍇Classifier"):
            image = gr.Image(type="pil", height=600, width=600)
            output = gr.Textbox(label="Prediction", interactive=False)
            btn = gr.Button("Classify")
            btn.click(fn=predictor, inputs=image, outputs=output)

        with gr.Tab("Info-point 📚"):
            gr.Markdown("### Select a grape disease to learn about it:")
            disease_selector = gr.Radio(
                choices=list(dinfo.keys()), 
                label="Disease",
            )
            disease_info_output = gr.Markdown()
            disease_selector.change(fn=show_disease_info, inputs=disease_selector, outputs=disease_info_output)

        with gr.Tab("Course 🎓"):
            gr.Markdown(
                """
                ### 🎓 Sapienza University - Course Page  
                Visit the official course page here:  
                [🔗 Click to open the course site](https://corsidilaurea.uniroma1.it/it/view-course-details/2023/30786/20190322090929/197d55d9-5acb-4191-a987-c8b7be267506/54c4f618-4005-4e6e-b494-b317b4661eb4/5bdc55e7-ac18-48d8-a75c-a7e4dab0f26c/3583e1b7-4c99-4d44-811f-53b3082bad7f)
                """
            )

        with gr.Tab("Me 👨‍💻"):
            gr.Markdown(
                """
                ### 👨‍💻 About Me  
                Carlo Da Roma from Sapienza University for the AI Lab course.  
                GitHub: [🔗 github.com/CarloDaRomadev](https://github.com/CarloDaRomadev)
                """
            )
                
if __name__ == "__main__":
    ui.launch()
