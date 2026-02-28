import gradio as gr
import cv2
import numpy as np
from face_utils import FaceSystem
import os

# Initialize the face recognition system
face_system = FaceSystem(known_faces_dir="known_faces")

def image_process(image):
    if image is None:
        return None
    # Convert image for processing
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_image = face_system.process_frame(image_bgr)
    # Convert back for Gradio
    return cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

def webcam_process(frame):
    if frame is None:
        return None
    # Frame is already an array
    processed_frame = face_system.process_frame(frame)
    return processed_frame

# Create the Gradio Interface
with gr.Blocks(title="AI Face Detection & Recognition") as demo:
    gr.Markdown("# AI Face Detection & Recognition")
    gr.Markdown("Upload an image or use your webcam to detect and recognize faces. Add photos of known people to the 'known_faces' folder for identification.")
    
    with gr.Tab("Image Mode"):
        with gr.Row():
            input_img = gr.Image(label="Input Image")
            output_img = gr.Image(label="Output Image")
        btn = gr.Button("Analyze Image")
        btn.click(fn=image_process, inputs=input_img, outputs=output_img)
        
    with gr.Tab("Webcam Mode"):
        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Stream", type="numpy")
        webcam_output = gr.Image(label="Processed Stream")
        webcam_input.stream(fn=webcam_process, inputs=webcam_input, outputs=webcam_output, time_limit=30, stream_every=0.1)

if __name__ == "__main__":
    demo.launch()
