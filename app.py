import gradio as gr
from inference import full_pipeline

# =========================
# DISEASE EXPLANATIONS
# =========================
disease_info = {
    "CNV": "Choroidal Neovascularization (CNV) involves abnormal blood vessel growth beneath the retina. It can cause leakage, bleeding, and vision distortion.",
    
    "DME": "Diabetic Macular Edema (DME) is caused by fluid accumulation in the macula due to diabetes, leading to retinal swelling and blurred vision.",
    
    "DRUSEN": "Drusen are yellow deposits under the retina, often associated with early age-related macular degeneration (AMD).",
    
    "NORMAL": "The retina appears normal with no visible signs of disease or abnormalities."
}


# =========================
# MAIN FUNCTION
# =========================
def run_model(image_path):
    try:
        prediction, confidence, top2, gradcam_img = full_pipeline(image_path)

        # Prediction text
        pred_text = f"### Prediction: **{prediction}**"
        conf_text = f"{confidence * 100:.2f}%"

        # Top-2 formatting
        top2_text = ""
        for i, (label, score) in enumerate(top2):
            top2_text += f"{i+1}. {label} → {score*100:.2f}%\n"

        # Disease explanation
        explanation = disease_info.get(prediction, "No information available.")

        return pred_text, conf_text, top2_text, explanation, gradcam_img

    except Exception as e:
        return f"Error: {str(e)}", "0%", "", "", None


# =========================
# UI DESIGN
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # Title
    gr.Markdown("# Retinal OCT Classification (EfficientNet + CBAM + Random Forest)")
    gr.Markdown(
        "Upload an OCT image to get **Prediction, Confidence, Top-2 results, Grad-CAM, and disease explanation**."
    )

    # Disclaimer
    gr.Markdown("**Disclaimer:** This is an AI-assisted tool and NOT a medical diagnosis.")

    # Layout
    with gr.Row():

        # LEFT
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload OCT Image")

            submit_btn = gr.Button("Analyze", variant="primary")
            clear_btn = gr.Button("Clear")

        # RIGHT
        with gr.Column():
            prediction_output = gr.Markdown(label="Prediction")
            confidence_output = gr.Textbox(label="Confidence")
            top2_output = gr.Textbox(label="Top-2 Predictions")

    #  DISEASE INFO
    gr.Markdown("### Disease Explanation")
    explanation_output = gr.Textbox(label="About the Disease", lines=4)

    # Grad-CAM
    gr.Markdown("### Grad-CAM Visualization")
    gr.Markdown("Red regions indicate where the model is focusing.")

    gradcam_output = gr.Image(label="Grad-CAM Output")

    # Examples
    gr.Markdown("### Example Images")
    gr.Examples(
        examples=[
            "DRUSEN-53018-1.jpeg",
            "CNV-9997680-9.jpeg",
            "NORMAL-9251-5.jpeg",
            "dme_2287657_2.jpg"
        ],
        inputs=image_input
    )

    # Footer
    gr.Markdown("---")
    gr.Markdown("Built by **PaniChendra** | EfficientNet + CBAM + Random Forest + Grad-CAM ")

    # =========================
    # BUTTON ACTIONS
    # =========================
    submit_btn.click(
        fn=run_model,
        inputs=image_input,
        outputs=[
            prediction_output,
            confidence_output,
            top2_output,
            explanation_output,
            gradcam_output
        ]
    )

    clear_btn.click(
        fn=lambda: ("", "", "", "", None),
        inputs=[],
        outputs=[
            prediction_output,
            confidence_output,
            top2_output,
            explanation_output,
            gradcam_output
        ]
    )


# =========================
# RUN
# =========================
demo.launch()