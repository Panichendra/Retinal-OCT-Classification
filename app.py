import gradio as gr
from inference import predict_only, generate_gradcam


def predict_fn(image_path):
    try:
        # Prediction
        prediction, confidence, pred_idx, img_tensor, img = predict_only(image_path)

        # GradCAM (separate step)
        gradcam_path = generate_gradcam(pred_idx, img_tensor, img)

        return prediction, round(confidence, 4), gradcam_path

    except Exception as e:
        return f"Error: {str(e)}", 0, None


demo = gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Number(label="Confidence"),
        gr.Image(label="Grad-CAM")
    ],
    title="Retinal OCT Classification (EfficientNet + CBAM)",
    description="Upload an OCT image to get prediction + GradCAM visualization"
)

if __name__ == "__main__":
    demo.launch()
