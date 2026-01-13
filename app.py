import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import torch

# Set page configuration
st.set_page_config(
    page_title="Image Captioning App",
    page_icon="🖼️",
    layout="wide"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the BLIP model and processor once at startup."""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def generate_caption(image, processor, model, device):
    """Generate caption for the given image."""
    try:
        # Process the image
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            output = model.generate(**inputs, max_length=50)
        
        # Decode the generated caption
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def load_image_from_url(url):
    """Download and load image from URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image, None
    except requests.exceptions.RequestException as e:
        return None, f"Error downloading image: {str(e)}"
    except Exception as e:
        return None, f"Error loading image: {str(e)}"

# Main app
def main():
    st.title("🖼️ Image Captioning App")
    st.markdown("Generate captions for images using AI-powered vision-language models.")
    
    # Load model
    with st.spinner("Loading model... This may take a moment on first run."):
        processor, model, device = load_model()
    
    if processor is None or model is None:
        st.error("Failed to load model. Please refresh the page.")
        return
    
    st.success(f"Model loaded successfully! Running on: {device.upper()}")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Camera Capture", "🔗 Image URL"])
    
    # Tab 1: Upload Image
    with tab1:
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="upload"
        )
        
        if uploaded_file is not None:
            try:
                # Load and display image
                image = Image.open(uploaded_file).convert("RGB")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(image, processor, model, device)
                    st.markdown("### Generated Caption:")
                    st.info(caption)
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
    
    # Tab 2: Camera Capture
    with tab2:
        st.subheader("Capture Image from Camera")
        camera_image = st.camera_input("Take a picture", key="camera")
        
        if camera_image is not None:
            try:
                # Load and display image
                image = Image.open(camera_image).convert("RGB")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Captured Image", use_container_width=True)
                
                with col2:
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(image, processor, model, device)
                    st.markdown("### Generated Caption:")
                    st.info(caption)
            except Exception as e:
                st.error(f"Error processing camera image: {str(e)}")
    
    # Tab 3: Image URL
    with tab3:
        st.subheader("Load Image from URL")
        url = st.text_input(
            "Enter image URL",
            placeholder="https://example.com/image.jpg",
            key="url"
        )
        
        if st.button("Load and Caption", key="url_button"):
            if url.strip():
                with st.spinner("Downloading image..."):
                    image, error = load_image_from_url(url)
                
                if error:
                    st.error(error)
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption="Image from URL", use_container_width=True)
                    
                    with col2:
                        with st.spinner("Generating caption..."):
                            caption = generate_caption(image, processor, model, device)
                        st.markdown("### Generated Caption:")
                        st.info(caption)
            else:
                st.warning("Please enter a valid URL.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This app uses the BLIP model from Salesforce for image captioning. "
        "All processing is done locally without external API calls."
    )

if __name__ == "__main__":
    main()
