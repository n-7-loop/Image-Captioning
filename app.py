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
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image, None
    except requests.exceptions.MissingSchema:
        return None, "Invalid URL format. Please include http:// or https://"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the URL. Please check your internet connection."
    except requests.exceptions.Timeout:
        return None, "Request timed out. The server took too long to respond."
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP Error: {e.response.status_code}. The image could not be accessed."
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
        st.markdown("Paste an image URL below (e.g., from Unsplash, Imgur, or direct image links)")
        
        url = st.text_input(
            "Enter image URL",
            placeholder="https://images.unsplash.com/photo-example.jpg",
            key="url",
            help="Make sure the URL ends with an image extension (.jpg, .png, etc.) or points directly to an image"
        )
        
        # Automatically process URL when it's entered (no button needed)
        if url and url.strip():
            url = url.strip()
            
            # Check if URL starts with http:// or https://
            if not url.startswith(('http://', 'https://')):
                st.warning("⚠️ URL should start with http:// or https://")
            else:
                with st.spinner("Downloading and processing image..."):
                    image, error = load_image_from_url(url)
                
                if error:
                    st.error(f"❌ {error}")
                    st.info("💡 **Tips:**\n- Make sure the URL is a direct link to an image\n- Try right-clicking an image and selecting 'Copy image address'\n- Verify the URL works by pasting it in a new browser tab")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(image, caption="Image from URL", use_container_width=True)
                    
                    with col2:
                        with st.spinner("Generating caption..."):
                            caption = generate_caption(image, processor, model, device)
                        st.markdown("### Generated Caption:")
                        st.success(caption)
        
        # Example URLs section
        with st.expander("📋 Try these example image URLs"):
            st.markdown("""
            Click to copy and paste into the URL field above:
            
            1. `https://images.unsplash.com/photo-1575936123452-b67c3203c357?w=800`
            2. `https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg`
            3. `https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?w=800`
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This app uses the BLIP model from Salesforce for image captioning. "
        "All processing is done locally without external API calls."
    )

if __name__ == "__main__":
    main()
