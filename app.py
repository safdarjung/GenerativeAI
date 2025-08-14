# AI Video Summarizer Pro - Local Video, Image & YouTube
# pip install streamlit langchain-google-genai openai-whisper moviepy scikit-image yt-dlp transformers

import base64
import cv2
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from typing import List, Dict
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time
import json
from datetime import datetime
import hashlib
import yt_dlp
import re  # Added for better hashtag parsing
import torch
from PIL import Image
import torchvision.transforms as transforms
import io
import math # Added for SinusoidalPosEmb
from transformers import BertTokenizer  # Added for token decoding

# Model architecture classes from your notebook
def extract_patches(image_tensor, patch_size=16):
    # Get the dimensions of the image tensor
    bs, c, h, w = image_tensor.size()

    # Define the Unfold layer with appropriate parameters
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    # Apply Unfold to the image tensor
    unfolded = unfold(image_tensor)

    # Reshape the unfolded tensor to match the desired output shape
    # Output shape: BSxLxH, where L is the number of patches in each dimension
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

    return unfolded

# sinusoidal positional embeds
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Define a decoder module for the Transformer architecture
class Decoder(torch.nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()

        # Create an embedding layer for tokens
        self.embedding = torch.nn.Embedding(num_emb, hidden_size)
        # Initialize the embedding weights
        self.embedding.weight.data = 0.001 * self.embedding.weight.data

        # Initialize sinusoidal positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        # Create multiple decoder layers
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads,
                                                   dim_feedforward=hidden_size * 4, dropout=0.0,
                                                   batch_first=True)
        # TransformerDecoder will clone the decoder_layer "num_layers" times
        self.decoder_layers = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Define a linear layer for output prediction
        self.fc_out = torch.nn.Linear(hidden_size, num_emb)

    def forward(self, input_seq, encoder_output, input_padding_mask=None,
                encoder_padding_mask=None):
        # Embed the input sequence
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        # Add positional embeddings to the input embeddings
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb
        causal_mask = torch.triu(torch.ones(l, l, device=input_seq.device), 1).bool()

        # Pass the embeddings through each transformer block
        output = self.decoder_layers(tgt=embs, memory=encoder_output, tgt_mask=causal_mask,
                                     tgt_key_padding_mask=input_padding_mask,
                                     memory_key_padding_mask=encoder_padding_mask)

        return self.fc_out(output)

# Define an Vision Encoder module for the Transformer architecture
class VisionEncoder(torch.nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16, hidden_size=128, num_layers=3, num_heads=4):
        super(VisionEncoder, self).__init__()

        self.patch_size = patch_size
        self.fc_in = torch.nn.Linear(channels_in * patch_size * patch_size, hidden_size)

        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = torch.nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.02))

        # Create multiple transformer blocks as layers
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                                   dim_feedforward=hidden_size * 4, dropout=0.0,
                                                   batch_first=True)
        # TransformerEncoder will clone the encoder_layer "num_layers" times
        self.encoder_layers = torch.nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, image):
        bs = image.shape[0]

        patch_seq = extract_patches(image, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)

        # Add a unique embedding to each token embedding
        embs = patch_emb + self.pos_embedding

        # Pass the embeddings through each transformer block
        output = self.encoder_layers(embs)

        return output

# Define an Vision Encoder-Decoder module for the Transformer architecture
class VisionEncoderDecoder(torch.nn.Module):
    def __init__(self, image_size, channels_in, num_emb, patch_size=16,
                 hidden_size=128, num_layers=(3, 3), num_heads=4):
        super(VisionEncoderDecoder, self).__init__()

        # Create an encoder and decoder with specified parameters
        self.encoder = VisionEncoder(image_size=image_size, channels_in=channels_in, patch_size=patch_size,
                               hidden_size=hidden_size, num_layers=num_layers[0], num_heads=num_heads)

        self.decoder = Decoder(num_emb=num_emb, hidden_size=hidden_size,
                               num_layers=num_layers[1], num_heads=num_heads)

    def forward(self, input_image, target_seq, padding_mask):
        # Generate padding masks for the target sequence
        bool_padding_mask = padding_mask == 0

        # Encode the input sequence
        encoded_seq = self.encoder(image=input_image)

        # Decode the target sequence using the encoded sequence
        decoded_seq = self.decoder(input_seq=target_seq,
                                   encoder_output=encoded_seq,
                                   input_padding_mask=bool_padding_mask)
        return decoded_seq

SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "webp", "bmp"]
MAX_FILE_SIZE_MB = 200

def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e8ed;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(45deg, #d4edda, #ffffff);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(45deg, #f8f9fa, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_image_for_llm(image_file) -> List[Dict[str, any]]:
    try:
        image_data = image_file.read()
        if len(image_data) > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"❌ Image too large: {len(image_data)/1024/1024:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")
            return []
        base64_image = base64.b64encode(image_data).decode("utf-8")
        return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return []

@st.cache_data
def get_transcript_from_video(video_path: str) -> str:
    video_path_obj = Path(video_path)
    audio_path = video_path_obj.with_suffix(".wav")
    try:
        with VideoFileClip(video_path) as video_clip:
            if video_clip.audio:
                video_clip.audio.write_audiofile(str(audio_path), logger=None)
            else:
                return "No audio track found in this video."
        # Use the faster 'tiny' model
        model = whisper.load_model("tiny")
        result = model.transcribe(str(audio_path))
        transcript = result.get("text", "").strip()
        return transcript if transcript else "Could not extract meaningful speech from audio."
    except Exception as e:
        if "ffmpeg" in str(e).lower():
            st.error("❌ Transcription failed: FFmpeg not found")
            st.info("💡 **To enable transcription, install FFmpeg:**")
            st.info("• **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`")
            st.info("• **WSL/Windows:** Download from https://ffmpeg.org/download.html")
            st.info("• **macOS:** `brew install ffmpeg`")
            return "Transcription unavailable: FFmpeg not installed. Install FFmpeg to enable audio transcription."
        else:
            st.error(f"❌ Transcription failed: {e}")
        return "Transcription unavailable due to processing error."
    finally:
        if audio_path.exists():
            audio_path.unlink()

def inspect_model_file(model_path: str):
    """Debug function to inspect the contents of a model file"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        st.markdown("### 🔍 Model File Inspection")
        
        if isinstance(checkpoint, dict):
            st.info(f"📁 Checkpoint contains {len(checkpoint)} keys:")
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    st.text(f"  • {key}: Tensor {list(value.shape)}")
                elif isinstance(value, torch.nn.Module):
                    st.text(f"  • {key}: PyTorch Module")
                elif isinstance(value, dict):
                    st.text(f"  • {key}: Dictionary with {len(value)} items")
                else:
                    st.text(f"  • {key}: {type(value).__name__} - {str(value)[:100]}")
        else:
            st.info(f"📁 Direct model object of type: {type(checkpoint).__name__}")
            if hasattr(checkpoint, 'eval'):
                st.success("✅ Object has eval() method - this looks like a valid model!")
            else:
                st.warning("⚠️ Object doesn't have eval() method")
        
        return checkpoint
        
    except Exception as e:
        st.error(f"❌ Failed to inspect model file: {e}")
        return None

def load_custom_captioning_model(model_path: str):
    """Load the custom trained captioning model"""
    try:
        # Load your custom model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        st.info(f"🔍 Loading model on device: {device}")
        
        # Handle different model saving formats
        if isinstance(checkpoint, dict):
            st.info(f"📁 Checkpoint contains {len(checkpoint)} keys: {list(checkpoint.keys())}")
            
            # If it's a state dictionary or checkpoint dictionary
            if 'model' in checkpoint:
                # Checkpoint contains model in 'model' key
                model = checkpoint['model']
                st.success("✅ Found model in 'model' key")
            elif 'state_dict' in checkpoint:
                # Checkpoint contains state_dict - need to recreate model architecture
                st.info("🔄 Recreating model architecture from state_dict...")
                model = recreate_model_from_checkpoint(checkpoint)
                if model is None:
                    return None, None
                st.success("✅ Model architecture recreated and state_dict loaded")
            elif 'model_state_dict' in checkpoint:
                # Checkpoint contains model_state_dict - need to recreate model architecture
                st.info("🔄 Recreating model architecture from model_state_dict...")
                model = recreate_model_from_checkpoint(checkpoint)
                if model is None:
                    return None, None
                st.success("✅ Model architecture recreated and state_dict loaded")
            elif 'weights' in checkpoint:
                # Checkpoint contains weights
                model = checkpoint['weights']
                st.success("✅ Found model in 'weights' key")
            elif 'checkpoint' in checkpoint:
                # Checkpoint contains checkpoint
                model = checkpoint['checkpoint']
                st.success("✅ Found model in 'checkpoint' key")
            else:
                # Try to find any key that might contain the model
                model_keys = [k for k in checkpoint.keys() if isinstance(checkpoint[k], torch.nn.Module)]
                if model_keys:
                    model = checkpoint[model_keys[0]]
                    st.success(f"✅ Found model in key: {model_keys[0]}")
                else:
                    # Show all available keys for debugging
                    st.error("❌ Could not find model in checkpoint.")
                    st.info("Available keys:")
                    for key, value in checkpoint.items():
                        st.text(f"  • {key}: {type(value).__name__}")
                    st.info("💡 **Solution:** Your model file might be a state_dict. You need to create the model architecture first.")
                    return None, None
        else:
            # Direct model object
            model = checkpoint
            st.success("✅ Direct model object loaded")
        
        # Ensure model is in evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
            st.success("✅ Model set to evaluation mode")
        else:
            st.warning("⚠️ Loaded object doesn't have eval() method. This might not be a valid PyTorch model.")
            return None, None
        
        # Check if model has forward method
        if hasattr(model, 'forward'):
            st.success("✅ Model has forward method")
        else:
            st.warning("⚠️ Model doesn't have forward method")
        
        # Move model to device
        model = model.to(device)
        st.success(f"✅ Model moved to {device}")
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Failed to load custom model: {e}")
        st.info("💡 **Troubleshooting:**")
        st.info("1. Make sure your model file is a valid PyTorch model (.pt file)")
        st.info("2. Try using the 'Inspect Model File' button to see what's inside")
        st.info("3. If it's a state_dict, you need to create the model architecture first")
        return None, None

def recreate_model_from_checkpoint(checkpoint):
    """Recreate the model architecture and load state_dict from checkpoint"""
    try:
        # Extract model parameters from checkpoint
        # You may need to adjust these based on your specific model configuration
        model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
        
        if model_state_dict is None:
            st.error("❌ No model_state_dict found in checkpoint")
            return None
        
        # Get configuration from session state if available, otherwise use defaults
        if hasattr(st, 'session_state') and 'model_config' in st.session_state:
            config = st.session_state.model_config
            image_size = config['image_size']
            channels_in = config['channels_in']
            num_emb = config['num_emb']
            patch_size = config['patch_size']
            hidden_size = config['hidden_size']
            num_layers = config['num_layers']
            num_heads = config['num_heads']
        else:
            # Default parameters - you may need to adjust these
            # These should match the parameters used when training your model
            image_size = 128  # From your training (128/8 = 16, 16² = 256 patches)
            channels_in = 3    # RGB images
            num_emb = 30522    # Default vocabulary size (you may need to adjust this)
            patch_size = 8     # From your notebook
            hidden_size = 192  # From your notebook
            num_layers = (6, 6)  # From your notebook
            num_heads = 8      # From your notebook
        
        st.info(f"🔧 Creating model with parameters:")
        st.info(f"  • Image size: {image_size}")
        st.info(f"  • Channels: {channels_in}")
        st.info(f"  • Vocabulary size: {num_emb}")
        st.info(f"  • Patch size: {patch_size}")
        st.info(f"  • Hidden size: {hidden_size}")
        st.info(f"  • Layers: {num_layers}")
        st.info(f"  • Heads: {num_heads}")
        
        # Create the model architecture
        model = VisionEncoderDecoder(
            image_size=image_size,
            channels_in=channels_in,
            num_emb=num_emb,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Load the state_dict
        model.load_state_dict(model_state_dict)
        st.success("✅ State dict loaded successfully")
        
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        st.success(f"✅ Model moved to {device}")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to recreate model: {e}")
        st.info("💡 **Solution:** You may need to adjust the model parameters to match your training configuration")
        return None

def generate_caption_with_custom_model(model, device, image_data, temperature: float = 0.1, max_length: int = 50):
    """Generate caption using custom trained model with autoregressive decoding"""
    try:
        # Convert base64 image data to PIL Image
        if isinstance(image_data, dict) and 'image_url' in image_data:
            # Extract base64 data from the image_url format
            base64_str = image_data['image_url']['url'].split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            # Handle direct image data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transformations based on your model's requirements
        # Your model expects 128x128 images based on the architecture
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Match your model's expected input size
            transforms.ToTensor(),
            # Note: Your model might not need normalization, but adding it for safety
            # You can remove this if your model was trained without normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Ensure model is on the same device as input tensors
        model = model.to(device)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Encode the image once
        with torch.no_grad():
            encoded_seq = model.encoder(image=image_tensor)
        
        # Start with BOS token (e.g., 101 for [CLS] in BERT)
        bos_token = 101  # Adjust based on your tokenizer (BERT [CLS])
        generated_tokens = [bos_token]
        
        for _ in range(max_length - 1):
            # Prepare input sequence and padding mask
            target_seq = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            padding_mask = torch.ones_like(target_seq)  # All valid so far
            
            # Decode
            output = model.decoder(
                input_seq=target_seq,
                encoder_output=encoded_seq,
                input_padding_mask=(padding_mask == 0)
            )
            
            # Get logits for next token
            next_token_logits = output[:, -1, :] / temperature  # Apply temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(next_token_probs, dim=-1).item()
            
            generated_tokens.append(next_token)
            
            if next_token == 102:  # EOS token in BERT ([SEP])
                break
        
        # Decode to text
        caption = convert_tokens_to_text(torch.tensor(generated_tokens))
        
        return caption
        
    except Exception as e:
        st.error(f"❌ Custom model caption generation failed: {e}")
        return f"Caption generation failed: {str(e)}"

def convert_tokens_to_text(tokens):
    """Convert token IDs to text using the tokenizer"""
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Load tokenizer (cache it if possible)
        token_list = tokens.cpu().numpy().tolist()
        # Filter out padding (0), start (101), and stop at EOS (102)
        filtered_tokens = []
        for t in token_list:
            if t == 102:  # EOS token
                break
            if t != 0 and t != 101:  # Skip padding and BOS
                filtered_tokens.append(t)
        
        if not filtered_tokens:
            return "No valid caption generated"
        
        # Decode to text
        caption = tokenizer.decode(filtered_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return caption.strip().capitalize() + '.'  # Make it sentence-like
        
    except Exception as e:
        return f"Token conversion failed: {str(e)}"

def generate_batch_captions_custom_model(model, device, batch_frames: List[Dict[str, any]], temperature: float = 0.1) -> tuple[List[str], List[List[str]]]:
    """Generate captions using custom model instead of Gemini"""
    captions = []
    batch_hashtags = []
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    for frame in batch_frames:
        try:
            # Generate caption using custom model
            caption = generate_caption_with_custom_model(model, device, frame, temperature)
            
            # Generate hashtags based on the caption (you can enhance this)
            hashtags = generate_hashtags_from_caption(caption)
            
            captions.append(caption)
            batch_hashtags.append(hashtags)
            
        except Exception as e:
            st.warning(f"⚠️ Frame caption generation failed: {e}")
            captions.append(f"Frame analysis unavailable")
            batch_hashtags.append([])
    
    return captions, batch_hashtags

def generate_hashtags_from_caption(caption: str) -> List[str]:
    """Generate hashtags from caption text"""
    # Simple hashtag generation - you can enhance this
    words = caption.lower().split()
    hashtags = []
    
    # Common objects, actions, and themes
    common_tags = ['nature', 'people', 'city', 'food', 'travel', 'art', 'sports', 'technology']
    
    for word in words:
        if len(word) > 3 and word.isalpha():
            # Convert to CamelCase
            hashtag = '#' + word.title()
            hashtags.append(hashtag)
    
    # Add some common tags based on content
    for tag in common_tags:
        if tag in caption.lower():
            hashtags.append(f"#{tag.title()}")
    
    return hashtags[:3]  # Return max 3 hashtags

def extract_keyframes(video_path: str, scene_change_threshold: float, min_time_between_frames: int, max_frames: int, progress_callback=None) -> List[Dict[str, any]]:
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Display video metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Frames", f"{total_frames:,}")
        with col2:
            st.metric("🎬 Frame Rate", f"{fps:.1f} FPS")
        with col3:
            st.metric("⏱️ Duration", f"{duration:.1f}s")
        with col4:
            st.metric("🎯 Target Keyframes", max_frames)
        
        prev_frame = None
        last_frame_time = -min_time_between_frames
        frame_num = 0
        
        while cap.isOpened() and len(frames) < max_frames:
            success, frame = cap.read()
            if not success:
                break
            current_time = frame_num / fps
            if current_time - last_frame_time >= min_time_between_frames:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                is_keyframe = False
                if prev_frame is not None:
                    prev_resized = cv2.resize(prev_frame, (128, 72))
                    curr_resized = cv2.resize(gray_frame, (128, 72))
                    ssim_score, _ = ssim(prev_resized, curr_resized, full=True)
                    hist1 = cv2.calcHist([prev_resized], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([curr_resized], [0], None, [256], [0, 256])
                    hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    if ssim_score < scene_change_threshold or hist_corr < 0.85:
                        is_keyframe = True
                else:
                    is_keyframe = True
                if is_keyframe:
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    base64_frame = base64.b64encode(buffer).decode("utf-8")
                    frames.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"},
                        "timestamp": current_time,
                        "frame_number": frame_num,
                        "time_formatted": f"{int(current_time//60)}:{int(current_time%60):02d}"
                    })
                    last_frame_time = current_time
                prev_frame = gray_frame
            frame_num += 1
            if progress_callback and frame_num % 60 == 0:
                progress = min(len(frames) / max_frames, frame_num / total_frames)
                progress_callback(progress, len(frames), current_time)
        cap.release()
        return frames
    except Exception as e:
        st.error(f"❌ Keyframe extraction failed: {e}")
        return []

@st.cache_data
def generate_batch_captions(_llm, batch_frames: List[Dict[str, any]], temperature: float = 0.1) -> tuple[List[str], List[List[str]]]:
    prompt = """
    You are an expert analyst and social media expert. For each image, provide:
    1. A detailed, descriptive caption (1-2 sentences)
    2. 2-3 creative, relevant hashtags for each image, focusing on objects, themes, actions, moods, and unique visual elements. Use CamelCase, no spaces, each hashtag starts with #.

    **INSTRUCTIONS:**
    - Analyze each image thoroughly
    - Describe the main subject, action, setting, and visual elements
    - Be specific about colors, objects, people, and activities
    - Keep each caption concise but informative (1-2 sentences)
    - Number each caption clearly
    - After each caption, list 2-3 hashtags on new lines

    **OUTPUT FORMAT:**
    1. [Caption for first image]
    #HashtagOne #HashtagTwo #HashtagThree
    2. [Caption for second image]
    #HashtagFour #HashtagFive
    ...and so on.

    Begin analysis:
    """

    try:
        content = batch_frames + [{"type": "text", "text": prompt}]
        message = HumanMessage(content=content)
        response = _llm.invoke(
            [message],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 1800,
                "top_p": 0.9,
                "top_k": 40
            }
        )
        captions = []
        batch_hashtags = []
        lines = response.content.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line and any(line.startswith(f"{j}.") for j in range(1, 31)):
                caption = line.split('.', 1)[1].strip() if '.' in line else line
                if caption and len(caption) > 5:
                    captions.append(caption)
                # Look for hashtags in next lines
                hashtags = []
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith('#'):
                    hashtags.extend([tag for tag in lines[j].strip().split() if tag.startswith('#')])
                    j += 1
                batch_hashtags.append(hashtags)
                i = j
            else:
                i += 1
        while len(captions) < len(batch_frames):
            captions.append(f"Visual content in frame {len(captions)+1}")
            batch_hashtags.append([])
        return captions[:len(batch_frames)], batch_hashtags[:len(batch_frames)]
    except Exception as e:
        st.warning(f"⚠️ Caption/hashtag generation error: {e}")
        return [f"Frame {i+1}: Analysis unavailable" for i in range(len(batch_frames))], [[] for _ in range(len(batch_frames))]

@st.cache_data
def generate_comprehensive_summary(_llm, captions: List[str], transcript: str, content_info: Dict = None) -> str:
    try:
        context_parts = []
        if content_info and content_info.get("title"):
            context_parts.append(f"**Title:** {content_info['title']}")
        if content_info and content_info.get("duration"):
            duration = content_info['duration']
            context_parts.append(f"**Duration:** {duration//60}:{duration%60:02d}")
        valid_captions = [cap for cap in captions if not ("unavailable" in cap.lower() or len(cap) < 10)]
        if valid_captions:
            context_parts.append(f"**Visual Analysis ({len(valid_captions)} keyframes):**")
            for i, caption in enumerate(valid_captions[:10], 1):
                context_parts.append(f"{i}. {caption}")
        if transcript and len(transcript.strip()) > 20:
            context_parts.append(f"**Audio Transcript:**\n{transcript[:1000]}...")
        context = "\n".join(context_parts)
        
        prompt = f"""
        Analyze this content and create a comprehensive summary.

        {context}

        **TASK:** Write a detailed 2-3 sentence summary that captures:
        1. The main topic, theme, or subject matter
        2. Key activities, events, or information presented
        3. The overall purpose, message, or takeaway
        4. Notable visual or audio elements

        **STYLE:** Professional, informative, and engaging. Write for someone who hasn't seen the content.

        **Summary:**
        """

        message = HumanMessage(content=[{"type": "text", "text": prompt}])
        response = _llm.invoke([message], generation_config={"temperature": 0.1, "max_output_tokens": 300})
        return response.content.strip()
    except Exception as e:
        st.error(f"❌ Summary generation failed: {e}")
        return "Unable to generate comprehensive summary due to processing error."

@st.cache_data
def generate_smart_hashtags(_llm, summary: str, captions: List[str], content_info: Dict = None, temperature: float = 0.3) -> tuple[List[str], str]:
    try:
        context = f"Content Summary: {summary}"
        if content_info and content_info.get("title"):
            context += f"\nContent Title: {content_info['title']}"
        if captions:
            context += "\nKey Visual Descriptions:\n" + "\n".join(captions[:5])  # Include top captions for better context

        prompt = f"""
        You are an expert in social media content analysis and hashtag generation. Your task is to create highly relevant, creative, and descriptive hashtags for the following content. Focus on named entities, objects, themes, actions, moods, and unique visual or audio elements. Avoid generic or overused tags.

        {context}

        **INSTRUCTIONS:**
        - Carefully analyze the summary and visual descriptions to identify specific people, places, objects, activities, moods, and themes.
        - Use hashtags that are unique, memorable, and likely to attract engagement or search traffic.
        - Prefer multi-word hashtags that combine context, e.g., #SunsetYogaSession, #UrbanStreetArt, #GoldenRetrieverPuppy.
        - Include hashtags for any detected named entities (people, brands, locations), objects, colors, moods, and activities.
        - Avoid generic tags like #video, #photo, #nature, #fun, #love, #instagood, etc.
        - Mix broad and niche tags for better reach and discoverability.
        - Generate exactly 8-12 hashtags, each on a new line, starting with # and using CamelCase for readability.
        - Do not use spaces, punctuation, or numbers in hashtags unless part of a proper noun.
        - Do not repeat hashtags.
        - **IMPORTANT:** Your response must contain ONLY the list of hashtags, one per line. Do not include any other text, explanations, or numbering. Start immediately with the first #hashtag.

        **EXAMPLES (do not copy these; generate new ones based on the content):**
        #GoldenSunsetLandscape
        #HikingTrailAdventure
        #MajesticMountains
        #NaturePhotography
        #UrbanStreetArt
        #SunsetYogaSession
        #GoldenRetrieverPuppy
        #RainyCityEvening
        #MountainBikingJourney
        #ColorfulMarketScene
        """

        message = HumanMessage(content=[{"type": "text", "text": prompt}])
        response = _llm.invoke(
            [message],
            generation_config={
                "temperature": temperature,  # Use the passed temperature for creativity
                "max_output_tokens": 400,
                "top_p": 0.95,
                "top_k": 50
            }
        )
        raw_response = response.content.strip()
        if not raw_response:
            raw_response = "No response generated from AI model."
        # Extract hashtags more robustly
        hashtags = []
        for line in raw_response.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Clean up any extra spaces or punctuation
                tag = re.sub(r'[^a-zA-Z0-9#]', '', line).strip()
                if tag and len(tag) > 2:
                    hashtags.append(tag)
        hashtags = list(set(hashtags))  # Remove duplicates
        # If no hashtags, return empty list
        return hashtags[:12], raw_response
    except Exception as e:
        st.warning(f"⚠️ Hashtag generation failed: {e}")
        return [], f"Error occurred during generation: {str(e)}"

def display_keyframes(keyframes: List[Dict]):
    if not keyframes:
        return
    cols_per_row = min(5, len(keyframes))
    rows = [keyframes[i:i + cols_per_row] for i in range(0, len(keyframes), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for i, (col, frame) in enumerate(zip(cols, row)):
            with col:
                try:
                    img_data = base64.b64decode(frame['image_url']['url'].split(',')[1])
                    st.image(
                        img_data,
                        caption=f"Frame {frame.get('frame_number', i+1)}\n{frame.get('time_formatted', '0:00')}",
                        width=150
                    )
                except:
                    st.error(f"Frame {i+1}: Preview failed")

def create_export_data(summary: str, hashtags: List[str], captions: List[str], content_info: Dict, processing_stats: Dict, model_name: str) -> Dict:
    return {
        "analysis_results": {
            "summary": summary,
            "hashtags": hashtags,
            "captions": [{"frame": i+1, "caption": cap, "timestamp": f"{i*2:.1f}s"} for i, cap in enumerate(captions)]
        },
        "metadata": content_info,
        "processing_statistics": processing_stats,
        "export_info": {
            "generated_at": datetime.now().isoformat(),
            "version": "2.5",  # Updated version
            "model": model_name
        }
    }

def display_results(summary: str, hashtags: List[str], captions: List[str], content_info: Dict, processing_stats: Dict, transcript: str = "", debug_response: str = "", model_name: str = ""):
    processing_time = processing_stats.get("processing_time", 0)
    st.success(f"🎉 **Analysis Complete!** Processed in {processing_time:.1f} seconds")
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### 📄 AI-Generated Summary")
        st.markdown(f"**{summary}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🏷️ Smart Hashtags")
        if not hashtags:
            st.warning("⚠️ No hashtags generated. Check the raw response below, try increasing AI creativity, or re-analyzing.")
        else:
            hashtag_cols = st.columns(3)
            for i, hashtag in enumerate(hashtags):
                with hashtag_cols[i % 3]:
                    st.code(hashtag, language="text")
        with st.expander("🛠️ Debug: Raw Hashtag Response", expanded=True):  # Expanded by default for visibility
            if debug_response:
                st.text(debug_response)
            else:
                st.text("No raw response available. AI model did not return any content.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if transcript and len(transcript.strip()) > 20:
            with st.expander("📝 Full Transcript", expanded=False):
                st.text_area("Transcript Content", transcript, height=200, disabled=True)
    
    with col2:
        if captions:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Frame Captions & Hashtags")
            # Try to get batch hashtags from debug_response if available, else fallback to empty
            batch_hashtags = []
            if debug_response and isinstance(debug_response, dict) and 'batch_hashtags' in debug_response:
                batch_hashtags = debug_response['batch_hashtags']
            elif hasattr(display_results, 'batch_hashtags'):
                batch_hashtags = getattr(display_results, 'batch_hashtags')
            # If not available, fallback to empty lists
            if not batch_hashtags or len(batch_hashtags) != len(captions):
                batch_hashtags = [[] for _ in captions]
            for i, (caption, frame_tags) in enumerate(zip(captions[:10], batch_hashtags[:10])):
                st.markdown(f"**{i+1}.** {caption}")
                if frame_tags:
                    st.markdown(' '.join(frame_tags))
            if len(captions) > 10:
                with st.expander(f"Show {len(captions) - 10} more captions"):
                    for i, (caption, frame_tags) in enumerate(zip(captions[10:], batch_hashtags[10:]), 11):
                        st.markdown(f"**{i}.** {caption}")
                        if frame_tags:
                            st.markdown(' '.join(frame_tags))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Processing Statistics")
        stats_to_show = {
            "🎞️ Keyframes": processing_stats.get("keyframes_extracted", len(captions)),
            "⏱️ Processing Time": f"{processing_time:.1f}s",
            "🎙️ Transcript Words": processing_stats.get("transcript_length", 0),
            "📝 Captions Generated": len(captions)
        }
        for label, value in stats_to_show.items():
            if value:
                st.metric(label, value)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 💾 Export Results")
        export_data = create_export_data(summary, hashtags, captions, content_info, processing_stats, model_name)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"analysis_{int(time.time())}.json",
                mime="application/json"
            )
        with col2:
            text_export = f"SUMMARY:\n{summary}\n\nHASHTAGS:\n" + "\n".join(hashtags)
            if transcript:
                text_export += f"\n\nTRANSCRIPT:\n{transcript}"
            st.download_button(
                "📄 TXT", 
                data=text_export,
                file_name=f"summary_{int(time.time())}.txt",
                mime="text/plain"
            )

def download_youtube_video(url: str) -> str:
    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f"temp_youtube_{hashlib.md5(url.encode()).hexdigest()[:8]}.%(ext)s",
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            return video_path
    except Exception as e:
        st.error(f"❌ YouTube download failed: {e}")
        return None

def main():
    st.set_page_config(
        page_title="AI Video Summarizer Pro",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_css()
    
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI Video Summarizer Pro</h1>
        <p>Powered by Google Gemini • Local Videos, Images & YouTube</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🤖 Model Configuration")
        
        # Add model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["Google Gemini (Cloud)", "Custom Trained Model (Local)"],
            help="Choose between cloud-based Gemini or your local custom model"
        )
        
        if model_type == "Google Gemini (Cloud)":
            gemini_models = {
                "Gemini 2.5 Flash ⚡ (Recommended)": "gemini-2.5-flash",
                "Gemini 1.5 Pro 🧠": "gemini-1.5-pro", 
                "Gemini 1.0 Pro Vision 👁️": "gemini-pro-vision"
            }
            selected_model = st.selectbox("Select Gemini Model", list(gemini_models.keys()))
            model_name = gemini_models[selected_model]
            api_key = st.text_input("🔑 Google API Key", type="password", help="Get your free API key from Google AI Studio")
            
            if not api_key:
                st.warning("⚠️ Please enter your Google API Key")
                st.info("💡 **Get your key:** [Google AI Studio](https://makersuite.google.com/app/apikey)")
                st.stop()
            
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)
                st.success("✅ Connected to Gemini")
                custom_model = None
                device = None
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
                st.stop()
        
        else:  # Custom Trained Model
            st.markdown("### 🎯 Custom Model Settings")
            model_path = st.text_input(
                "🖼️ Path to Model File",
                value="/home/safda/WorkSpace/AI_hashtag_generator/captioning_model.pt",
                help="Enter the path to your .pt model file"
            )
            
            # Model configuration parameters
            with st.expander("⚙️ Model Configuration", expanded=False):
                st.info("Adjust these parameters to match your training configuration")
                st.success("✅ **Recommended settings from your training:**")
                st.success("• Image Size: 128 (128/8 = 16, 16² = 256 patches)")
                st.success("• Patch Size: 8")
                st.success("• Hidden Size: 192")
                st.success("• Layers: (6, 6)")
                st.success("• Heads: 8")
                col1, col2 = st.columns(2)
                with col1:
                    image_size = st.number_input("Image Size", min_value=64, max_value=512, value=128, step=32)
                    channels_in = st.number_input("Input Channels", min_value=1, max_value=4, value=3)
                    patch_size = st.number_input("Patch Size", min_value=4, max_value=32, value=8, step=4)
                with col2:
                    hidden_size = st.number_input("Hidden Size", min_value=64, max_value=512, value=192, step=32)
                    num_emb = st.number_input("Vocabulary Size", min_value=1000, max_value=100000, value=30522, step=1000)
                    num_layers_encoder = st.number_input("Encoder Layers", min_value=1, max_value=12, value=6)
                    num_layers_decoder = st.number_input("Decoder Layers", min_value=1, max_value=12, value=6)
                    num_heads = st.number_input("Number of Heads", min_value=1, max_value=16, value=8)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Inspect Model File"):
                    if model_path and Path(model_path).exists():
                        inspect_model_file(model_path)
                    else:
                        st.error(f"❌ Model file not found: {model_path}")
            
            with col2:
                if st.button("🔄 Load Custom Model"):
                    if model_path and Path(model_path).exists():
                        # Store configuration in session state
                        st.session_state.model_config = {
                            'image_size': image_size,
                            'channels_in': channels_in,
                            'num_emb': num_emb,
                            'patch_size': patch_size,
                            'hidden_size': hidden_size,
                            'num_layers': (num_layers_encoder, num_layers_decoder),
                            'num_heads': num_heads
                        }
                        
                        custom_model, device = load_custom_captioning_model(model_path)
                        if custom_model:
                            # Store model in session state to persist across refreshes
                            st.session_state.custom_model = custom_model
                            st.session_state.device = device
                            st.success(f"✅ Custom model loaded successfully on {device}")
                            llm = None  # Not needed for custom model
                        else:
                            st.error("❌ Failed to load custom model")
                            st.stop()
                    else:
                        st.error(f"❌ Model file not found: {model_path}")
                        st.stop()
            
            # Show model info if loaded (check session state first)
            if 'custom_model' in st.session_state and st.session_state.custom_model:
                custom_model = st.session_state.custom_model
                device = st.session_state.device
                st.success(f"✅ Custom model loaded on {device} (persisted in session)")
                llm = None
            elif 'custom_model' in locals() and custom_model:
                st.success(f"✅ Custom model loaded on {device}")
                llm = None
            else:
                st.warning("⚠️ Please load your custom model first")
                st.stop()
        
        st.markdown("---")
        st.markdown("### ⚙️ Processing Settings")
        
        with st.expander("🎛️ Advanced Settings", expanded=True):
            max_frames = st.slider("🎞️ Maximum Keyframes", 5, 50, 20)
            scene_threshold = st.slider("🎯 Scene Change Sensitivity", 0.60, 0.95, 0.80, 0.05)
            min_interval = st.slider("⏱️ Min Time Between Frames (seconds)", 1, 10, 3)
            batch_size = st.slider("📦 Batch Size", 1, 6, 4)
            temperature = st.slider("🌡️ AI Creativity", 0.0, 1.0, 0.3, 0.1)  # Increased max for more creativity
    
    st.markdown("### 📁 Input Selection")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 📷 Image Analysis")
        image_file = st.file_uploader("Upload Image", type=SUPPORTED_IMAGE_FORMATS)
        if image_file:
            file_size = len(image_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 Size: {file_size:.1f}MB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 🎥 Video Upload")
        video_file = st.file_uploader("Upload Video", type=SUPPORTED_VIDEO_FORMATS)
        if video_file:
            file_size = len(video_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 Size: {file_size:.1f}MB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("#### 📺 YouTube Video")
    youtube_url = st.text_input("Enter YouTube URL")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 **Start AI Analysis**", use_container_width=True, type="primary"):
        if not any([image_file, video_file, youtube_url]):
            st.error("❌ Please provide an input (image, video file, or YouTube URL)")
            st.stop()
        
        start_time = time.time()
        processing_stats = {"start_time": start_time}
        
        try:
            content_path_str = None
            content_info = {}
            
            if image_file:
                st.markdown("### 🖼️ Image Analysis")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image_file, caption=f"Analyzing: {image_file.name}", width=400)
                
                progress_bar = st.progress(0, "🔍 Analyzing image...")
                frame_data = load_image_for_llm(image_file)
                progress_bar.progress(0.3, "🤖 Generating caption...")
                
                if frame_data:
                    # Use appropriate caption generation based on model type
                    if model_type == "Custom Trained Model (Local)":
                        # Check session state for custom model
                        if 'custom_model' in st.session_state and st.session_state.custom_model:
                            custom_model = st.session_state.custom_model
                            device = st.session_state.device
                            captions, batch_hashtags = generate_batch_captions_custom_model(custom_model, device, frame_data, temperature)
                        else:
                            st.error("❌ Custom model not loaded. Please load your custom model first.")
                            st.stop()
                    else:
                        captions, batch_hashtags = generate_batch_captions(llm, frame_data, temperature)
                    
                    progress_bar.progress(0.6, "📝 Creating summary...")
                    
                    # For custom model, we need to handle summary generation differently
                    if model_type == "Custom Trained Model (Local)":
                        # Check session state for custom model
                        if 'custom_model' in st.session_state and st.session_state.custom_model:
                            # Use a simple summary based on captions since we don't have Gemini
                            summary = f"Image analysis completed using custom model. {len(captions)} caption(s) generated."
                        else:
                            st.error("❌ Custom model not loaded. Please load your custom model first.")
                            st.stop()
                    else:
                        summary = generate_comprehensive_summary(llm, captions, "", {"title": image_file.name})
                    
                    progress_bar.progress(0.9, "🏷️ Generating hashtags...")
                    
                    if model_type == "Custom Trained Model (Local)":
                        # Check session state for custom model
                        if 'custom_model' in st.session_state and st.session_state.custom_model:
                            # Use hashtags from captions only
                            final_hashtags = list(set(tag for tags in batch_hashtags for tag in tags))
                            debug_response = "Custom model analysis completed"
                        else:
                            st.error("❌ Custom model not loaded. Please load your custom model first.")
                            st.stop()
                    else:
                        hashtags, debug_response = generate_smart_hashtags(llm, summary, captions, {"title": image_file.name}, temperature)
                        # Combine all hashtags from captions and smart hashtags
                        all_caption_tags = set(tag for tags in batch_hashtags for tag in tags)
                        final_hashtags = list(set(hashtags) | all_caption_tags)
                    
                    progress_bar.progress(1.0, "✅ Analysis complete!")
                    
                    processing_stats.update({
                        "processing_time": time.time() - start_time,
                        "content_type": "image",
                        "captions_generated": len(captions)
                    })
                    
                    # Update model name for display
                    display_model_name = "Custom Trained Model" if model_type == "Custom Trained Model (Local)" else model_name
                    display_results(summary, final_hashtags, captions, {"title": image_file.name}, processing_stats, "", debug_response, display_model_name)
            
            elif video_file or youtube_url:
                st.markdown("### 🎬 Video Analysis")
                transcript = ""
                transcript_enabled = True
                video_duration = None
                # Add transcript option for long videos
                if video_file:
                    file_size_mb = len(video_file.getvalue()) / (1024 * 1024)
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        st.error(f"❌ File too large: {file_size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")
                        st.stop()
                    st.info(f"📁 Processing: **{video_file.name}** ({file_size_mb:.1f}MB)")
                    video_bytes = video_file.getvalue()
                    file_hash = hashlib.md5(video_bytes).hexdigest()[:8]
                    content_path_obj = Path(f"temp_video_{file_hash}.mp4")
                    content_path_obj.write_bytes(video_bytes)
                    content_path_str = str(content_path_obj)
                    content_info = {"title": video_file.name, "source": "local_upload"}
                    try:
                        with VideoFileClip(content_path_str) as clip:
                            video_duration = int(clip.duration)
                    except:
                        video_duration = None
                elif youtube_url:
                    st.info(f"📥 Downloading YouTube video: {youtube_url}")
                    content_path_str = download_youtube_video(youtube_url)
                    if not content_path_str:
                        st.stop()
                    content_path_obj = Path(content_path_str)
                    ydl_opts = {'quiet': True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(youtube_url, download=False)
                    content_info = {"title": info['title'], "source": "youtube", "duration": info.get('duration')}
                    video_duration = info.get('duration')
                # Show transcript option if video is long
                if video_duration and video_duration > 600:
                    transcript_enabled = st.checkbox(f"Generate transcript for long video ({video_duration//60}:{video_duration%60:02d})?", value=False)
                # Progress: transcript
                transcript_progress = st.progress(0, "Preparing transcript...")
                if content_path_str:
                    st.video(content_path_str)
                    if transcript_enabled:
                        transcript_progress.progress(0.2, "Transcribing audio...")
                        transcript = get_transcript_from_video(content_path_str)
                        transcript_progress.progress(1.0, "Transcript ready!")
                    else:
                        transcript_progress.progress(1.0, "Transcript skipped.")
                    # Progress: keyframe extraction
                    st.markdown("### 🎞️ Keyframe Extraction")
                    progress_container = st.empty()
                    def progress_callback(progress, frames_found, current_time):
                        progress_container.progress(
                            progress,
                            f"🎬 Extracted {frames_found}/{max_frames} keyframes @ {current_time:.1f}s"
                        )
                    keyframes = extract_keyframes(content_path_str, scene_threshold, min_interval, max_frames, progress_callback)
                    progress_container.progress(1.0, f"✅ Extracted {len(keyframes)} keyframes")
                    # Progress: caption generation
                    if keyframes:
                        st.markdown("### 🖼️ Extracted Keyframes")
                        display_keyframes(keyframes)
                        st.markdown("### 🎯 Caption Generation")
                        all_captions = []
                        caption_progress = st.progress(0, "🤖 Starting caption generation...")
                        all_batch_hashtags = []
                        for i in range(0, len(keyframes), batch_size):
                            batch = keyframes[i:i + batch_size]
                            
                            # Use appropriate caption generation based on model type
                            if model_type == "Custom Trained Model (Local)":
                                # Check session state for custom model
                                if 'custom_model' in st.session_state and st.session_state.custom_model:
                                    custom_model = st.session_state.custom_model
                                    device = st.session_state.device
                                    batch_captions, batch_hashtags = generate_batch_captions_custom_model(custom_model, device, batch, temperature)
                                else:
                                    st.error("❌ Custom model not loaded. Please load your custom model first.")
                                    st.stop()
                            else:
                                batch_captions, batch_hashtags = generate_batch_captions(llm, batch, temperature)
                            
                            all_captions.extend(batch_captions)
                            all_batch_hashtags.extend(batch_hashtags)
                            
                            progress = (i + batch_size) / len(keyframes)
                            caption_progress.progress(
                                min(progress, 1.0),
                                f"🤖 Processed {min(i + batch_size, len(keyframes))}/{len(keyframes)} frames"
                            )
                        
                        # Combine all hashtags from captions
                        all_caption_tags = set(tag for tags in all_batch_hashtags for tag in tags)
                        
                        # Handle summary generation based on model type
                        if model_type == "Custom Trained Model (Local)":
                            # Check session state for custom model
                            if 'custom_model' in st.session_state and st.session_state.custom_model:
                                summary = f"Video analysis completed using custom model. {len(all_captions)} captions generated from {len(keyframes)} keyframes."
                            else:
                                st.error("❌ Custom model not loaded. Please load your custom model first.")
                                st.stop()
                        else:
                            with st.spinner("📝 Creating comprehensive summary..."):
                                summary = generate_comprehensive_summary(llm, all_captions, transcript, content_info)
                        
                        # Handle hashtag generation based on model type
                        if model_type == "Custom Trained Model (Local)":
                            # Check session state for custom model
                            if 'custom_model' in st.session_state and st.session_state.custom_model:
                                final_hashtags = list(all_caption_tags)
                                debug_response = "Custom model analysis completed"
                            else:
                                st.error("❌ Custom model not loaded. Please load your custom model first.")
                                st.stop()
                        else:
                            with st.spinner("🏷️ Generating smart hashtags..."):
                                hashtags, debug_response = generate_smart_hashtags(llm, summary, all_captions, content_info, temperature)
                            final_hashtags = list(set(hashtags) | all_caption_tags)
                        processing_stats.update({
                            "processing_time": time.time() - start_time,
                            "content_type": "video",
                            "keyframes_extracted": len(keyframes),
                            "captions_generated": len(all_captions),
                            "transcript_length": len(transcript.split()) if transcript else 0
                        })
                        
                        # Update model name for display
                        display_model_name = "Custom Trained Model" if model_type == "Custom Trained Model (Local)" else model_name
                        display_results(summary, final_hashtags, all_captions, content_info, processing_stats, transcript, debug_response, display_model_name)
                    # Cleanup
                    try:
                        content_path_obj.unlink()
                    except:
                        pass
        
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
