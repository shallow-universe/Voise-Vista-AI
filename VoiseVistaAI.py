#python -m streamlit run VoiseVistaAI.py

import os
import json
import whisper
import moviepy.editor as mp
from moviepy.editor import concatenate_audioclips, AudioFileClip, VideoFileClip
import streamlit as st
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import librosa
import numpy as np


load_dotenv()

# Azure     
sub_key = os.getenv('AZURE_SPEECH_KEY')
az_oai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
az_oai_api_key =  os.getenv('AZURE_OPENAI_API_KEY') 
az_oai_version =  os.getenv('AZURE_OPENAI_API_VERSION')
az_oai_model = 'gpt-4o'

# Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=az_oai_endpoint,
    api_key=az_oai_api_key,
    api_version=az_oai_version
)


# Define directory paths
VIDEO_DIR = "videos"
EXTRACTED_AUDIO_DIR = "Process/extracted_audio_file"
SEGMENT_AUDIO_DIR = "Process/segment_audios_file"
TRANSCRIPTS_DIR = "Process/transcripts_file"
OUTPUT_VIDEO_DIR = "New_translated_videos"

# Ensure all necessary directories exist
def create_directories():
    directories = [VIDEO_DIR, EXTRACTED_AUDIO_DIR, SEGMENT_AUDIO_DIR, TRANSCRIPTS_DIR, OUTPUT_VIDEO_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# Step 1: Video Input (Extract audio from video)
def extract_audio_from_video(video_path):
    print("\n\nRunning Step 1: Extract Audio from Video\n")
    
    try:
        video_clip = mp.VideoFileClip(video_path)
        
        # Construct audio path
        video_filename = os.path.basename(video_path)
        audio_filename = os.path.splitext(video_filename)[0] + ".mp3"
        audio_path = os.path.join(EXTRACTED_AUDIO_DIR, audio_filename)
        
        # Extract audio
        video_clip.audio.write_audiofile(audio_path, codec='mp3')
        print(f"Extracted audio saved to {audio_path}")
        
        video_clip.close()
        return audio_path
    
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

# Step 2: Speech-to-Text Conversion (with timestamps)
def transcribe_audio_with_timestamps(audio_path):
    print("\n\nRunning Step 2: Transcribe Audio\n")

    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)
        
        # Load audio file for gender detection
        y, sr = librosa.load(audio_path)
        
        transcript = []
        for segment in result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            
            # Extract segment audio for gender detection
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            # Detect gender with confidence
            gender, confidence = detect_gender(segment_audio, sr)  
            print(f"Detected gender: {gender} (Confidence: {confidence:.2f})")
            print("------------------------")
            
            timestamped_text = {
                "start": start_time, 
                "end": end_time, 
                "text": text,
                "gender": gender,
                "gender_confidence": float(confidence)  # Convert to float for JSON serialization
            }
            
            transcript.append(timestamped_text)
        
        # Save transcript
        transcript_path = os.path.join(TRANSCRIPTS_DIR, "Original_transcript.json")
        with open(transcript_path, "w", encoding='utf-8') as f:
            json.dump(transcript, f, indent=4, ensure_ascii=False)
        
        return transcript
    
    except Exception as e:
        print(f"Error in transcription: {e}")
        raise


# Add this new function for gender detection
def detect_gender(audio_segment, sample_rate):
    try:
        # 1. Calculate MFCCs (for vocal tract characteristics)
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=13)
        first_mfcc = np.mean(mfccs[1]) # First MFCC after the energy coefficient
        
        # 2. Calculate weighted pitch using fundamental frequency
        pitches, magnitudes = librosa.piptrack(
            y=audio_segment, 
            sr=sample_rate,
            fmin=50,     # Lowered to catch deeper male voices
            fmax=300     # Reduced to avoid noise
        )
        
        # Get weighted pitch for voiced segments
        voiced_pitches = pitches[magnitudes > np.max(magnitudes) * 0.1]
        weighted_pitch = np.mean(voiced_pitches) if len(voiced_pitches) > 0 else 0
        
        # 3. Calculate mean formant frequency
        S = np.abs(librosa.stft(y=audio_segment))
        formants = np.mean(librosa.feature.mfcc(S=librosa.power_to_db(S), sr=sample_rate))
        
        # 4. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)
        mean_centroid = np.mean(spectral_centroid)
        
        male_score = 0  
        female_score = 0
        
        # Print feature analysis for debugging
        print("\nFeature Analysis:")
        print(f"First MFcc: {first_mfcc:.2f}")
        print(f"Weighted pitch: {weighted_pitch:.2f} Hz")
        print(f"Mean Formant: {formants:.2f} Hz")
        print(f"Mean Spectral Centroid: {mean_centroid:.2f}")
        
        # Enhanced scoring system
        if weighted_pitch > 0:  # Only score if we detected pitch
            # Pitch-based scoring (primary feature)
            if weighted_pitch < 140:  # Clear male range
                male_score += 1
            elif weighted_pitch < 165:  # Possible male
                male_score += 1
            elif weighted_pitch > 175:  # Female range
                female_score += 1
            
            # MFCC scoring (vocal tract characteristics)
            if first_mfcc < 0:  # Male characteristic
                male_score += 1
            elif first_mfcc > 10:  # Female characteristic
                female_score += 1
            
        
        print(f"Male Score: {male_score}, Female Score: {female_score}")
        
        # Calculate confidence
        total_score = male_score + female_score
        confidence = (abs(male_score - female_score) / total_score) if total_score > 0 else 0
        
        # Decision with strong male bias
        if male_score > female_score:
            return "male", min(confidence + 0.1, 1.0)  # Add slight confidence boost for male
        else:
            # Require stronger evidence for female classification
            if female_score > male_score:
                return "female", confidence
            else:
                return "male", 0.6  # Default to male in ambiguous cases
                
    except Exception as e:
        print(f"Error in gender detection: {e}")
        return "unknown", 0.0


# Step 3: Detect language and translate the transcript
def detect_and_translate(transcript, target_language="hi"):
    print("\n\nRunning Step 3: Translation\n")

    translated_segments = []
    system_command = f"""You are a highly skilled translator. Translate the following text to {target_language} while maintaining context and gender-specific language. 
    The text is spoken by a specific gender (male or female) speaker. Ensure the translated text maintains appropriate gender-specific pronouns and forms where applicable. 
    Ensure the translated text can be spoken within a similar time frame as the original. Only give translated text as output"""

    for segment in transcript:
        original_text = segment["text"]
        audio_duration = segment["end"] - segment["start"]
        gender = segment["gender"]

        prompt = f"Original text: {original_text}\nTranslate to: {target_language}\nSpeaker gender: {gender}\nAudio duration: {audio_duration} seconds, only give translated text in output"

        try:
            response = client.chat.completions.create(
                model=az_oai_model,
                messages=[
                    {"role": "system", "content": system_command},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            translated_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            translated_text = original_text  # Fallback to original text

        segment["translated_text"] = translated_text
        segment["detected_language"] = target_language
        translated_segments.append(segment)

    # Save translated transcript
    translated_transcript_path = os.path.join(TRANSCRIPTS_DIR, "translated_transcript.json")
    with open(translated_transcript_path, "w", encoding='utf-8') as f:
        json.dump(translated_segments, f, indent=4, ensure_ascii=False)

    return translated_segments


#Step 4:
def convert_text_to_speech(translated_transcript, language_code="hi"):
    print("\n\nRunning Step 4: Text to Speech\n")

    # Azure Speech configuration
    subscription_key = sub_key
    region = "eastus"
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    
    # Define voice mapping for different languages
    voice_mapping = {
        "hi": {
            "male": "hi-IN-MadhurNeural",
            "female": "hi-IN-SwaraNeural"
        },
        "es": {
            "male": "es-ES-AlvaroNeural",
            "female": "es-ES-ElviraNeural"
        },
        "fr": {
            "male": "fr-FR-HenriNeural",
            "female": "fr-FR-DeniseNeural"
        },
        "de": {
            "male": "de-DE-ConradNeural",
            "female": "de-DE-KatjaNeural"
        },
        "zh-CN": {
            "male": "zh-CN-YunxiNeural",
            "female": "zh-CN-XiaoxiaoNeural"
        },
        "ar": {
            "male": "ar-SA-HamedNeural",
            "female": "ar-SA-ZariyahNeural"
        },
        "ru": {
            "male": "ru-RU-DmitryNeural",
            "female": "ru-RU-SvetlanaNeural"
        }
    }

    # Set default voices if language not found
    default_voices = {
        "male": "en-US-GuyNeural",
        "female": "en-US-JennyNeural"
    }
    
    for idx, segment in enumerate(translated_transcript):
        try:
            text = segment['translated_text']
            if not text or text.isspace():
                print(f"Empty text for segment {idx}, skipping...")
                segment['audio_file'] = None
                continue

            audio_filename = f"segment_{idx}.wav"
            audio_path = os.path.join(SEGMENT_AUDIO_DIR, audio_filename)
            gender = segment["gender"].lower()

            # Select appropriate voice based on language and gender
            if language_code in voice_mapping:
                if gender in ["male", "female"]:
                    voice_name = voice_mapping[language_code][gender]
                else:
                    voice_name = voice_mapping[language_code]["female"]  # default to female
            else:
                voice_name = default_voices.get(gender, default_voices["female"])

            speech_config.speech_synthesis_voice_name = voice_name

            # Configure audio output
            audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)
            
            # Set the output format to WAV
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
            )

            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )

            # Synthesize speech
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Verify the file was created and is valid
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    try:
                        # Verify audio file is readable
                        with AudioFileClip(audio_path) as test_audio:
                            if test_audio.duration > 0:
                                print(f"Successfully synthesized speech for segment {idx}")
                                segment['audio_file'] = audio_path
                            else:
                                raise Exception("Generated audio has zero duration")
                    except Exception as audio_error:
                        print(f"Error validating audio file for segment {idx}: {audio_error}")
                        segment['audio_file'] = None
                else:
                    print(f"Generated audio file is invalid or empty for segment {idx}")
                    segment['audio_file'] = None
            else:
                print(f"Speech synthesis failed for segment {idx}: {result.reason}")
                if result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = speechsdk.CancellationDetails(result)
                    print(f"Speech synthesis canceled: {cancellation_details.reason}")
                    print(f"Error details: {cancellation_details.error_details}")
                segment['audio_file'] = None

        except Exception as e:
            print(f"Error processing segment {idx}: {str(e)}")
            segment['audio_file'] = None
            continue

    return translated_transcript



# Step 5: Replace Audio in Video
def replace_audio_in_video(video_path, translated_audio_segments):
    print("\n\nRunning Step 5: Replace Audio\n")

    try:
        # Load original video
        video_clip = VideoFileClip(video_path)
        
        # Construct output video path
        video_filename = os.path.basename(video_path)
        output_video_filename = f"New_translated_{video_filename}"
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, output_video_filename)

        # Collect valid audio clips
        audio_clips = []
        for segment in translated_audio_segments:
            audio_file = segment.get('audio_file')
            
            if audio_file and os.path.exists(audio_file):
                try:
                    # Verify audio file is valid before adding
                    temp_audio = AudioFileClip(audio_file)
                    if temp_audio.duration > 0:  # Basic validation
                        start_time = segment['start']
                        audio_clips.append(temp_audio.set_start(start_time))
                        print(f"Successfully added audio clip: {audio_file}")
                    else:
                        print(f"Invalid audio duration for {audio_file}")
                except Exception as e:
                    print(f"Error processing audio clip {audio_file}: {str(e)}")
                    continue

        # Prepare final audio
        if audio_clips:
            try:
                final_audio = concatenate_audioclips(audio_clips)
                final_video = video_clip.set_audio(final_audio)
            except Exception as e:
                print(f"Error concatenating audio clips: {e}")
                final_video = video_clip
        else:
            print("No valid audio clips found. Using original video.")
            final_video = video_clip

        # Write final video
        final_video.write_videofile(
            output_video_path,
            codec="libx264",
            audio_codec="aac",
            fps=video_clip.fps,
            audio_fps=44100
        )

        # Close clips
        video_clip.close()
        if 'final_video' in locals():
            final_video.close()

        print(f"Translated video saved: {output_video_path}")
        return output_video_path

    except Exception as e:
        print(f"Fatal error in video processing: {e}")
        return video_path


# Add this new function to handle transcript editing
def display_editable_transcript(transcript):
    """Display and allow editing of transcript segments"""
    edited_transcript = []
    
    st.markdown("### Edit Transcript")
    st.markdown("Review and edit the transcript below before translation. You can modify the text and gender for each segment.")
    
    for idx, segment in enumerate(transcript):
        with st.expander(f"Segment {idx+1} - {int(segment['start']//60)}:{int(segment['start']%60):02d}", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Make text editable
                edited_text = st.text_area(
                    "Text",
                    value=segment['text'],
                    key=f"text_{idx}",
                    height=100
                )
                
            with col2:
                # Add gender selection dropdown
                edited_gender = st.selectbox(
                    "Gender",
                    options=["male", "female", "unknown"],
                    index=["male", "female", "unknown"].index(segment.get('gender', 'unknown')),
                    key=f"gender_{idx}"
                )
                
                # Display timing information
                st.text(f"Start: {int(segment['start']//60)}:{int(segment['start']%60):02d}")
                st.text(f"End: {int(segment['end']//60)}:{int(segment['end']%60):02d}")
            
            # Create updated segment with edited values
            edited_segment = segment.copy()
            edited_segment['text'] = edited_text
            edited_segment['gender'] = edited_gender
            edited_transcript.append(edited_segment)
    
    # Add a confirm edit button
    if st.button("Confirm Transcript Edits", use_container_width=True):
        return edited_transcript
    return None

# Update the main function
def main():
    st.set_page_config(layout="wide")
    
    # Custom CSS for better spacing and layout
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 20px;
        }
        .stVideo {
            width: 100%;
        }
        .transcript-container {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .stExpander {
            border: none;
            box-shadow: none;
        }
        .streamlit-expanderHeader {
            font-size: 1em;
            color: #0066cc;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Voise Vista.ai")
    st.write("##### an AI Video Translation App")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state if not exists
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'edited_transcript' not in st.session_state:
        st.session_state.edited_transcript = None
    if 'translated_transcript' not in st.session_state:
        st.session_state.translated_transcript = None
    
    # Create directories and setup language options
    create_directories()
    LANGUAGES = {
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese (Simplified)": "zh-CN",
        "Arabic": "ar",
        "Russian": "ru"
    }

    # Video upload and language selection
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Upload a Video File", 
            type=["mp4", "mkv", "mov", "avi"],
            key="video_uploader"
        )
        selected_language = st.selectbox(
            "Select Target Language", 
            list(LANGUAGES.keys()),
            index=0,
            key="language_selector"
        )
        target_language = LANGUAGES[selected_language]

    if uploaded_file:
        # Save uploaded file
        video_path = os.path.join(VIDEO_DIR, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display original video
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("### Original Video")
            st.video(video_path)

        # Generate Transcript button
        if st.button("Generate Transcript", use_container_width=True):
            try:
                with st.spinner("Generating transcript..."):
                    audio_path = extract_audio_from_video(video_path)
                    st.session_state.transcript = transcribe_audio_with_timestamps(audio_path)
                st.session_state.current_step = 'edit_transcript'
                st.rerun()

            except Exception as e:
                st.error(f"Failed to generate transcript: {e}")

        # Show editable transcript if available
        if st.session_state.current_step == 'edit_transcript' and st.session_state.transcript:
            edited = display_editable_transcript(st.session_state.transcript)
            if edited:
                st.session_state.edited_transcript = edited
                st.session_state.current_step = 'translate'
                st.rerun()

        # Process translation if transcript is edited
        if st.session_state.current_step == 'translate' and st.session_state.edited_transcript:
            if st.button("Start Translation", use_container_width=True):
                try:
                    with st.spinner("Processing translation..."):
                        translated_transcript = detect_and_translate(
                            st.session_state.edited_transcript, 
                            target_language
                        )
                        audio_translations = convert_text_to_speech(
                            translated_transcript, 
                            target_language
                        )
                        translated_video_path = replace_audio_in_video(
                            video_path, 
                            audio_translations
                        )

                    # Display results
                    with right_col:
                        st.markdown("### Translated Video")
                        st.video(translated_video_path)
                        
                        with open(translated_video_path, "rb") as file:
                            st.download_button(
                                label="Download Translated Video",
                                data=file,
                                file_name=f"translated_{os.path.basename(video_path)}",
                                mime="video/mp4",
                                use_container_width=True
                            )

                    # Display transcripts
                    st.markdown("### Final Transcripts")
                    transcript_col1, transcript_col2 = st.columns(2)
                    
                    with transcript_col1:
                        st.markdown("#### Original Transcript")
                        for segment in translated_transcript:
                            start_time = f"{int(segment['start']//60)}:{int(segment['start']%60):02d}"
                            st.markdown(f"""
                            <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;'>
                                <p style='color: #666; margin-bottom: 5px;'>⏱️ {start_time}</p>
                                <p style='margin: 0;'>{segment['text']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with transcript_col2:
                        st.markdown("#### Translated Transcript")
                        for segment in translated_transcript:
                            start_time = f"{int(segment['start']//60)}:{int(segment['start']%60):02d}"
                            st.markdown(f"""
                            <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;'>
                                <p style='color: #666; margin-bottom: 5px;'>⏱️ {start_time}</p>
                                <p style='margin: 0;'>{segment['translated_text']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Translation failed: {e}")

if __name__ == "__main__":
    main()
