# Voise Vista.ai 🎙️🎬

**Break the Language Barrier with AI-Powered Video Translation**

Voise Vista.ai is an innovative application that leverages the power of Artificial Intelligence to translate video content seamlessly.  Upload your video, and let Voise Vista.ai handle the rest - from transcribing the speech to generating natural-sounding translations in your target language. 🌎🗣️

## Features ✨

- **Effortless Video Translation:** Translate your videos into multiple languages effortlessly, expanding your audience reach.
- **Accurate Speech Recognition:** Advanced speech-to-text technology ensures accurate transcription of your video's audio.
- **Natural-Sounding Voiceovers:** AI-powered text-to-speech generates voiceovers in the target language, preserving the natural flow and intonation of speech.
- **Gender-Specific Translations:** Voise Vista.ai intelligently detects speaker gender and adjusts translations to maintain appropriate pronouns and linguistic nuances. 
- **Simple and Intuitive Interface:** The user-friendly Streamlit interface makes video translation a breeze.

## How It Works ⚙️

1. **Upload Your Video:** Choose the video you want to translate. Voise Vista.ai supports various popular video formats. 
2. **Generate Transcript:** Click "Generate Transcript" and our AI engine will extract the audio, transcribe it, and detect the speaker's gender.
3. **Review and Edit (Optional):** Fine-tune the generated transcript for maximum accuracy.
4. **Select Target Language:**  Choose the language you want to translate your video into. 
5. **Translate:**  Voise Vista.ai performs the translation, taking into account context and speaker gender for natural-sounding results.
6. **Synthesize Voiceover:**  The translated text is converted into high-quality speech using Azure Cognitive Services, selecting an appropriate voice based on gender.
7. **Replace Original Audio:**  The original audio track is seamlessly replaced with the new translated voiceover, maintaining perfect synchronization.
8. **Download Your Translated Video:** Download your translated video, ready to share with a wider audience!

## Installation and Setup 🔧

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Anant2003jain/Voise-Vista.ai.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Voise-Vista.ai
   ```

3. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

4. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variables:**
   - Create a `.env` file in the root directory.
   - Add the following environment variables with your Azure credentials:
     ```
     AZURE_SPEECH_KEY=your_azure_speech_key
     AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
     AZURE_OPENAI_API_KEY=your_azure_openai_api_key
     AZURE_OPENAI_API_VERSION=your_azure_openai_api_version 
     ```

6. **Run the app:**

   ```bash
   python -m streamlit run VoiseVistaAI.py
   ```

## Screenshots 📸
1. **Home Page**

![HomePage](https://github.com/user-attachments/assets/6d82c11e-36e6-4dec-a2b8-cc789c375186)

2. **Select Target Language and Generate Transcript**

![SelectVideo_Lang_Generate](https://github.com/user-attachments/assets/71fb76c2-c706-4b23-a118-8eba4a43b6a9)

3. **Edit Text and Gender (Optional)**

![Edit_SaveTranscript](https://github.com/user-attachments/assets/87a03e00-76ea-4f34-86a0-8ac937a11b3b)

4. **Start Translation**
   - This Process might take some time

![StartTranslation](https://github.com/user-attachments/assets/f2934b1b-b5df-468f-ae7f-3617747a9af7)

5. **View and Download Translated Video**

![OutputVideo](https://github.com/user-attachments/assets/6ca2d51f-9164-4801-8e2f-58788535896b)

6. **Review Transcripts**

![Original_translatedTranscript](https://github.com/user-attachments/assets/9a7bc1db-4765-4a42-80b6-3bf3eb5202ba)



## Technologies Used 💻

- **Python:** The core programming language for the app.
- **Streamlit:** Framework for building the interactive web application.
- **Whisper:**  State-of-the-art speech-to-text model.
- **Azure OpenAI:**  Powerful language model (GPT-4) used for high-quality translations.
- **Azure Cognitive Services:**  Provides text-to-speech capabilities for natural-sounding voiceovers.
- **Librosa:** Python library for audio analysis, used for gender detection. 

## Contributing 🤝

We encourage contributions to Voise Vista.ai!  To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License 📄

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.

## Connect with Me  
Feel free to star the repository and share your feedback!

For updates, insights, or to connect, feel free to check out my [LinkedIn profile](https://www.linkedin.com/in/anant-jain-1720671a7).
