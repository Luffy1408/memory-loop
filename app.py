import streamlit as st
import cv2
import torch
from torch import hub
from time import time
import numpy as np
import os
from groq import Groq
import tempfile
import base64
import io
from PIL import Image
import face_recognition
import sqlite3
import pickle
from datetime import datetime
import edge_tts
import pygame
import threading
import asyncio


@st.cache
class ObjectDetection:
    def __init__(self, out_file="testing.avi"):
        self.out_file = out_file
        self.model = hub.load(
            'ultralytics/yolov5',
            'yolov5s',
            pretrained=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_video_stream(self):
        # change the number to 0 if you only have 1 camera
        stream = cv2.VideoCapture(1)  # 0 means read from the default camera, 1 the next camera, and so on...
        return stream

    """
    The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
    """

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    """
    The function below orchestrates the entire operation and performs the real-time parsing for video stream.
    """

    def __call__(self):
        player = self.get_video_stream()  # Get your video stream.
        assert player.isOpened()  # Make sure that there is a stream.
        # Below code creates a new video writer object to write our
        # output stream.
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")  # Using MJPEG codex
        out = cv2.VideoWriter(self.out_file, four_cc, 20,
                              (x_shape, y_shape))
        ret, frame = player.read()  # Read the first frame.
        frame_window = st.image([])
        while True:  # Run until stream is out of frames
            start_time = time()  # We would like to measure the FPS.
            ret, frame = player.read()
            assert ret
            results = self.score_frame(frame)  # Score the Frame
            frame = self.plot_boxes(results, frame)  # Plot the boxes.
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)  # Measure the FPS.
            print(f"Frames Per Second : {fps}")
            # cv2.imshow('frame', frame)
            frame_window.image(frame)
            out.write(frame)  # Write the frame onto the output.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # ret, frame = player.read()  # Read next frame.


###############
## Functions ##
###############

def transcribe_audio(audio_file, client):
    """Transcribe audio using Groq Whisper API"""
    try:
        with open(audio_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        return transcription.text
    except Exception as e:
        return f"Error transcribing: {str(e)}"


def translate_to_hindi(text, client):
    """Translate text to Hindi using Groq API"""
    try:
        from groq import Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Translate the following text to Hindi. Only return the Hindi translation, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return text  # Return original text on error


# Voice options for Edge-TTS (human-like voices)
TTS_VOICES = {
    'en': {
        'male': 'en-US-GuyNeural',      # American English male (natural)
        'female': 'en-US-JennyNeural',   # American English female (natural)
        'uk_male': 'en-GB-RyanNeural',   # British English male
        'uk_female': 'en-GB-SoniaNeural' # British English female
    },
    'hi': {
        'male': 'hi-IN-MadhurNeural',
        'female': 'hi-IN-SwaraNeural'
    }
}

# Default voice type (can be overridden by sidebar selection)
_default_voice_type = "female"

def get_voice_type():
    """Get the current voice type from session state or default."""
    if "voice_type_select" in st.session_state:
        voice_map = {
            "Female (Jenny)": "female",
            "Male (Guy)": "male",
            "UK Female": "uk_female",
            "UK Male": "uk_male"
        }
        return voice_map.get(st.session_state.voice_type_select, "female")
    return _default_voice_type


async def generate_speech_edge(text, output_path, voice='en-US-JennyNeural'):
    """Generate speech using Edge-TTS (human-like voice)."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"Edge-TTS error: {e}")
        return False


def text_to_speech(text, language='en', voice_type='female'):
    """
    Convert text to speech using Edge-TTS (human-like voice).
    Returns the audio file path if successful, None otherwise.
    """
    try:
        # Create temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_audio.close()

        # Get appropriate voice
        lang_code = 'hi' if language.lower() in ['hi', 'hindi'] else 'en'
        voice_key = voice_type if voice_type in TTS_VOICES.get(lang_code, {}) else 'female'
        voice = TTS_VOICES[lang_code][voice_key]

        # Run async TTS
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(generate_speech_edge(text, temp_audio.name, voice))
        loop.close()

        if success:
            return temp_audio.name
        return None
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None


def play_audio_async(audio_path):
    """Play audio file asynchronously using pygame (plays through system default audio)."""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
    except Exception as e:
        print(f"Audio playback error: {e}")
    finally:
        # Clean up temp file
        try:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
        except:
            pass


def play_memory_audio(text, language='en', voice_type='female'):
    """
    Convert text to speech and play automatically.
    Plays through system default audio device (Bluetooth speaker if selected).
    """
    if not text or not text.strip():
        return

    audio_path = text_to_speech(text, language, voice_type)
    if audio_path:
        # Play in background thread to not block UI
        audio_thread = threading.Thread(target=play_audio_async, args=(audio_path,))
        audio_thread.daemon = True
        audio_thread.start()

###############
## Dashboard ##
###############

st.set_page_config(
    page_title="Memory Loop - Roche Dementia Hackathon",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Memory Loop helps people with dementia remember daily objects and their loved ones. "
                 "Memory Loop captures moments with objects & people and stores the stories associated with them. "
                 "Whenever the person focuses on an object or person, the digital memory will start talking about it, "
                 "reminding the person of the history behind that object or person. Developed during the Roche"
                 " Dementia Hackathon Challenge by Team 4 (Women in AI and Robotics)."
    }
)
st.title("Memory Loop")
st.markdown("Welcome to **_Memory Loop_**, a virtual person and object recognition system that"
            " enables you to **attach memories** to any everyday item. Using AI and Augmented Reality, we help"
            " dementia & Alzheimer patients remember past experiences and sentiments.")

st.sidebar.image("MemoryLoop.png", width=300)
st.sidebar.title("Upload a new memory")

with st.sidebar.form(key='Form1'):
    uploaded_file = st.file_uploader("Choose an image")
    user_word = st.text_input("Enter a name", "e.g. Ada Lovelace")
    category = st.radio("Choose a category", ("Person", "Object", "Landscape"))
    description = st.text_area('Describe the memory', 'It was the best of times,'
                                                      ' the worst of times,'
                                                      ' the age of wisdom, the age of foolishness, ...')
    spec = st.checkbox('Mark as extremely special')
    submitted = st.form_submit_button(label='Submit memory ⚡')

# Person Management Section
st.sidebar.markdown("---")
st.sidebar.title("👥 Manage Persons")

# Initialize database
import database
database.init_db()

# Get all persons
all_persons = database.get_all_known_faces()

if all_persons:
    st.sidebar.markdown("### ✏️ Edit Person Names")

    for person in all_persons:
        with st.sidebar.container():
            st.markdown(f"**Person #{person['id']}**")

            # Editable name field with a more prominent input
            new_name = st.text_input(
                "Name:",
                value=person["name"],
                key=f"edit_name_{person['id']}",
                placeholder="Enter person's name..."
            )

            # Save button prominently placed
            if st.button("💾 Save Name", key=f"save_name_{person['id']}", type="primary"):
                if new_name.strip():
                    database.update_person_name(person["id"], new_name.strip())
                    st.session_state.known_faces = database.get_all_known_faces()
                    st.success(f"✅ Updated to: {new_name}")
                    st.rerun()
                else:
                    st.error("Name cannot be empty")

            # Show conversation count (get all conversations)
            all_conversations = database.get_conversations_for_person(person["id"], limit=100)
            st.caption(f"📝 {len(all_conversations)} conversation(s) recorded")

            # Play memory audio button (use most recent - first in list due to ORDER BY id DESC)
            if all_conversations:
                last_conv = all_conversations[0]
                if st.button(f"🔊 Play Last Memory", key=f"play_{person['id']}"):
                    # Use the potentially updated name from the text input
                    current_name = new_name if new_name.strip() else person["name"]
                    memory_text = f"{current_name}'s last memory: {last_conv['transcription']}"
                    play_memory_audio(memory_text, 'en', get_voice_type())
                    st.info("Playing audio...")

            # Delete person button
            if st.button("🗑️ Delete Person", key=f"delete_{person['id']}", type="secondary"):
                st.session_state[f"confirm_delete_{person['id']}"] = True

            # Confirm delete dialog
            if st.session_state.get(f"confirm_delete_{person['id']}", False):
                st.warning(f"⚠️ Delete {person['name']} and all their data?")
                col_del1, col_del2 = st.columns(2)
                with col_del1:
                    if st.button("✅ Yes, Delete", key=f"confirm_yes_{person['id']}"):
                        # Delete face image file if exists
                        if person.get("image_path") and os.path.exists(person["image_path"]):
                            try:
                                os.unlink(person["image_path"])
                            except:
                                pass
                        database.delete_known_face(person["id"])
                        st.session_state.known_faces = database.get_all_known_faces()
                        st.session_state[f"confirm_delete_{person['id']}"] = False
                        st.success(f"Deleted {person['name']}")
                        st.rerun()
                with col_del2:
                    if st.button("❌ Cancel", key=f"confirm_no_{person['id']}"):
                        st.session_state[f"confirm_delete_{person['id']}"] = False
                        st.rerun()

            st.sidebar.markdown("---")
else:
    st.sidebar.info("No persons saved yet. Record a video to add someone!")

# Medical History Section
st.sidebar.markdown("---")
st.sidebar.title("💊 Medical History")

# Select person for medical history
if all_persons:
    medical_person_options = {p["name"]: p["id"] for p in all_persons}
    selected_person_name = st.sidebar.selectbox(
        "Select Person",
        options=list(medical_person_options.keys()),
        key="medical_person_select"
    )
    selected_person_id = medical_person_options.get(selected_person_name)

    if selected_person_id:
        # Get current routines
        routines = database.get_medical_routines(selected_person_id)

        st.sidebar.markdown(f"**{len(routines)} medication(s) for {selected_person_name}**")

        # Display existing routines
        for routine in routines:
            with st.sidebar.expander(f"📅 {routine['medicine_name']} - {routine['time_of_day'] or 'Any time'}"):
                st.markdown(f"**Dosage:** {routine['dosage'] or 'Not specified'}")
                st.markdown(f"**Frequency:** {routine['frequency'] or 'Not specified'}")
                st.markdown(f"**Notes:** {routine['notes'] or 'None'}")

                # Edit routine
                if st.button("✏️ Edit", key=f"edit_routine_{routine['id']}"):
                    st.session_state[f"editing_routine_{routine['id']}"] = True

                if st.session_state.get(f"editing_routine_{routine['id']}", False):
                    edit_medicine = st.text_input("Medicine Name", value=routine['medicine_name'], key=f"edit_med_{routine['id']}")
                    edit_dosage = st.text_input("Dosage", value=routine['dosage'] or "", key=f"edit_dos_{routine['id']}")
                    edit_freq = st.text_input("Frequency", value=routine['frequency'] or "", key=f"edit_freq_{routine['id']}")
                    edit_time = st.text_input("Time of Day", value=routine['time_of_day'] or "", key=f"edit_time_{routine['id']}")
                    edit_notes = st.text_input("Notes", value=routine['notes'] or "", key=f"edit_notes_{routine['id']}")

                    col_save_edit, col_cancel_edit = st.columns(2)
                    with col_save_edit:
                        if st.button("💾 Save", key=f"save_edit_{routine['id']}"):
                            database.update_medical_routine(
                                routine['id'],
                                medicine_name=edit_medicine,
                                dosage=edit_dosage,
                                frequency=edit_freq,
                                time_of_day=edit_time,
                                notes=edit_notes
                            )
                            st.session_state[f"editing_routine_{routine['id']}"] = False
                            st.rerun()
                    with col_cancel_edit:
                        if st.button("Cancel", key=f"cancel_edit_{routine['id']}"):
                            st.session_state[f"editing_routine_{routine['id']}"] = False
                            st.rerun()

                # Delete routine
                if st.button("🗑️ Delete", key=f"del_routine_{routine['id']}"):
                    database.delete_medical_routine(routine['id'])
                    st.rerun()

        # Add new medication
        st.sidebar.markdown("### ➕ Add Medication")
        with st.sidebar.form(key=f"add_med_{selected_person_id}"):
            new_med_name = st.text_input("Medicine Name *", key=f"new_med_name_{selected_person_id}")
            new_med_dosage = st.text_input("Dosage (e.g., 1 pill, 5ml)", key=f"new_med_dosage_{selected_person_id}")
            new_med_freq = st.text_input("Frequency (e.g., Daily, Twice daily)", key=f"new_med_freq_{selected_person_id}")
            new_med_time = st.text_input("Time of Day (e.g., Morning, 8:00 AM)", key=f"new_med_time_{selected_person_id}")
            new_med_notes = st.text_input("Notes", key=f"new_med_notes_{selected_person_id}")

            if st.form_submit_button("Add Medication"):
                if new_med_name.strip():
                    database.add_medical_routine(
                        selected_person_id,
                        new_med_name.strip(),
                        new_med_dosage.strip(),
                        new_med_freq.strip(),
                        new_med_time.strip(),
                        new_med_notes.strip()
                    )
                    st.success(f"Added {new_med_name} for {selected_person_name}")
                    st.rerun()
                else:
                    st.error("Medicine name is required")
else:
    st.sidebar.info("Add persons first to manage medical history")

# Global Medical Schedule View
st.sidebar.markdown("---")
st.sidebar.title("📋 Today's Schedule")
if st.sidebar.button("View All Medications"):
    st.session_state["show_all_medications"] = not st.session_state.get("show_all_medications", False)

if st.session_state.get("show_all_medications", False):
    st.sidebar.markdown("### All Medications")
    all_routines = database.get_all_medical_routines()
    if all_routines:
        for routine in all_routines:
            st.sidebar.markdown(f"**{routine['person_name']}**")
            st.sidebar.markdown(f"- {routine['medicine_name']} ({routine['time_of_day'] or 'Any time'})")
            st.sidebar.markdown(f"  Dosage: {routine['dosage'] or 'N/A'}")
    else:
        st.sidebar.info("No medications scheduled")

# Voice settings
st.sidebar.markdown("---")
st.sidebar.title("🎤 Voice Settings")
voice_option = st.sidebar.selectbox("Voice Type", ["Female (Jenny)", "Male (Guy)", "UK Female", "UK Male"], key="voice_type_select")

# run = st.checkbox('Run')
run = st.selectbox("", ("Pick an AI model to start!", "Face & person recognition", "Object detection", "Record Live Memory", "Record Video with Live Subtitles"))

if submitted and run not in ["Face & person recognition", "Object detection"]:
    st.subheader('New memory unlocked!')
    st.image(uploaded_file)
    st.subheader(f'Meet {user_word}')
    if spec:
        st.markdown("_[Marked as extremely special]_")
    st.write(f'{description}')

# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(1)
# a = ObjectDetection()

elif run in ["Face & person recognition", "Object detection"]:

    st.sidebar.title("Adjust layout")
    width = st.sidebar.slider(
        label="Width", min_value=0, max_value=100, value=50, format="%d%%"
    )
    side = max((100 - width) / 2, 0.01)

    if run == "Face & person recognition":
        video_file = open('MemoryLoop.mp4', 'rb')
        audio_file = open("gaby_memoryloop.mp3", "rb")
        face = True
    else:
        video_file = open('MemoryLoop_Object.mp4', 'rb')
        audio_file = open("micro_memoryloop.mp3", "rb")
        face = False
    video_bytes = video_file.read()

    col1, container, col2 = st.columns([side, width, side], gap="medium")

    if submitted:
        col1.subheader('New memory unlocked! ✨')
        col1.image(uploaded_file)
        col1.subheader(f'Meet {user_word}')
        if spec:
            col1.markdown("_[Marked as extremely special]_")
        col1.write(f'{description}')

        with col1.expander("Check out the nerd stats!"):
            st.subheader("Dominant emotion detected:")
            st.metric(label="Happiness", value="95%", delta="+9%")

    container.video(video_bytes, start_time=0)

    col2.subheader("Special memory found! 🧠")
    col2.write("Click here to play audio description")
    audio_bytes = audio_file.read()
    col2.audio(audio_bytes, format="audio/ogg", start_time=0)

    col2.subheader("Related memories")

    if face:
        tab0, tab1, tab2, tab3 = col2.tabs(["Gaby", "Banu", "Queby", "Urska"])

        with tab0:
            st.image("memoryloop_pics/0.png")

        with tab1:
            st.image("memoryloop_pics/1.png")

        with tab2:
            st.image("memoryloop_pics/2.png")

        with tab3:
            st.image("memoryloop_pics/3.png")

    else:
        tab4, tab5, tab6, tab8 = col2.tabs(["Microwave", "Stove", "Fridge", "Spoon"])

        with tab4:
            st.image("memoryloop_pics/4.png")

        with tab5:
            st.image("memoryloop_pics/5.png")

        with tab6:
            st.image("memoryloop_pics/6.png")

        with tab8:
            st.image("memoryloop_pics/8.png")

elif run == "Record Live Memory":
    st.header("Record a Live Memory")
    st.markdown("Record a conversation or story and transcribe it using AI. Perfect for capturing memories with loved ones.")

    # API Key input
    api_key = st.text_input("Enter your Groq API key (get free at console.groq.com)", type="password", key="audio_api_key")
    if not api_key:
        st.warning("Please enter your Groq API key to enable transcription.")
        st.markdown("📝 **Get a free API key at [console.groq.com](https://console.groq.com)**")
    else:
        client = Groq(api_key=api_key)

        # Audio recording section
        st.subheader("Step 1: Record Your Memory")
        st.info("Click the microphone button below to start recording. Click again to stop.")

        audio_value = st.audio_input("Record your memory")

        if audio_value:
            st.success("Recording captured!")
            st.audio(audio_value)

            # Save to temp file for transcription
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_value.getvalue())
                temp_path = f.name

            st.subheader("Step 2: Transcribe")
            if st.button("Transcribe Recording", type="primary"):
                with st.spinner("Transcribing with Whisper AI..."):
                    transcription = transcribe_audio(temp_path, client)

                st.subheader("Transcription Result")
                st.write(transcription)

                # Save as memory section
                st.subheader("Step 3: Save as Memory")
                memory_name = st.text_input("Person's name", key="record_name")
                memory_category = st.radio("Category", ("Person", "Object", "Landscape"), key="record_category")

                if st.button("Save Memory", key="save_memory"):
                    st.success(f"Memory saved for {memory_name}!")
                    st.markdown(f"**Category:** {memory_category}")
                    st.markdown(f"**Transcription:** {transcription}")

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

    # Instructions
    with st.expander("How to get a free Groq API key"):
        st.markdown("""
        1. Go to [console.groq.com](https://console.groq.com)
        2. Sign up for a free account
        3. Navigate to API Keys section
        4. Create a new API key
        5. Copy and paste it above

        The free tier includes generous rate limits for Whisper transcription!
        """)

elif run == "Record Video with Live Subtitles":
    st.header("Record Video with Live Subtitles")
    st.markdown("Record video while speech is transcribed and displayed as subtitles in real-time. Perfect for capturing memories with context.")

    # Initialize database
    import database
    database.init_db()

    # Initialize session state for recording
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "subtitle_buffer" not in st.session_state:
        st.session_state.subtitle_buffer = ""
    if "full_transcription" not in st.session_state:
        st.session_state.full_transcription = []
    if "recorded_frames" not in st.session_state:
        st.session_state.recorded_frames = []
    if "subtitle_language" not in st.session_state:
        st.session_state.subtitle_language = "English"
    # Face recognition session state
    if "known_faces" not in st.session_state:
        st.session_state.known_faces = database.get_all_known_faces()
    if "detected_person" not in st.session_state:
        st.session_state.detected_person = None
    if "pending_face_data" not in st.session_state:
        st.session_state.pending_face_data = None
    if "show_save_dialog" not in st.session_state:
        st.session_state.show_save_dialog = False
    if "show_person_popup" not in st.session_state:
        st.session_state.show_person_popup = False
    if "popup_person_data" not in st.session_state:
        st.session_state.popup_person_data = None
    if "auto_save_on_stop" not in st.session_state:
        st.session_state.auto_save_on_stop = False
    if "frame_errors" not in st.session_state:
        st.session_state.frame_errors = 0
    # Persist face boxes between frames
    if "last_face_boxes" not in st.session_state:
        st.session_state.last_face_boxes = []  # List of (name, location, is_known, person_data)
    if "face_missing_frames" not in st.session_state:
        st.session_state.face_missing_frames = 0

    # API Key input
    api_key = st.text_input("Enter your Groq API key (get free at console.groq.com)", type="password", key="video_api_key")

    if not api_key:
        st.warning("Please enter your Groq API key to enable real-time transcription.")
        st.markdown("📝 **Get a free API key at [console.groq.com](https://console.groq.com)**")
    else:
        client = Groq(api_key=api_key)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Controls")

            # Language selector for subtitles
            st.subheader("Subtitle Language")
            subtitle_language = st.radio(
                "Choose language",
                ["English", "Hindi"],
                index=0 if st.session_state.subtitle_language == "English" else 1,
                key="subtitle_lang_radio",
                disabled=st.session_state.is_recording
            )
            st.session_state.subtitle_language = subtitle_language

            # Recording buttons - never disabled
            start_recording = st.button("🎬 Start Recording", type="primary")
            stop_recording = st.button("⏹️ Stop Recording")

            if start_recording and not st.session_state.is_recording:
                st.session_state.is_recording = True
                st.session_state.subtitle_buffer = ""
                st.session_state.full_transcription = []
                st.session_state.recorded_frames = []
                st.session_state.audio_chunks = []
                st.session_state.pending_face_data = None  # Reset pending face data
                st.session_state.detected_person = None  # Reset detected person
                st.session_state.frame_errors = 0  # Reset frame errors
                st.session_state.last_face_boxes = []  # Reset face boxes
                st.session_state.face_missing_frames = 0  # Reset missing frames counter
                st.success("Recording started! Speak clearly into your microphone.")

            if stop_recording and st.session_state.is_recording:
                st.session_state.is_recording = False

                # Save conversation to the detected person (known face) if available
                if st.session_state.get("detected_person") and st.session_state.detected_person:
                    # A known person was recognized - save conversation to their existing record
                    person_id = st.session_state.detected_person["id"]
                    person_name = st.session_state.detected_person["name"]

                    if st.session_state.full_transcription:
                        full_text = " ".join([t["text"] for t in st.session_state.full_transcription])
                        if full_text.strip():
                            database.save_conversation(person_id, full_text)
                            st.success(f"Conversation saved for {person_name}!")
                            st.info(f"Last conversation: '{full_text[:100]}...'")
                        else:
                            st.info(f"No conversation recorded for {person_name}.")
                    else:
                        st.info(f"No conversation recorded for {person_name}.")

                    # Update known faces in session state
                    st.session_state.known_faces = database.get_all_known_faces()
                    st.session_state.detected_person = None

                # If unknown face was detected, save as new person
                elif st.session_state.pending_face_data:
                    # Save face to database automatically
                    face_encoding = st.session_state.pending_face_data["face_encoding"]
                    face_frame = st.session_state.pending_face_data["face_frame"]

                    # Save face image
                    faces_dir = os.path.join(os.path.dirname(__file__), "faces")
                    os.makedirs(faces_dir, exist_ok=True)
                    timestamp = int(time())
                    face_image_path = os.path.join(faces_dir, f"Unknown_{timestamp}.jpg")
                    cv2.imwrite(face_image_path, face_frame)

                    # Insert into database with temporary name
                    person_id = database.save_known_face(f"Unknown_{timestamp}", face_encoding, face_image_path)

                    # Save last conversation if available
                    if st.session_state.full_transcription:
                        full_text = " ".join([t["text"] for t in st.session_state.full_transcription])
                        if full_text.strip():
                            database.save_conversation(person_id, full_text)
                            st.info(f"Face and conversation saved! Last conversation: '{full_text[:100]}...'")
                        else:
                            st.info("Face saved! No conversation was recorded.")
                    else:
                        st.info("Face saved! No conversation was recorded.")

                    # Update known faces in session state
                    st.session_state.known_faces = database.get_all_known_faces()
                    st.session_state.pending_face_data = None

                st.success("Recording stopped!")

        # Camera selection
        camera_index = st.sidebar.selectbox("Select Camera", [0, 1], index=0)

        if st.session_state.is_recording:
            st.subheader("Live Preview")
            # Create columns for video and save button
            col_video, col_save = st.columns([4, 1])
            with col_video:
                video_placeholder = st.empty()
                # Notification placeholder for person recognition popup
                notification_placeholder = st.empty()
            with col_save:
                st.write("")  # Spacer
                st.write("")  # Spacer
                # Show save button only when an unknown face is detected
                if st.session_state.pending_face_data:
                    if st.button("💾 Save Face", key="save_face_live"):
                        st.session_state.show_save_dialog = True
                    # Show preview of detected face
                    st.image(st.session_state.pending_face_data["face_frame"], caption="Detected Face", width=150)
                else:
                    st.info("No face detected")
            subtitle_placeholder = st.empty()
            status_placeholder = st.empty()

            # Open camera with retry
            cap = None
            max_retries = 3
            for retry in range(max_retries):
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    break
                st.warning(f"Camera {camera_index} not available, retrying... ({retry + 1}/{max_retries})")
                import time as time_module
                time_module.sleep(1)

            if not cap or not cap.isOpened():
                st.error(f"Cannot open camera {camera_index}. Please check if camera is connected and not in use by another application.")
                st.session_state.is_recording = False
                st.stop()

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Test camera read
            ret_test, frame_test = cap.read()
            if not ret_test:
                st.error("Camera opened but failed to read frames. Try closing other apps using the camera.")
                cap.release()
                st.session_state.is_recording = False
                st.stop()

            # Audio setup
            try:
                import pyaudio
                import wave
                PYAUDIO_AVAILABLE = True
            except ImportError:
                PYAUDIO_AVAILABLE = False
                st.error("PyAudio is not installed. Please install it with: `pip install pyaudio`")
                st.info("On Windows, you may need to run: `pip install pipwin && pipwin install pyaudio`")
                st.session_state.is_recording = False
                st.stop()  # Stop script execution

            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            RECORD_SECONDS = 2  # Capture 2 seconds at a time for transcription
            SILENCE_THRESHOLD = 30  # Threshold for detecting silence (lower = more sensitive)

            def is_silence(audio_data, threshold=SILENCE_THRESHOLD):
                """Check if audio chunk is silence by measuring RMS energy"""
                if audio_data.size == 0:
                    return True
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                return rms < threshold

            audio = pyaudio.PyAudio()
            audio_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

            recording_start = time()
            frame_count = 0
            face_frame_count = 0
            audio_buffer = []
            last_subtitle = ""
            silence_count = 0  # Track consecutive silence frames

            while st.session_state.is_recording:
                # Capture video frame
                ret, frame = cap.read()
                if not ret:
                    frame_errors = st.session_state.get("frame_errors", 0) + 1
                    st.session_state.frame_errors = frame_errors
                    if frame_errors > 10:
                        st.error("Camera stopped responding. Please restart the recording.")
                        break
                    continue  # Skip this frame and try again
                st.session_state.frame_errors = 0

                # Face detection every 10 frames for performance
                if frame_count % 10 == 0:
                    # Convert BGR to RGB for face_recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Find faces in the frame
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    # Reset face boxes list when faces are detected
                    current_face_boxes = []

                    # Check against known faces
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Compare with known faces
                        matches = []
                        names = []
                        for known_face in st.session_state.known_faces:
                            match = face_recognition.compare_faces([known_face["face_encoding"]], face_encoding, tolerance=0.6)
                            if match[0]:
                                matches.append(True)
                                names.append(known_face["name"])

                        if matches:
                            # Person recognized - show most recent match
                            person_name = names[-1]
                            person_data = next((f for f in st.session_state.known_faces if f["name"] == person_name), None)
                            if person_data:
                                st.session_state.detected_person = person_data
                                st.session_state.show_person_popup = True
                                st.session_state.popup_person_data = person_data

                                # Store face box for persistent display
                                current_face_boxes.append((person_name, (left, top, right, bottom), True, person_data))
                            else:
                                # Store face box for persistent display
                                current_face_boxes.append((person_name, (left, top, right, bottom), True, None))
                        else:
                            # Unknown face detected - store for potential saving
                            if st.session_state.pending_face_data is None:
                                st.session_state.pending_face_data = {
                                    "face_encoding": face_encoding,
                                    "face_frame": frame[top:bottom, left:right].copy(),
                                    "location": (top, right, bottom, left)
                                }
                            # Store face box for persistent display
                            current_face_boxes.append(("Unknown", (left, top, right, bottom), False, None))

                    # Update persistent face boxes
                    if current_face_boxes:
                        st.session_state.last_face_boxes = current_face_boxes
                        st.session_state.face_missing_frames = 0
                    else:
                        # No faces detected in this frame
                        st.session_state.face_missing_frames += 1
                        # Only clear boxes after 15 frames (~1.5 seconds) of no faces
                        if st.session_state.face_missing_frames > 15:
                            st.session_state.last_face_boxes = []
                            st.session_state.detected_person = None

                # Draw face boxes on every frame (persistent display)
                for face_name, (left, top, right, bottom), is_known, person_data in st.session_state.last_face_boxes:
                    if is_known:
                        # Draw green box for known face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, f"KNOWN: {face_name}", (left, top - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        # Add welcome message
                        cv2.putText(frame, f"Welcome back, {face_name}!", (10, frame.shape[0] - 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        # Draw blue box for unknown face
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.putText(frame, "Unknown", (left, top - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Capture audio chunk
                audio_data = np.frombuffer(audio_stream.read(CHUNK * RECORD_SECONDS, exception_on_overflow=False), dtype=np.int16)
                audio_buffer.extend(audio_data.tolist())

                # Check if we have enough audio data for transcription (~2 seconds)
                if len(audio_buffer) >= RATE * RECORD_SECONDS:
                    # Convert audio buffer to wav format for Groq API
                    audio_array = np.array(audio_buffer[:RATE * RECORD_SECONDS], dtype=np.int16)

                    # Check for silence before transcribing
                    if not is_silence(audio_array):
                        silence_count = 0  # Reset silence counter
                        # Create temporary wav file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                            wf = wave.open(temp_audio.name, 'wb')
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(audio.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(audio_array.tobytes())
                            wf.close()

                            # Transcribe audio chunk using Whisper Large V3 (better accuracy)
                            try:
                                with open(temp_audio.name, "rb") as f:
                                    transcription = client.audio.transcriptions.create(
                                        file=(temp_audio.name, f.read()),
                                        model="whisper-large-v3",  # Better model for accuracy
                                        response_format="verbose_json",
                                        language="en",  # Force English transcription
                                    )

                                if transcription.text.strip():
                                    original_text = transcription.text.strip()
                                    # Translate to Hindi if selected
                                    if st.session_state.subtitle_language == "Hindi":
                                        translated_text = translate_to_hindi(original_text, client)
                                        st.session_state.subtitle_buffer = translated_text
                                        last_subtitle = translated_text
                                        st.session_state.full_transcription.append({
                                            "timestamp": time() - recording_start,
                                            "text": translated_text,
                                            "original": original_text
                                        })
                                    else:
                                        st.session_state.subtitle_buffer = original_text
                                        last_subtitle = original_text
                                        st.session_state.full_transcription.append({
                                            "timestamp": time() - recording_start,
                                            "text": original_text
                                        })
                                # Keep last subtitle if no speech detected in this chunk
                            except Exception as e:
                                print(f"Transcription error: {e}")
                                pass  # Silently handle transcription errors

                            # Clean up temp file
                            try:
                                os.unlink(temp_audio.name)
                            except:
                                pass
                    else:
                        # Silence detected, increment counter but keep subtitle for a few frames
                        silence_count += 1
                        # Only clear subtitle after 3 consecutive silence frames (6 seconds)
                        if silence_count >= 3:
                            st.session_state.subtitle_buffer = ""

                    # Remove processed audio from buffer
                    audio_buffer = audio_buffer[RATE * RECORD_SECONDS:]

                # Draw subtitle on frame
                if st.session_state.subtitle_buffer:
                    # Create semi-transparent background for subtitle
                    overlay = frame.copy()
                    subtitle_height = 80
                    y_pos = frame.shape[0] - subtitle_height - 10

                    # Draw background rectangle
                    cv2.rectangle(overlay, (10, y_pos - 50), (frame.shape[1] - 10, y_pos + 20), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                    # Wrap text to fit width
                    max_chars_per_line = frame.shape[1] // 12
                    words = st.session_state.subtitle_buffer.split()
                    lines = []
                    current_line = ""

                    for word in words:
                        if len(current_line) + len(word) < max_chars_per_line:
                            current_line += " " + word if current_line else word
                        else:
                            lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)

                    # Draw subtitle text
                    for i, line in enumerate(lines[-2:]):  # Show last 2 lines
                        text_y = y_pos + (i * 25)
                        cv2.putText(frame, line.strip(), (20, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Add recording indicator
                elapsed = int(time() - recording_start)
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {elapsed}s", (55, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Convert frame with subtitles to RGB for display
                frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display frame
                video_placeholder.image(frame_display, width=640)

                # Show notification popup when a known person is detected
                if st.session_state.show_person_popup and st.session_state.popup_person_data:
                    person = st.session_state.popup_person_data
                    last_conversation = database.get_last_conversation_for_person(person["id"])

                    notification_placeholder.success(f"**Person Recognized: {person['name']}**")

                    with notification_placeholder.expander("View Last Conversation", expanded=True):
                        if last_conversation:
                            st.write(f"_Last conversation: {last_conversation['transcription']}_")
                            st.caption(f"Recorded on: {last_conversation['recorded_at']}")
                        else:
                            st.write("No previous conversations recorded.")
                else:
                    notification_placeholder.empty()

                # Store frame for later video creation
                st.session_state.recorded_frames.append(frame.copy())

                frame_count += 1

            # Release camera and audio
            cap.release()
            audio_stream.stop_stream()
            audio_stream.close()
            audio.terminate()

            # Create video from recorded frames
            if st.session_state.recorded_frames:
                st.subheader("Recording Complete!")

                # Create output video
                output_path = tempfile.mktemp(suffix=".avi")
                height, width = st.session_state.recorded_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

                for frame in st.session_state.recorded_frames:
                    out.write(frame)
                out.release()

                st.video(output_path)

                # Display full transcription
                if st.session_state.full_transcription:
                    st.subheader("Full Transcription")
                    full_text = " ".join([t["text"] for t in st.session_state.full_transcription])
                    st.write(full_text)

                    # Auto-play transcription as audio
                    st.subheader("🔊 Audio Playback")
                    st.info("Playing memory audio through system speaker (Bluetooth if connected)...")

                    # Determine language from transcription
                    audio_lang = 'en'
                    if any(t.get("original") for t in st.session_state.full_transcription):
                        audio_lang = st.session_state.get("subtitle_language", "English").lower()
                        if audio_lang == "hindi":
                            audio_lang = 'hi'

                    # Generate audio file
                    audio_file_path = text_to_speech(full_text, audio_lang)

                    if audio_file_path:
                        # Play audio automatically
                        play_memory_audio(full_text, audio_lang)

                        # Also provide audio player for manual replay
                        st.audio(audio_file_path, format="audio/mp3")

                        # Replay button
                        col_replay1, col_replay2 = st.columns(2)
                        with col_replay1:
                            if st.button("🔊 Replay Audio", key="replay_audio"):
                                play_memory_audio(full_text, audio_lang)
                                st.success("Playing audio...")
                        with col_replay2:
                            # Language selection for replay
                            replay_lang = st.selectbox("Language", ["English", "Hindi"], key="replay_lang")

                    # Save as memory option
                    st.subheader("Save as Memory")
                    col_save1, col_save2 = st.columns(2)
                    with col_save1:
                        memory_name = st.text_input("Name", key="video_memory_name")
                    with col_save2:
                        memory_category = st.radio("Category", ["Person", "Object", "Landscape"], key="video_memory_category")

                    if st.button("💾 Save Video Memory"):
                        # Save video file to memories folder
                        memories_dir = os.path.join(os.path.dirname(__file__), "memories")
                        os.makedirs(memories_dir, exist_ok=True)

                        timestamp = int(time())
                        video_filename = f"{memory_name.replace(' ', '_')}_{timestamp}.avi"
                        video_save_path = os.path.join(memories_dir, video_filename)

                        # Copy video to memories folder
                        import shutil
                        shutil.copy(output_path, video_save_path)

                        # Save transcription as text file
                        transcription_path = os.path.join(memories_dir, f"{memory_name.replace(' ', '_')}_{timestamp}.txt")
                        with open(transcription_path, "w") as f:
                            f.write(full_text)

                        # Save audio file
                        audio_save_path = os.path.join(memories_dir, f"{memory_name.replace(' ', '_')}_{timestamp}.mp3")
                        if audio_file_path and os.path.exists(audio_file_path):
                            shutil.copy(audio_file_path, audio_save_path)

                        st.success(f"Memory saved! Video: {video_filename}")
                        st.info(f"Transcription saved to: {transcription_path}")
                        if audio_file_path:
                            st.info(f"Audio saved to: {audio_save_path}")

                        # Play confirmation audio
                        confirmation_text = f"Memory saved for {memory_name}. {full_text}"
                        play_memory_audio(confirmation_text, audio_lang)

                # Clean up temp file
                try:
                    os.unlink(output_path)
                except:
                    pass

                # Reset recording state
                st.session_state.is_recording = False
                st.session_state.recorded_frames = []

        # Handle person popup when known face is detected
        if st.session_state.show_person_popup and st.session_state.popup_person_data:
            person = st.session_state.popup_person_data

            # Get last conversation for this person
            last_conversation = database.get_last_conversation_for_person(person["id"])

            # Auto-play welcome message and last memory
            welcome_text = f"Welcome back, {person['name']}!"
            if last_conversation:
                memory_text = f"Your last memory: {last_conversation['transcription']}"
                full_speech = f"{welcome_text}. {memory_text}"
            else:
                full_speech = welcome_text

            # Play TTS automatically when person is recognized
            play_memory_audio(full_speech, 'en')

            # Define dialog function
            @st.dialog(f"Person Recognized: {person['name']}")
            def show_person_dialog():
                st.write(f"**Welcome back, {person['name']}!**")

                if person.get("image_path"):
                    try:
                        st.image(person["image_path"], width=200)
                    except:
                        pass

                if last_conversation:
                    st.subheader("Last Conversation")
                    st.write(last_conversation["transcription"])
                    st.caption(f"Recorded on: {last_conversation['recorded_at']}")

                    # Button to replay memory audio
                    if st.button("🔊 Replay Memory Audio", key="replay_person_memory"):
                        memory_audio_text = f"{person['name']}'s memory: {last_conversation['transcription']}"
                        play_memory_audio(memory_audio_text, 'en')
                        st.success("Playing memory audio...")
                else:
                    st.info("No previous conversations recorded.")

                if st.button("Close", key="close_popup"):
                    st.session_state.show_person_popup = False
                    st.session_state.popup_person_data = None
                    st.rerun()

            show_person_dialog()

        # Handle save face data dialog
        if st.session_state.show_save_dialog and st.session_state.pending_face_data:
            @st.dialog("Save Face Data")
            def show_save_dialog():
                st.write("Save this person's face data and link it to the last recorded conversation.")

                # Show captured face
                face_frame = st.session_state.pending_face_data["face_frame"]
                st.image(face_frame, caption="Captured Face", width=150)

                # Show preview of conversation to be saved
                if st.session_state.full_transcription:
                    full_text = " ".join([t["text"] for t in st.session_state.full_transcription])
                    if full_text.strip():
                        st.subheader("Conversation to Save")
                        st.info(full_text)
                    else:
                        st.warning("No conversation recorded. Only face data will be saved.")
                else:
                    st.warning("No conversation recorded. Only face data will be saved.")

                person_name = st.text_input("Person's Name", key="save_face_name")

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button("Save", type="primary", key="confirm_save_face"):
                        if person_name.strip():
                            # Save face to database
                            face_encoding = st.session_state.pending_face_data["face_encoding"]

                            # Save face image
                            faces_dir = os.path.join(os.path.dirname(__file__), "faces")
                            os.makedirs(faces_dir, exist_ok=True)
                            timestamp = int(time())
                            face_image_path = os.path.join(faces_dir, f"{person_name.replace(' ', '_')}_{timestamp}.jpg")
                            cv2.imwrite(face_image_path, face_frame)

                            # Insert into database
                            person_id = database.save_known_face(person_name, face_encoding, face_image_path)

                            # Link last conversation if available
                            if st.session_state.full_transcription:
                                full_text = " ".join([t["text"] for t in st.session_state.full_transcription])
                                if full_text.strip():
                                    database.save_conversation(person_id, full_text)

                            # Update session state
                            st.session_state.known_faces = database.get_all_known_faces()
                            st.session_state.pending_face_data = None
                            st.session_state.show_save_dialog = False

                            st.success(f"Face data saved for {person_name}!")
                            st.rerun()
                        else:
                            st.error("Please enter a name.")

                with col_d2:
                    if st.button("Cancel", key="cancel_save_face"):
                        st.session_state.pending_face_data = None
                        st.session_state.show_save_dialog = False
                        st.rerun()

            show_save_dialog()

    # Instructions
    with st.expander("How to use Video Recording with Live Subtitles"):
        st.markdown("""
        1. Enter your Groq API key above
        2. Select your camera from the sidebar
        3. Click 'Start Recording'
        4. Speak clearly - your words will appear as subtitles
        5. Click 'Stop Recording' when done
        6. Review your video and transcription
        7. Optionally save as a memory for later viewing
        """)
