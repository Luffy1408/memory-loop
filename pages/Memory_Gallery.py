"""
Memory Gallery Page - View all saved persons and conversations.
"""

import streamlit as st
import database
import os
from datetime import datetime

# Initialize database
database.init_db()

st.set_page_config(
    page_title="Memory Gallery - DigiMemoir",
    page_icon="📚",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/nathanyaqueby/roche-dementia-hackathon',
        'Report a bug': "https://github.com/nathanyaqueby/roche-dementia-hackathon",
        'About': "DigiMemoir Memory Gallery - View all saved persons and conversations."
    }
)

st.title("📚 Memory Gallery")
st.markdown("Browse all saved persons and their conversation memories.")

# Navigation tabs
tab1, tab2 = st.tabs(["👥 People", "💬 All Conversations"])

# Get base directory for images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with tab1:
    st.header("Saved People")

    # Get all persons with conversation count
    persons = database.get_all_persons_with_conversation_count()

    if not persons:
        st.info("No people saved yet. Start recording to add people to your memory gallery!")
    else:
        st.write(f"**Total: {len(persons)} people saved**")

        # Display persons in a grid
        cols_per_row = 3
        for i in range(0, len(persons), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, col in enumerate(cols):
                if i + j < len(persons):
                    person = persons[i + j]

                    with col.container(border=True):
                        # Person image
                        if person["image_path"] and os.path.exists(person["image_path"]):
                            try:
                                col.image(person["image_path"], width=150)
                            except:
                                col.image("DigiMemoir.png", width=150)
                        else:
                            col.image("DigiMemoir.png", width=150)

                        # Person info
                        col.subheader(f"👤 {person['name']}")

                        # Created date
                        created_at = person.get('created_at', 'Unknown')
                        if created_at and created_at != 'Unknown':
                            try:
                                dt = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                                col.caption(f"Added: {dt.strftime('%b %d, %Y')}")
                            except:
                                col.caption(f"Added: {created_at}")

                        # Conversation count badge
                        count = person.get('conversation_count', 0)
                        if count > 0:
                            col.markdown(f"💬 **{count} conversation{'s' if count != 1 else ''}**")
                        else:
                            col.markdown("_No conversations yet_")

                        # View conversations button
                        if col.button(f"View Memories", key=f"view_{person['id']}"):
                            st.session_state.selected_person = person['id']
                            st.rerun()

        # Show selected person's conversations
        if "selected_person" in st.session_state:
            st.divider()
            selected_id = st.session_state.selected_person

            # Find person name
            person_name = next((p['name'] for p in persons if p['id'] == selected_id), "Unknown")

            col_back, col_title = st.columns([1, 4])
            with col_back:
                if st.button("← Back to All"):
                    del st.session_state.selected_person
                    st.rerun()
            with col_title:
                st.header(f"💬 {person_name}'s Conversations")

            conversations = database.get_conversations_for_person(selected_id, limit=100)

            if not conversations:
                st.info(f"No conversations recorded for {person_name} yet.")
            else:
                for conv in conversations:
                    with st.container(border=True):
                        # Timestamp
                        recorded_at = conv.get('recorded_at', 'Unknown')
                        if recorded_at and recorded_at != 'Unknown':
                            try:
                                dt = datetime.strptime(recorded_at, '%Y-%m-%d %H:%M:%S')
                                st.caption(f"📅 {dt.strftime('%B %d, %Y at %I:%M %p')}")
                            except:
                                st.caption(f"📅 {recorded_at}")

                        # Transcription
                        st.write(conv['transcription'])

                        # Video link if available
                        if conv.get('video_path') and os.path.exists(conv['video_path']):
                            st.video(conv['video_path'])

with tab2:
    st.header("All Conversations")

    # Get all conversations with person data
    conversations = database.get_all_conversations_with_persons()

    if not conversations:
        st.info("No conversations recorded yet. Start recording to capture memories!")
    else:
        st.write(f"**Total: {len(conversations)} conversations**")

        # Filter by person
        person_names = ["All"] + list(set(c['person_name'] for c in conversations))
        filter_person = st.selectbox("Filter by person", person_names)

        # Apply filter
        if filter_person != "All":
            filtered = [c for c in conversations if c['person_name'] == filter_person]
        else:
            filtered = conversations

        st.write(f"Showing {len(filtered)} conversation{'s' if len(filtered) != 1 else ''}")

        # Display conversations
        for conv in filtered:
            with st.container(border=True):
                # Header with person name and date
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**👤 {conv['person_name']}**")

                with col2:
                    recorded_at = conv.get('recorded_at', 'Unknown')
                    if recorded_at and recorded_at != 'Unknown':
                        try:
                            dt = datetime.strptime(recorded_at, '%Y-%m-%d %H:%M:%S')
                            st.caption(f"📅 {dt.strftime('%b %d, %Y')}")
                        except:
                            st.caption(f"📅 {recorded_at}")

                # Conversation text
                st.write(conv['transcription'])

                # Video if available
                if conv.get('video_path') and os.path.exists(conv['video_path']):
                    st.video(conv['video_path'])

# Statistics at the bottom
st.divider()
st.header("📊 Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    persons_count = len(database.get_all_known_faces())
    st.metric(label="Total People", value=persons_count)

with col2:
    conversations_count = len(database.get_all_conversations())
    st.metric(label="Total Conversations", value=conversations_count)

with col3:
    # Calculate total words
    all_convs = database.get_all_conversations()
    total_words = sum(len(c['transcription'].split()) for c in all_convs if c['transcription'])
    st.metric(label="Total Words", value=total_words)