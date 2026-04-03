"""
Memory Gallery Page - View all saved persons and conversations.
"""

import streamlit as st
import database
import os
from datetime import datetime


# Load custom CSS for Crystal Glass UI
def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles.css")
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Apply custom styling
load_css()

st.markdown("""
<div class="top-page-nav">
    <a href="/" target="_self" class="nav-pill">🏠 Home</a>
    <a href="/Memory_Gallery" target="_self" class="nav-pill active-nav">🖼️ Memory Gallery</a>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(0,22,54,0.08);
    border-radius: 14px 14px 0 0;
    padding: 0.6rem 1rem;
    color: #4B5563;
}

.stTabs [aria-selected="true"] {
    background: #001636 !important;
    color: white !important;
    border-color: #001636 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize database
database.init_db()

st.set_page_config(
    page_title="Memory Gallery - Memory Loop",
    page_icon="📚",
    layout="wide",
    menu_items={
        'About': "Memory Loop Gallery - View all saved persons and conversations."
    }
)

st.title("📚 Memory Gallery")
st.markdown("""
<div style="
    background: rgba(255, 230, 140, 0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 32px;
    padding: 3rem 2rem;
    margin: 1.5rem auto 2rem auto;
    max-width: 1100px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(0,0,0,0.08);
">
    <div style="
        position: absolute;
        inset: 0;
        background-image: url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23001636' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\");
        opacity: 0.6;
        pointer-events: none;
    "></div>
    <div style="position: relative; z-index: 1; max-width: 800px; margin: 0 auto; text-align: center;">
        <h1 style="
            font-size: 3rem;
            font-weight: 800;
            color: #001636;
            margin: 0 0 1rem 0;
            letter-spacing: -0.03em;
        ">
            Memory Gallery
        </h1>
        <p style="font-size: 1.1rem; color: #1A3A5C; margin: 0; line-height: 1.6;">
            Browse all saved persons and their conversation memories.
            Click on a person to view their recorded memories and stories.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation tabs
tab1, tab2 = st.tabs(["👥 People", "💬 All Conversations"])

# Get base directory for images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with tab1:
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.72);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0,22,54,0.08);
        border-radius: 24px;
        padding: 1.2rem 1.5rem
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <h2 style="font-size: 1.5rem; font-weight: 700; color: #001636; margin: 0;">
            👥 Saved People
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Get all persons with conversation count
    persons = database.get_all_persons_with_conversation_count()

    if not persons:
        st.info("No people saved yet. Start recording to add people to your memory gallery!")
    else:
        st.markdown(f"**Total: {len(persons)} people saved**")

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
                                col.image("MemoryLoop.png", width=150)
                        else:
                            col.image("MemoryLoop.png", width=150)

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
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.72);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0,22,54,0.08);
        border-radius: 24px;
        padding: 1.2rem 1.5rem
    ">
        <h2 style="font-size: 1.5rem; font-weight: 700; color: #001636; margin: 0;">
            💬 All Conversations
        </h2>
    </div>
    """, unsafe_allow_html=True)

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
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border-radius: 20px;
    padding: 1.5rem;
    border-left: 4px solid #FBBF24;
">
    <h2 style="font-size: 1.5rem; font-weight: 700; color: #001636; margin: 0 0 1rem 0;">
        📊 Statistics
    </h2>
</div>
""", unsafe_allow_html=True)

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