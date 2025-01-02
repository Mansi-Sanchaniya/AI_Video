import streamlit as st
import yt_dlp
from pytube import Playlist
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import io
import os
import cv2
from yt_dlp import YoutubeDL


# Function to download a YouTube video using yt-dlp and a cookie file
def download_video(url):
    download_status = ""  # Initialize download_status to avoid referencing undefined variable
    downloaded_video_path = None

    # Define the directory to store the video
    temp_dir = "temp_videos"  # Name of the temporary directory

    # Create the directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Create the directory

    # Set up yt-dlp options, saving video to the temp directory
    ydl_opts = {
        'outtmpl': os.path.join('%(title)s.%(ext)s'),  # Save video to the temp directory
    }

    # Use yt-dlp to download the video
    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            download_status = "Video downloaded successfully!"  # Set the success status
            # Prepare the full path of the downloaded video (fixing path issue)
            downloaded_video_path = os.path.join(temp_dir, f"{ydl.prepare_filename(ydl.extract_info(url, download=False))}")
            st.success(download_status)
        except Exception as e:
            download_status = f"Error downloading video: {str(e)}"  # Set the error message
            st.error(download_status)  # Display error message

    print(downloaded_video_path)
    return download_status, downloaded_video_path  # Return the status for further checking if needed


# Function to get video URLs from multiple playlists or individual video links
def get_video_urls_multiple(input_urls):
    video_urls = []
    urls = input_urls.split(",")  # Split input by comma
    for url in urls:
        url = url.strip()  # Remove any leading/trailing spaces
        if "playlist" in url:
            playlist = Playlist(url)
            video_urls.extend(playlist.video_urls)  # Add all video URLs in the playlist
        else:
            video_urls.append(url)  # Treat as a single video URL
    return video_urls


# Function to get transcript for a video using its YouTube ID
def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        # Fetch the transcript (if available)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        return None


# Function to check if a video is under Creative Commons license using YouTube Data API and description

# Function to format the transcript into a readable form
def format_transcript(transcript):
    formatted_transcript = []
    for entry in transcript:
        start_time = entry['start']  # Timestamp
        duration = entry['duration']
        text = entry['text']  # Transcript text
        formatted_transcript.append(f"[{start_time}s - {start_time + duration}s] {text}")
    return formatted_transcript


# Function to process input (multiple playlists or individual videos) and fetch transcripts for all videos
def process_input(input_urls):
    video_urls = get_video_urls_multiple(input_urls)
    if not video_urls:
        return []

    all_transcripts = []  # List to hold all transcripts

    video_chunks = {}  # Dictionary to store video-specific transcripts

    # Use another ThreadPoolExecutor to fetch transcripts concurrently
    with ThreadPoolExecutor() as transcript_executor:
        future_to_video = {transcript_executor.submit(get_transcript, video_url): video_url for video_url in video_urls}
        for idx, future in enumerate(as_completed(future_to_video)):
            video_url = future_to_video[future]
            try:
                transcript = future.result()
                if transcript:
                    formatted_transcript = format_transcript(transcript)
                    video_chunks[video_url] = formatted_transcript  # Store by video URL
                else:
                    video_chunks[video_url] = ["Transcript not available"]
            except Exception as e:
                video_chunks[video_url] = ["Transcript extraction failed"]
                print(f"Error getting transcript for {video_url}: {e}")

    # Reassemble the output in the original order of video URLs
    for video_url in video_urls:
        all_transcripts.append(
            {"video_url": video_url, "transcript": video_chunks.get(video_url, ["No transcript found"])})
    return all_transcripts

# Function to process the query and extract relevant transcript segments
def process_query(query, stored_transcripts, threshold=0.3):  # Adjusted threshold for more precise results
    if not query:
        st.warning("Please enter a query to search in the transcripts.")
        return []

    if not stored_transcripts:
        st.warning("No transcripts available. Please process a playlist or video first.")
        return []

    all_transcripts_text = []
    for video in stored_transcripts:
        video_info = f"Video: {video['video_url']}\n"
        if isinstance(video['transcript'], list):
            for line in video['transcript']:
                all_transcripts_text.append(video_info + line)

    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = all_transcripts_text
    query_vector = vectorizer.fit_transform([query])
    text_vectors = vectorizer.transform(corpus)

    # Calculate cosine similarity using sklearn's cosine_similarity (which works directly with sparse matrices)
    cosine_similarities = cosine_similarity(query_vector, text_vectors)

    # Now, cosine_similarities will be a 2D numpy array where we can access the first row (the result for the query)
    relevant_sections = []
    for idx, score in enumerate(cosine_similarities[0]):
        if score > threshold:  # Only include sections that pass the similarity threshold
            relevant_sections.append(corpus[idx])

    return relevant_sections


# Simulating your process functions for this demonstration
def process_transcripts(input_urls, progress_bar, status_text):
    total_steps = 100  # Example total steps for the process
    start_time = time.time()  # Track the start time
    for step in range(total_steps):
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        time_remaining = elapsed_time / (step + 1) * (total_steps - step - 1)  # Estimate remaining time
        time_remaining_str = f"{time_remaining:.2f} seconds remaining"  # Format remaining time

        time.sleep(0.1)  # Simulate a task
        progress_bar.progress(step + 1, text=f"Extracting transcripts: {step + 1}% done")
        status_text.text(time_remaining_str)  # Update the remaining time text

    return "Transcripts Extracted!"  # Once complete


def clip_and_merge_videos(segments, video_path, output_path):
    temp_clips = []

    for start, end, _ in segments:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = f"temp_clip_{len(temp_clips)}.mp4"

        out = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            if start <= current_time <= end:
                if out is None:
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))
                out.write(frame)

            frame_idx += 1
            if current_time > end:
                break

        cap.release()
        if out:
            out.release()
            temp_clips.append(temp_output)

    # Merge all clips manually without ffmpeg
    if temp_clips:
        # Open the first clip to get properties
        cap = cv2.VideoCapture(temp_clips[0])
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Write the merged video file
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Write all the frames from the temporary clips to the output
        for clip in temp_clips:
            cap = cv2.VideoCapture(clip)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()

        out.release()

        # Remove temporary clips
        for clip in temp_clips:
            os.remove(clip)

    return output_path

def main():
    st.set_page_config(page_title="Video & Playlist Processor", page_icon="🎬", layout="wide")

    st.markdown("""
    <style>
        .css-1d391kg {padding: 30px;}
        .stTextArea>div>div>textarea {
            font-size: 14px;
            line-height: 1.8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #ff5c5c;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #ff7d7d;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Video and Playlist Processor")

    input_urls = st.text_input("Enter YouTube Playlist(s) or Video URL(s) or both (comma-separated):")

    if 'stored_transcripts' not in st.session_state:
        st.session_state.stored_transcripts = []
    if 'transcript_text' not in st.session_state:
        st.session_state.transcript_text = ""

    if input_urls:
        # Create columns for button and progress bar side by side
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("Extract Transcripts"):
                progress_bar = col2.progress(0, text="Starting transcript extraction Please Hold...")
                status_text = col2.empty()  # Placeholder for dynamic status updates

                st.session_state.stored_transcripts = process_input(input_urls)
                progress_bar.progress(50, text="Processing transcripts...")
                status_text.text("Processing transcripts...")
                progress_bar.progress(100, text="Transcripts extracted successfully.")
                status_text.text("Transcripts extracted successfully.")
                if st.session_state.stored_transcripts:
                    transcript_text = ""
                    for video in st.session_state.stored_transcripts:
                        transcript_text += f"\nTranscript for video {video['video_url']}:\n"
                        if isinstance(video['transcript'], list):
                            for line in video['transcript']:
                                transcript_text += line + "\n"
                        else:
                            transcript_text += video['transcript'] + "\n"
                        transcript_text += "-" * 50 + "\n"
                    st.session_state.transcript_text = transcript_text

    if st.session_state.transcript_text:
        st.subheader("Extracted Transcripts")
        st.text_area("Transcripts", st.session_state.transcript_text, height=300, key="transcripts_area")

    query = st.text_input("Enter your query to extract relevant information:")
    if query:
        # Create columns for button and progress bar side by side
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("Process Query"):
                progress_bar = col2.progress(0, text="Starting query processing...")
                status_text = col2.empty()

                relevant_sections = process_query(query, st.session_state.stored_transcripts)
                progress_bar.progress(50, text="Analyzing query...")
                status_text.text("Analyzing query...")
                progress_bar.progress(100, text="Query processed successfully.")
                status_text.text("Query processed successfully.")
                if relevant_sections:
                    st.session_state.query_output = "\n".join(relevant_sections)
                else:
                    st.session_state.query_output = "No relevant content found for the query."

    if 'query_output' in st.session_state and st.session_state.query_output:
        st.subheader("Relevant Output for Your Query")
        st.text_area("Query Output", st.session_state.query_output, height=300, key="query_output_area")

    if input_urls and query:
        with col1:
            if st.button("Download Video(s)"):
                progress_bar = col2.progress(0, text="Starting video download. Please hold...")
                status_text = col2.empty()  # Placeholder for dynamic status updates

                for url in input_urls.split(","):
                    url = url.strip()
                    status_text.text(f"Downloading video from {url}...")
                    download_status, downloaded_video_path = download_video(url)
                    progress_bar.progress(100)
                    status_text.text(download_status)
                    if "successfully" in download_status:
                        st.success(f"Downloaded: {url}")
                        if downloaded_video_path:
                            st.video(downloaded_video_path)
                    else:
                        st.error(f"Failed to download: {url}")

if st.button("Combine and Play"):
        if 'query_output' in st.session_state and st.session_state.query_output:
            for url in input_urls.split(","):
                    url = url.strip()
                    downloaded_video_path = download_video(url)
            output_video_path = "output_video.mp4"
            final_path = clip_and_merge_videos(st.session_state.query_output,downloaded_video_path, output_video_path)
            st.video(final_path)
        else:
            st.error("No segments to combine. Process a query first.")


if __name__ == "__main__":
    main()
