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
import re
import cv2
from yt_dlp import YoutubeDL
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    st.text("Entered processs query function")
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


def extract_timestamps_from_section(section):
    try:
        # Strip any leading/trailing whitespaces
        section = section.strip()

        # Check if the section contains timestamp information in the correct format
        if '[' not in section or ']' not in section:
            return None  # Skip sections that do not contain timestamps in '[start_time - end_time]' format

        # Extract the timestamp part of the section (the part inside the brackets)
        timestamp_part = section[section.find('[') + 1:section.find(']')].strip()  # Extract content inside brackets
        times = timestamp_part.split(" - ")

        # Ensure two timestamps are found in the section
        if len(times) != 2:
            return None  # Return None to skip this section

        # Clean timestamps and remove any unnecessary decimal precision
        start_time = float(times[0].strip().replace("s", ""))
        end_time = float(times[1].strip().replace("s", ""))

        # Round to a reasonable precision (e.g., 2 decimal places)
        start_time = round(start_time, 2)
        end_time = round(end_time, 2)

        return start_time, end_time
    except Exception as e:
        print(f"Error extracting timestamps from section '{section}'. Exception: {e}")
        return None  # Return None in case of an error

def extract_video_segments(input_string):
    # This pattern looks for YouTube URLs with timestamps in the format of [start_time - end_time]
    pattern = r"(https://www\.youtube\.com/watch\?v=[\w-]+)\s*\[([\d\.]+s)\s*-\s*([\d\.]+s)\]"
    
    # Find all matching segments
    matches = re.findall(pattern, input_string)
    
    video_segments = []
    
    # For each match, process the video URL and timestamps
    for match in matches:
        url, start, end = match
        start_time = float(start[:-1])  # Remove the 's' and convert to float
        end_time = float(end[:-1])      # Remove the 's' and convert to float
        video_segments.append((url, start_time, end_time))
    
    return video_segments

def clip_and_merge_videos(segments, video_path, output_filename):
    st.text("Entered clip_and_merge_videos function")
    temp_dir = "temp_videos"  # Name of the temporary directory
    total_duration = 0
    
    # Create the directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Create the directory
        st.text(f"Created temporary directory: {temp_dir}")

    # Full output path for the final video
    output_path = os.path.join(temp_dir, output_filename)
    temp_clips = []
    st.write(f"Type of st.session_state.query_output: {type(segments)}")
    st.write(f"Value of st.session_state.query_output: {segments}")

    for segment in segments:
        st.text("Entered loop")
        # Extract the video URL, start and end times
        url, start_time, end_time = segment
        st.text(f"URL: {url}, Start: {start_time}s, End: {end_time}s")

        if not video_path or not isinstance(video_path, str):
            raise ValueError("Invalid video path provided")

        # Ensure the file exists before proceeding
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        # Open the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Define the output path for the temporary video clip
        temp_output = f"temp_clip_{len(temp_clips)}.mp4"

        out = None
        frame_idx = 0

        # Read and process the video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current time in seconds from the frame index
            current_time = frame_idx / fps

            # Check if the current time falls within the segment's start and end times
            if start_time <= current_time <= end_time:
                if out is None:
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))
                    st.text(f"Initialized writer for clip: {temp_output}")

                # Write the frame to the temporary video clip
                out.write(frame)

            frame_idx += 1

            # Stop once we surpass the end time
            if current_time > end_time:
                break

        # Release the video capture and output objects
        cap.release()
        if out:
            out.release()
            temp_clips.append(temp_output)  # Add the temporary clip to the list
            clip_duration = (frame_idx / fps)  # Calculate the duration of the clip
            total_duration += clip_duration  # Add the clip duration to the total
            st.text(f"Saved temporary clip: {temp_output}, Duration: {clip_duration}s")

    # Convert total duration to minutes
    total_duration_minutes = total_duration / 60
    st.text(f"Total duration of combined clips: {total_duration_minutes} minutes")

    # Merge all temporary clips into the final video using OpenCV
    if temp_clips:
        # Open the first clip to get the properties (resolution, fps, etc.)
        first_clip = cv2.VideoCapture(temp_clips[0])
        frame_width = int(first_clip.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(first_clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = first_clip.get(cv2.CAP_PROP_FPS)
        first_clip.release()

        # Initialize the final video writer
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Loop through and write each temporary clip into the final video
        for clip in temp_clips:
            temp_cap = cv2.VideoCapture(clip)
            while temp_cap.isOpened():
                st.text(f"Clip {clip} has content.")
                ret, frame = temp_cap.read()
                if not ret:
                    break
                out.write(frame)  # Write frame to final video
            temp_cap.release()

        # Release the final video writer
        out.release()
        st.text(f"Final video saved at: {output_path}")

        # Clean up temporary clips
        for clip in temp_clips:
            os.remove(clip)
            st.text(f"Deleted temporary clip: {clip}")

        return output_path  # Return the path to the merged video
    else:
        st.text("No clips to merge")
        return "No clips to merge"



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
                    download_status, downloaded_video_paths = download_video(url)
                    st.text(f"{downloaded_video_paths}")
                    progress_bar.progress(100)
                    status_text.text(download_status)
                    if "successfully" in download_status:
                        st.success(f"Downloaded: {url}")
                        if downloaded_video_paths:
                            st.video(downloaded_video_paths)
                    else:
                        st.error(f"Failed to download: {url}")

    if st.button("Combine and Play"):
        if 'query_output' in st.session_state and st.session_state.query_output:
            downloaded_video_path = []
            st.text(f"{downloaded_video_path}")
            # Ensure that `input_urls` is set and split correctly
            if input_urls:  
                for url in input_urls.split(","):
                    url = url.strip()  # Clean the URL
                    # Call the download_video function to download the video and get the path
                    download_status, video_path = download_video(url)  # Get the download status and path
                    if video_path:  # Check if a valid path was returned
                        downloaded_video_path.append(video_path)  # Add the path to the list
                        st.text(f"Downloaded video: {video_path}")  # Get the path of the downloaded video
            video_segments = extract_video_segments(st.session_state.query_output)
            output_filename = "final_video.mp4"
            final_path = clip_and_merge_videos(video_segments,video_path, output_filename)
            # Check if the final video file exists
            if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                st.success("Final video created successfully!")
                st.write(f"Clips stored in: {os.path.abspath('temp_videos')}")
                st.video(final_path)  # Display the final video
            else:
                st.error("Failed to create the final video. Please check the video segments and try again.")
        else:
            st.error("No segments to combine. Process a query first.")

if __name__ == "__main__":
    main()
