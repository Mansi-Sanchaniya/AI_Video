import streamlit as st
import yt_dlp
from pytube import Playlist
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from yt_dlp import YoutubeDL

def download_video(url):
    temp_dir = "temp_videos"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    ydl_opts = {
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s')
    }
    downloaded_video_path = None
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            downloaded_video_path = ydl.prepare_filename(info)
            st.success("Video downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
    return downloaded_video_path

def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        st.error(f"Transcript not available for {video_url}: {str(e)}")
        return None

def format_transcript(transcript):
    return [
        {
            "start": entry['start'],
            "end": entry['start'] + entry['duration'],
            "text": entry['text']
        } for entry in transcript
    ]

def process_query(query, transcript):
    if not transcript:
        return []
    texts = [entry['text'] for entry in transcript]
    vectorizer = TfidfVectorizer(stop_words='english')
    query_vector = vectorizer.fit_transform([query])
    text_vectors = vectorizer.transform(texts)
    similarities = cosine_similarity(query_vector, text_vectors)[0]
    threshold = 0.3
    relevant_segments = [
        {
            "start": transcript[i]['start'],
            "end": transcript[i]['end'],
            "text": transcript[i]['text']
        } for i, score in enumerate(similarities) if score > threshold
    ]
    return relevant_segments

def clip_video(video_path, segments):
    temp_dir = "temp_clips"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    clip_paths = []
    for i, segment in enumerate(segments):
        start_time = segment['start']
        end_time = segment['end']
        clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")

        with VideoFileClip(video_path) as video:
            clip = video.subclip(start_time, end_time)
            clip.write_videofile(clip_path, codec="libx264")

        clip_paths.append(clip_path)
    return clip_paths

def combine_clips(clip_paths):
    clips = [VideoFileClip(path) for path in clip_paths]
    final_clip = concatenate_videoclips(clips)
    output_path = "final_video.mp4"
    final_clip.write_videofile(output_path, codec="libx264")
    return output_path

def main():
    st.title("YouTube Video Query and Clipping")

    input_url = st.text_input("Enter a YouTube video URL:")
    query = st.text_input("Enter your query:")

    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'segments' not in st.session_state:
        st.session_state.segments = []
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None

    if st.button("Download and Extract Transcript"):
        st.session_state.video_path = download_video(input_url)
        if st.session_state.video_path:
            transcript = get_transcript(input_url)
            if transcript:
                st.session_state.transcript = format_transcript(transcript)
                st.success("Transcript extracted successfully!")

    if st.session_state.transcript:
        st.subheader("Transcript")
        st.write("\n".join([f"[{entry['start']}s - {entry['end']}s] {entry['text']}" for entry in st.session_state.transcript]))

    if st.button("Process Query"):
        if query and st.session_state.transcript:
            st.session_state.segments = process_query(query, st.session_state.transcript)
            if st.session_state.segments:
                st.success(f"Found {len(st.session_state.segments)} relevant segments!")
            else:
                st.warning("No relevant segments found.")

    if st.button("Combine and Play"):
        if st.session_state.segments and st.session_state.video_path:
            clip_paths = clip_video(st.session_state.video_path, st.session_state.segments)
            if clip_paths:
                final_video_path = combine_clips(clip_paths)
                st.success("Final video created successfully!")
                st.video(final_video_path)
            else:
                st.warning("No segments to combine.")
        else:
            st.warning("No segments to combine. Process a query first.")

if __name__ == "__main__":
    main()
