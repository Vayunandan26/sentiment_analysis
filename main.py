import os
import re
import boto3
import torch
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from googleapiclient.discovery import build

app = FastAPI(title="YouTube Sentiment Analyzer", lifespan=lifespan)

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "your-s3-bucket-name")
S3_PREFIX = os.getenv("S3_MODEL_PREFIX", "models/quantized_model/")
LOCAL_MODEL_DIR = "./local_quantized_model"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

ml_models = {}

class VideoRequest(BaseModel):
    url: str
    max_comments: int = 100

def download_s3_folder(bucket_name, s3_folder, local_dir):
    s3 = boto3.client('s3')
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        if 'Contents' not in result:
            continue
        for key in result['Contents']:
            file_key = key['Key']
            if file_key.endswith('/'):
                continue
            
            relative_path = os.path.relpath(file_key, s3_folder)
            local_file_path = os.path.join(local_dir, relative_path)
            
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            if not os.path.exists(local_file_path):
                print(f"Downloading {file_key} to {local_file_path}")
                s3.download_file(bucket_name, file_key, local_file_path)

def extract_video_id(url: str) -> str:
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")

def fetch_youtube_comments(video_id: str, max_results: int = 100):
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY environment variable is missing.")
        
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText"
        )
        response = request.execute()
        
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            
        return comments
    except Exception as e:
        raise RuntimeError(f"Failed to fetch comments: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    download_s3_folder(S3_BUCKET, S3_PREFIX, LOCAL_MODEL_DIR)
    
    print("Loading ONNX Model and Tokenizer...")
    ml_models["tokenizer"] = AutoTokenizer.from_pretrained(os.path.join(LOCAL_MODEL_DIR, "tokenizer"))
    ml_models["model"] = ORTModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    print("Model loaded successfully!")
    
    yield 
    ml_models.clear()
    print("Cleaned up model resources.")


@app.post("/analyze")
async def analyze_video(req: VideoRequest):
    try:
        video_id = extract_video_id(req.url)
        comments = fetch_youtube_comments(video_id, req.max_comments)
        
        if not comments:
            return {"message": "No comments found or comments are disabled."}
        inputs = tokenizer(comments, padding=True, truncation=True, max_length=64, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).numpy()

        results = []
        counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        
        for comment, pred in zip(comments, predictions):
            sentiment = LABEL_MAP[pred]
            results.append({"comment": comment, "sentiment": sentiment})
            counts[sentiment] += 1

        total = len(results)
        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
        overall_sentiment = max(counts, key=counts.get)

        return {
            "overall_sentiment": overall_sentiment,
            "percentages": percentages,
            "total_analyzed": total,
            "comments": results
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))