from flask import Flask, request, jsonify, send_file
from google.cloud import firestore, storage
import logging
import time
import re
import os
import datetime
import pandas as pd
from functools import wraps
from collections import Counter
from urllib.parse import quote
import io

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize cloud clients
try:
    db = firestore.Client()
    storage_client = storage.Client()
    logging.info("Successfully initialized Firestore and Storage clients")
except Exception as e:
    logging.error(f"Failed to initialize cloud clients: {str(e)}")
    raise

# Configuration
BUCKET_NAME = "allpdfs-bucket"
BASE_URL = "https://pdf-webhook-121762575730.us-central1.run.app"
# Score thresholds - adjust these to prioritize videos
VIDEO_THRESHOLD = 0.001  # Lower threshold for videos to prioritize them
TEXT_THRESHOLD = 0.05
IMAGE_THRESHOLD = 0.01

def firestore_retry_decorator(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    sleep_time = min(2 ** retries, 10)
                    logging.warning(f"Retry {retries}/{max_retries} - Sleeping {sleep_time}s: {str(e)}")
                    time.sleep(sleep_time)
            logging.error(f"Max retries reached for {func.__name__}")
            raise
        return wrapper
    return decorator

@app.route('/health', methods=['GET'])
def health():
    try:
        db.collection('pdf_text').limit(1).stream()
        storage_client.bucket(BUCKET_NAME).exists()
        return "OK", 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return "Service Unhealthy", 500, {'Content-Type': 'text/plain'}

def calculate_tfidf_score(query, text):
    try:
        if not query or not text:
            return 0.0
            
        query_words = set(re.findall(r'\w+', query.lower()))
        text_words = re.findall(r'\w+', text.lower())
        if not query_words or not text_words:
            return 0.0

        text_freq = Counter(text_words)
        total_words = len(text_words)
        base_score = sum(text_freq[word]/total_words for word in query_words if word in text_freq) / len(query_words)
        
        # Boost exact matches
        exact_match_bonus = 1.0 if query.lower() in text.lower() else 0.0
        
        # Boost title matches for videos
        title_match_bonus = 0.5 if "title" in text.lower() and any(word in text.lower() for word in query_words) else 0.0
        
        final_score = base_score + exact_match_bonus + title_match_bonus
        return final_score

    except Exception as e:
        logging.error(f"Error calculating score: {str(e)}")
        return 0.0

def get_public_url(image_path):
    try:
        if not image_path:
            return None

        # Encode the path properly
        encoded_path = quote(image_path)
        
        # Use the correct URL format with cache breaker
        public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{encoded_path}?t={int(time.time())}"
        
        return public_url
    except Exception as e:
        logging.error(f"Error generating public URL: {str(e)}")
        return None

def search_videos(query):
    try:
        logging.info("=== Starting Video Search ===")
        logging.info(f"Searching videos for query: {query}")
        
        # Query the videos collection from Firestore
        videos_ref = db.collection("videos")
        videos_docs = videos_ref.stream()
        
        results = []
        for doc in videos_docs:
            try:
                video_data = doc.to_dict()
                
                # Combine title and description for searching
                title = str(video_data.get('title', ''))
                description = str(video_data.get('description', ''))
                searchable_text = f"{title} {description}"
                
                # Calculate relevance score
                score = calculate_tfidf_score(query, searchable_text)
                
                # Log each video's score for debugging
                logging.info(f"Video: '{title}' - Score: {score}")

                if score > VIDEO_THRESHOLD:
                    video_data['score'] = score
                    video_data['type'] = 'video'
                    results.append(video_data)
                    logging.info(f"Added video to results: {title}")
            except Exception as e:
                logging.error(f"Error processing video document: {str(e)}")
                continue
                
        logging.info(f"Video search completed. Found {len(results)} matches")
        return results

    except Exception as e:
        logging.error(f"Video search failed: {str(e)}", exc_info=True)
        return []

@firestore_retry_decorator(max_retries=3)
def search_content_with_retry(query):
    results = []
    
    # Check if this is explicitly a video request
    is_video_request = any(word in query.lower() for word in ['video', 'watch', 'play', 'show video'])
    logging.info(f"Is video request: {is_video_request}")
    
    # For video requests, prioritize video results
    if is_video_request:
        video_results = search_videos(query)
        logging.info(f"Found {len(video_results)} video results for explicit video request")
        
        if video_results:
            # For video requests, use the best video match regardless of score
            best_video = max(video_results, key=lambda x: x["score"])
            logging.info(f"Best video match for explicit request: {best_video['title']} with score {best_video['score']}")
            return best_video
    
    # If not a video request or no videos found for video request, search other content
    try:
        text_docs = db.collection("pdf_text").stream()
        for doc in text_docs:
            data = doc.to_dict()
            content = data.get("content", "")
            score = calculate_tfidf_score(query, content)

            if score > TEXT_THRESHOLD:
                results.append({
                    "content": content,
                    "source": data.get("source_file", "Unknown"),
                    "page": data.get("page", "N/A"),
                    "score": score,
                    "type": "text"
                })
    except Exception as e:
        logging.error(f"Text search failed: {str(e)}")

    try:
        image_docs = db.collection("pdf_images_new").stream()
        for doc in image_docs:
            data = doc.to_dict()
            
            image_path = data.get("image_path", "")
            if not image_path:
                logging.warning(f"No image path for document {doc.id}")
                continue
                
            image_url = get_public_url(image_path)
            if not image_url:
                logging.warning(f"Could not generate URL for image: {image_path}")
                continue
                
            description = data.get("description", 
                f"Image from {data.get('source_file', 'unknown document')} page {data.get('page', '')}")
            score = calculate_tfidf_score(query, description)

            if score > IMAGE_THRESHOLD:
                results.append({
                    "image_url": image_url,
                    "description": description,
                    "source": data.get("source_file", "Unknown"),
                    "page": data.get("page", "N/A"),
                    "score": score,
                    "type": "image"
                })
    except Exception as e:
        logging.error(f"Image search failed: {str(e)}")
    
    # If not explicitly a video request, search for videos and add to results
    if not is_video_request:
        video_results = search_videos(query)
        results.extend(video_results)

    if results:
        best_match = max(results, key=lambda x: x["score"])
        logging.info(f"Found best match with score {best_match['score']} of type {best_match['type']}")
        return best_match
    
    logging.info("No results found")
    return None

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        if not request.is_json:
            logging.error("Received non-JSON request")
            return error_response("Invalid content type", 400)

        data = request.get_json()
        logging.info(f"Received request data: {data}")

        # Extract query from different possible formats
        query = ""
        try:
            # Extract tag from Dialogflow CX request
            tag = data.get("fulfillmentInfo", {}).get("tag", "")
            
            # Extract query from Dialogflow CX request format
            if "sessionInfo" in data and "parameters" in data["sessionInfo"]:
                query = data["sessionInfo"]["parameters"].get("query", "")
            # Extract from text field (direct API calls)
            elif "text" in data:
                query = data["text"]
            # Extract from payload (custom format)
            elif "payload" in data and "queryText" in data["payload"]:
                query = data["payload"]["queryText"]
            
            logging.info(f"Webhook tag: {tag}, Query: {query}")
        except Exception as e:
            logging.error(f"Error extracting query: {str(e)}")

        query = query.strip()
        logging.info(f"Final extracted query: '{query}'")
        
        if not query:
            return error_response("Invalid query", 400)

        # Force video search for explicit video requests
        if any(word in query.lower() for word in ['video', 'watch', 'play', 'show video']):
            logging.info("Explicit video request detected, forcing video search")
            video_results = search_videos(query)
            if video_results:
                best_video = max(video_results, key=lambda x: x["score"])
                logging.info(f"Forcing video result: {best_video['title']}")
                return success_response(best_video, query)

        result = search_content_with_retry(query)
        if not result:
            return error_response("No results found", 404)

        logging.info(f"Search result type: {result['type']}")
        logging.info(f"Search result score: {result['score']}")
        
        response = success_response(result, query)
        logging.info(f"Generated response: {response.get_json()}")
        return response

    except Exception as e:
        logging.error(f"Webhook error: {str(e)}", exc_info=True)
        return error_response("Service unavailable", 503)

def success_response(data, query=""):
    if data["type"] == "text":
        return jsonify({
            "sessionInfo": {
                "parameters": {
                    "has_image": False,
                    "has_video": False,
                    "query": query
                }
            },
            "fulfillmentResponse": {
                "messages": [{
                    "text": {
                        "text": [f"{data.get('content', '')}\n\nSource: {data.get('source', 'Unknown')}"]
                    }
                }]
            }
        })
    elif data["type"] == "video":
        video_url = data.get("video_url", "")
        title = data.get("title", "")
        description = data.get("description", "")
        duration = data.get("duration", "")
        views = data.get("views", 0)
        
        return jsonify({
            "sessionInfo": {
                "parameters": {
                    "query": query,
                    "has_video": True,
                    "has_image": False,
                    "video_title": title,
                    "video_url": video_url,
                    "video_description": description,
                    "video_duration": duration,
                    "video_views": views
                }
            },
            "fulfillmentResponse": {
                "messages": [
                    {
                        "text": {
                            "text": [
                                f"🎬 **{title}**\n📺 {description}\n⏱ Duration: {duration} | 👁 Views: {views}\n🔗 Watch here: {video_url}"
                            ]
                        }
                    },
                    {
                        "payload": {
                            "richContent": [
                                [
                                    {
                                        "type": "video",
                                        "rawUrl": video_url,
                                        "accessibilityText": title
                                    }
                                ]
                            ]
                        }
                    }
                ]
            }
        })
    
    elif data["type"] == "image":
        image_url = data.get("image_url", "")
        logging.info(f"Preparing image response with URL: {image_url}")
        
        return jsonify({
            "sessionInfo": {
                "parameters": {
                    "has_image": True,
                    "has_video": False,
                    "query": query,
                    "image_url": image_url,
                    "source": data.get('source', 'Unknown'),
                    "page": data.get('page', 'N/A')
                }
            },
            "fulfillmentResponse": {
                "messages": [
                    {
                        "text": {
                            "text": [f"I found an image from {data.get('source', 'Unknown')} (Page {data.get('page', 'N/A')})"]
                        }
                    },
                    {
                        "payload": {
                            "richContent": [
                                [
                                    {
                                        "type": "image",
                                        "rawUrl": image_url,
                                        "accessibilityText": f"Image from {data.get('source', 'Unknown')}"
                                    }
                                ]
                            ]
                        }
                    }
                ]
            }
        })

def error_response(message, status_code=400):
    return jsonify({
        "sessionInfo": {
            "parameters": {
                "has_image": False,
                "has_video": False
            }
        },
        "fulfillmentResponse": {
            "messages": [{
                "text": {
                    "text": [f"Error: {message}"]
                }
            }]
        }
    }), status_code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
