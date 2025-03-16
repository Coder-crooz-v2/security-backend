from resemblyzer import preprocess_wav, VoiceEncoder
import uvicorn
import numpy as np
from itertools import groupby
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from numpy.linalg import norm
from dotenv import load_dotenv
import tempfile
import os
import speech_recognition as sr
from pymongo import MongoClient
import difflib
import cv2
import time
from deepface import DeepFace
from pydantic import BaseModel
import base64

load_dotenv()

app = FastAPI(
    title="Voice Biometrics API",
    description="A production-ready API for voice enrollment and verification with basic anti-spoofing measures.",
)

MONGODB_URI = os.getenv("MONGODB_URI")
if MONGODB_URI is None:
    raise ValueError("Please set the MONGODB_URI environment variable.")

client = MongoClient(MONGODB_URI)
db_voice = client.voice_biometrics
voices_collection = db_voice.embeddings
db_face = client.face_recognition
faces_collection = db_face.faces

class FaceRequest(BaseModel):
    username: str

def cosine_similarity(a: tuple, b: tuple):
    return np.dot(a, b) / (norm(a) * norm(b))

def text_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity ratio between two strings using difflib.
    """
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def encode_image_to_base64(image: np.ndarray) -> str:
    # Encode an image (numpy array) as base64.
    ret, buf = cv2.imencode('.jpg', image)
    if not ret:
        raise Exception("Image encoding failed.")
    b64_str = base64.b64encode(buf).decode('utf-8')
    return b64_str

def decode_base64_to_image(b64_str: str) -> np.ndarray:
    # Decode a base64 string back to a numpy image.
    buf = base64.b64decode(b64_str)
    image = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return image

def capture_face(prompt: str = "Press 'c' to capture your face or wait for automatic capture..."):
    cap = cv2.VideoCapture(0)
    captured_face = None
    start_time = time.time()
    print(prompt)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Show instructions on the frame.
        cv2.putText(frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Face Capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        # If 'c' is pressed, capture the face immediately.
        if key == ord('c'):
            captured_face = frame.copy()
            print("Face captured!")
            break
        
        # If 3 seconds have passed, automatically capture the current frame.
        if time.time() - start_time > 5:
            captured_face = frame.copy()
            print("Auto-capturing face due to timeout.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_face

wavs = []
@app.get("/test-point")
def test():
    return JSONResponse(
        status_code=200,
        content={"message":"Hello this end point is working"}
    )

@app.post("/upload-voice")
def upload_voice(audio_file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.file.read())
            tmp.flush()
            tmp_path = tmp.name
        # Process the temporary file.
        wav = preprocess_wav(tmp_path)
        wavs.append(wav)
        print(wavs)
        return JSONResponse(
            status_code=200,
            content={"message": "Upload successful."}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Upload failed due to internal server error "+str(e))
    finally:
        os.remove(tmp_path)

@app.post("/register-voice")
def register_user(username: str = Form(...)):
    global wavs
    try:
        print(wavs)
        wavs_array = np.concatenate(wavs, axis=0).astype(np.float32)
        wavs_array = wavs_array.reshape(1,-1)
        speaker_wavs = {username: [wavs_array[list(indices)] for _,indices in groupby(range(len(wavs_array)))]}
        wavs.clear()
        encoder = VoiceEncoder()
        utterances = speaker_wavs[username][0]
        utterance_embeds = np.array([encoder.embed_utterance(audio) for audio in utterances])
        embeds_list = utterance_embeds.tolist()
        insert_response = voices_collection.insert_one({"username": username, "embeddings": embeds_list})
        if insert_response.acknowledged:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Voice registration successful.",
                    "username": username
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "Voice registration failed."}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-voice")
async def verify_voice(username: str = Form(...), audio_file: UploadFile = File(...)):
    try:
        encoder = VoiceEncoder()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file.file.read())
            tmp.flush()
            tmp_path = tmp.name
        
        # Process the temporary file
        test_wav = preprocess_wav(tmp_path)
        os.remove(tmp_path)
        # Create a voice encoder object
        test_embeddings = encoder.embed_utterance(test_wav)
        record = voices_collection.find_one({"username": username})
        if record is None:
            return JSONResponse(
                status_code=400,
                content={"message": "Verification failed. No voice record found."}
            )
        utterance_embeds = np.array(record["embeddings"])
        similarities = [cosine_similarity(test_embeddings, embed) for embed in utterance_embeds]

        # Determine the most similar speaker
        predicted_index = np.argmax(similarities)
        if similarities[predicted_index] > 0.9:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Verification successful!!",
                    "username": username
                }
            )
        else: 
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Verification failed. Speaker not recognized."
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Verification failed due to intenal server error: "+str(e))

@app.post("/speech-to-text")
async def speech_to_text(provided_text: str = Form(...), audio_file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            tmp.write(audio_file.file.read())
            tmp.flush()
            src_path = tmp.name
        
        # Convert the .m4a file to a WAV file.

        # Initialize the recognizer and process the audio.
        recognizer = sr.Recognizer()
        with sr.AudioFile(src_path) as source:
            audio_data = recognizer.record(source)
        os.remove(src_path)

        # Use recognizer to convert speech to text with Google's API.
        detected_text = recognizer.recognize_google(audio_data)
        similarity = text_similarity(provided_text, detected_text)
        threshold = 0.8  # adjust threshold (0.0 to 1.0) as needed; higher means stricter

        result = similarity >= threshold

        return JSONResponse(
            status_code=200,
            content={
                "provided_text": provided_text,
                "detected_text": detected_text,
                "similarity": similarity,
                "result": result  # True if similarity meets or exceeds the threshold, else False.
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Speech detection failed: " + str(e))

@app.post("/register-face")
def register_face(request: FaceRequest):
    """
    Opens the camera, prompts the user to position their face, 
    captures a reference face image, encodes it and stores it in MongoDB.
    """
    try:
        username = request.username
        face_image = capture_face("Position your face and press 'c' to register.")
        if face_image is None:
            raise HTTPException(status_code=400, detail="No face captured for registration.")
        
        # Encode and store the image.
        face_encoded = encode_image_to_base64(face_image)
        record = {"username": username, "face": face_encoded}
        result = faces_collection.insert_one(record)
        if result.acknowledged:
            return JSONResponse(
                status_code=200,
                content={"message": "Face registered successfully.", "username": username}
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "Face registration failed."}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Registration error: " + str(e))


@app.post("/verify-face")
def verify_face(request: FaceRequest):
    """
    Fetches the registered face from MongoDB and opens the camera in live mode
    for 3 seconds. Each captured frame is compared with the registered reference.
    Returns a 'matched' response if one frame verifies and 'not matched' otherwise.
    """
    try:
        username = request.username

        # Retrieve the registered face from the database.
        record = faces_collection.find_one({"username": username})
        if record is None:
            raise HTTPException(status_code=404, detail="No registered face found for this user.")

        # Decode the stored reference face image.
        reference_face = decode_base64_to_image(record["face"])

        cap = cv2.VideoCapture(0)
        captured_frames = []
        capture_duration = 1  # seconds
        start_time = time.time()
        print("🔍 Live face comparison started... Capturing frames for 3 seconds.")

        # Capture frames for the specified duration.
        while time.time() - start_time < capture_duration:
            ret, frame = cap.read()
            if ret:
                captured_frames.append(frame.copy())
                cv2.imshow("Live Face Comparison", frame)
            # Wait for 1ms for a display refresh.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if not captured_frames:
            raise HTTPException(status_code=400, detail="No frames captured.")

        # Process each captured frame for verification.
        matched_count = 0
        print("🔍 Verifying captured frames...")
        frame_count = 1
        for frame in captured_frames:
            print("Frame: ", frame_count)
            frame_count += 1
            try:
                result = DeepFace.verify(frame, reference_face, enforce_detection=False)
                if result.get("verified", False):
                    matched_count += 1
            except Exception as e:
                # Log error if needed; skip this frame if DeepFace fails.
                print("DeepFace verification error:", e)
                continue

        total_frames = len(captured_frames)
        match_ratio = matched_count / total_frames
        print(f"Matched frames: {matched_count}/{total_frames} (Ratio: {match_ratio:.2f})")

        if match_ratio >= 0.7:
            return JSONResponse(
                status_code=200,
                content={"message": "Face verified successfully.", "username": username}
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Face not matched. Verification failed",
                    "username": username,
                    "match_ratio": match_ratio
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Verification error: " + str(e))

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
