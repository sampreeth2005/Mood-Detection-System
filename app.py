import cv2
from deepface import DeepFace
from textblob import TextBlob
from datetime import datetime
import json
import os

print("\nAI Mental Health Companion")
print("This system provides non-medical emotional support.")
print("Your data is stored locally and not shared.\n")


FILE_NAME = "emotion_history.json"

if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, "w") as f:
        json.dump([], f)


def save_emotion(emotion):
    with open(FILE_NAME, "r") as f:
        data = json.load(f)

    data.append({
        "emotion": emotion,
        "time": str(datetime.now())
    })

    with open(FILE_NAME, "w") as f:
        json.dump(data, f)


def detect_facial_emotion():
    print("Opening camera... Press 'q' to capture.")
    cap = cv2.VideoCapture(0)

    emotion = "neutral"

    while True:
        ret, frame = cap.read()
        cv2.imshow("Press Q to Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except:
                emotion = "neutral"
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion


def analyze_text_mood(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"


def ask_questions():
    print("\nPlease answer a few questions:")

    q1 = input("1. How are you feeling today? ")
    q2 = input("2. Did something stressful happen recently? ")
    q3 = input("3. Are you sleeping well? ")

    combined_text = q1 + " " + q2 + " " + q3
    return analyze_text_mood(combined_text)


def give_suggestions(mood):
    print("\n===== AI Recommendation Engine =====")

    if mood == "negative":
        print("It seems you may be feeling low.")
        print("Here are some suggestions:")
        print("- Try deep breathing for 5 minutes")
        print("- Take a short walk outside")
        print("- Talk to a trusted friend or family member")
        print("- Listen to calming music")
        print("- Consider reaching out to a mental health professional if needed")

    elif mood == "neutral":
        print("You seem stable.")
        print("Suggestions:")
        print("- Maintain a healthy routine")
        print("- Do something you enjoy today")

    else:
        print("Great! You seem positive.")
        print("Keep up the good energy!")
        print("- Share positivity with others")
        print("- Continue productive habits")


def analyze_pattern():
    with open(FILE_NAME, "r") as f:
        data = json.load(f)

    negative_count = sum(1 for d in data if d["emotion"] == "negative")

    if negative_count >= 3:
        print("\nNotice: Repeated negative moods detected.")
        print("You may consider professional mental health support.")


def main():
    print("Starting Emotion Detection...\n")

    facial_emotion = detect_facial_emotion()
    print("Facial Emotion Detected:", facial_emotion)

    text_mood = ask_questions()
    print("Text Mood Detected:", text_mood)

    # Final mood decision
    if facial_emotion in ["sad", "angry", "fear", "disgust"] or text_mood == "negative":
        final_mood = "negative"
    elif facial_emotion in ["happy", "surprise"] or text_mood == "positive":
        final_mood = "positive"
    else:
        final_mood = "neutral"

    print("\nFinal Mood Classification:", final_mood)

    save_emotion(final_mood)
    give_suggestions(final_mood)
    analyze_pattern()

if __name__ == "__main__":
    main()
