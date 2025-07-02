import speech_recognition as sr
import spacy

nlp = spacy.load("en_core_web_sm")
recognizer = sr.Recognizer()

def parse_voice_command(audio):
    try:
        text = recognizer.recognize_google(audio)
        doc = nlp(text.lower())
        command = {
            "action": None,
            "target": None,
            "id": None,
            "image": "nginx",
            "cpu": 0.1,
            "memory": "512Mi",
            "storage": "512Mi",
            "ports": ["80"]
        }

        # Extract action and target
        for token in doc:
            if token.text in ["deploy", "start", "create"]:
                command["action"] = "deploy"
            elif token.text in ["status", "check", "get"]:
                command["action"] = "status"
            elif token.text in ["stop", "terminate", "delete"]:
                command["action"] = "terminate"
            elif token.text == "deployment":
                command["target"] = "deployment"

        # Extract deployment details
        for i, token in enumerate(doc):
            # ID
            if token.text in ["id", "number"] and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.is_digit or next_token.text.isalnum():
                    command["id"] = next_token.text
            
            # Image
            if token.text in ["app", "application", "image"] and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.text in ["nginx", "ubuntu", "python"]:
                    command["image"] = next_token.text
            
            # CPU
            if token.text in ["cpu", "processor"] and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.is_digit or next_token.text.replace(".", "").isdigit():
                    command["cpu"] = float(next_token.text)
            
            # Memory
            if token.text in ["memory", "ram"] and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.text.endswith(("gb", "mb", "mi", "gi")) or next_token.text.isdigit():
                    command["memory"] = next_token.text.upper() if next_token.text.isalpha() else f"{next_token.text}Mi"
            
            # Storage
            if token.text in ["storage", "disk"] and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.text.endswith(("gb", "mb", "mi", "gi")) or next_token.text.isdigit():
                    command["storage"] = next_token.text.upper() if next_token.text.isalpha() else f"{next_token.text}Mi"
            
            # Ports
            if token.text in ["port", "ports"] and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.is_digit:
                    command["ports"] = [next_token.text]

        return command, text
    except sr.UnknownValueError:
        return {"action": None, "target": None, "id": None, "image": "nginx", "cpu": 0.1, "memory": "512Mi", "storage": "512Mi", "ports": ["80"]}, "Could not understand audio"
    except sr.RequestError as e:
        return {"action": None, "target": None, "id": None, "image": "nginx", "cpu": 0.1, "memory": "512Mi", "storage": "512Mi", "ports": ["80"]}, f"Speech recognition error: {e}"