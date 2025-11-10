import gradio as gr
from huggingface_hub import InferenceClient
from metrics import (
    init_metrics,
    SERVICE_NAME,
    ACTIVE_REQUESTS,
    instrument_chord_request,
)

# --- Rule-based chord dictionary ---
CHORDS = {
    frozenset(["C", "E", "G"]): "C Major",
    frozenset(["A", "C#", "E"]): "A Major",
    frozenset(["A", "C", "E"]): "A Minor",
    frozenset(["D", "F#", "A"]): "D Major",
    frozenset(["E", "G#", "B"]): "E Major",
    frozenset(["G", "B", "D"]): "G Major",
    frozenset(["F", "A", "C"]): "F Major",
}

init_metrics(
    port=int(os.getenv("METRICS_PORT", "8000")),
    service_name=os.getenv("SERVICE_NAME", "chord-bot-api"),
)

def identify_chord(notes):
    """
    Identify chord name from a list of notes using dictionary lookup.
    """
    key = frozenset([n.upper() for n in notes])
    return CHORDS.get(key, "Unknown Chord")

# --- Hugging Face model client (API-based) ---
client = InferenceClient("google/flan-t5-small")

def predict_chord(notes: str):
    """
    Takes a comma-separated string of notes and predicts the chord.
    """
    if not notes.strip():
        return "Please enter 2 or more notes."

    # Clean and split notes
    note_list = [n.strip() for n in notes.split(",") if n.strip()]
    if len(note_list) < 2:
        return "Please enter at least 2 notes."

    # Rule-based chord lookup
    chord = identify_chord(note_list)
    if chord != "Unknown Chord":
        return chord

    # Fall back to LLM
    prompt = f"Identify the musical chord made of notes: {', '.join(note_list)}"
    try:
        response = client.text_generation(prompt, max_new_tokens=20)
        return response.strip()
    except Exception as e:
        return f"Error calling Hugging Face API: {e}"

# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict_chord,
    inputs=gr.Textbox(lines=1, placeholder="Enter notes, e.g., C,E,G"),
    outputs="text",
    title="Chord Identifier (API-based)",
    description="Enter two or more musical notes (comma-separated) and get the chord name."
)

if __name__ == "__main__":
    import os

    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "8000"))

    demo.launch(server_name=host, server_port=port, share=True)

