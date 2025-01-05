import spaces
import gradio as gr
import json
import torch
import numpy as np
import librosa
from accelerate.utils.imports import is_cuda_available
from iso639 import iter_langs
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

device = "cuda" if is_cuda_available() else "cpu"
dtype = torch.float16 if is_cuda_available() else torch.float32


alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    dtype=dtype,
)


@spaces.GPU
def process_alignment(audio_waveform, text, language="eng"):
    print(f"{audio_waveform.shape=}, {text=}, {language=}")
    # Generate emissions
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=16
    )

    # Preprocess text
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language=language,
    )

    # Get alignments
    segments, scores, blank_id = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    # Get spans and word timestamps
    spans = get_spans(tokens_starred, segments, blank_id)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    return word_timestamps


def trim_audio(audio_array, sample_rate, word_timestamps):
    start_time = int(word_timestamps[0]["start"] * sample_rate)
    end_time = int(word_timestamps[-1]["end"] * sample_rate)
    print(f"{start_time=}, {end_time=}")
    trimmed_audio = audio_array[start_time:end_time]
    return (sample_rate, trimmed_audio)


def get_language_choices():
    return [f"{lang.pt3} - {lang.name}" for lang in iter_langs() if lang.pt3]


def align(audio, text, language="eng - English"):
    # Extract the ISO 639-3 code from the selected language
    iso_code = language.split(" - ")[0]

    # Convert the input audio to 16kHz mono
    sample_rate, audio_array = audio
    audio_array = (
        audio_array.astype(np.float32) / 32768.0
    )  # Convert to float32 and normalize
    print(f"{sample_rate=}, {audio_array.shape=}")

    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)  # Convert to mono if stereo
    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    # Convert to torch tensor and move to the correct device
    audio_waveform = torch.from_numpy(audio_array).to(device=device, dtype=dtype)

    # Process the alignment
    word_timestamps = process_alignment(audio_waveform, text, iso_code)

    # Trim the audio
    trimmed_audio = trim_audio(audio_array, 16000, word_timestamps)

    # Create JSON output
    output_json = {
        "input_text": text,
        "word_timestamps": word_timestamps,
        "language": language,
    }

    return trimmed_audio, json.dumps(output_json, indent=2)


def align_result_only(audio, text, language="eng - English"):
    # Extract the ISO 639-3 code from the selected language
    iso_code = language.split(" - ")[0]

    # Convert the input audio to 16kHz mono
    sample_rate, audio_array = audio
    audio_array = (
        audio_array.astype(np.float32) / 32768.0
    )  # Convert to float32 and normalize
    print(f"{sample_rate=}, {audio_array.shape=}")

    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)  # Convert to mono if stereo
    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    # Convert to torch tensor and move to the correct device
    audio_waveform = torch.from_numpy(audio_array).to(device=device, dtype=dtype)

    # Process the alignment
    word_timestamps = process_alignment(audio_waveform, text, iso_code)

    # Create JSON output
    output_json = {
        "input_text": text,
        "word_timestamps": word_timestamps,
        "language": language,
    }

    return json.dumps(output_json, indent=2)


# Create Gradio blocks
with gr.Blocks() as demo:
    gr.Markdown("# Forced Alignment")

    gr.Markdown(
        """
    This tool aligns audio with text and provides word-level timestamps.
    
    ## How to use:
    1. Upload an audio file or record audio
    2. Enter the corresponding text
    3. Select the language
    4. Click 'Process' to get the alignment results
    """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Input Audio")
            text_input = gr.Textbox(label="Input Text")
            language_input = gr.Dropdown(
                choices=get_language_choices(), label="Language", value="eng - English"
            )
            submit_button = gr.Button(
                "Get Alignment and Trimmed Audio", variant="primary"
            )
            submit_button_result_only = gr.Button(
                "Get Alignment Only", variant="secondary"
            )

        with gr.Column():
            audio_output = gr.Audio(label="Trimmed Output Audio")
            json_output = gr.JSON(label="Alignment Results")

    submit_button.click(
        fn=align,
        inputs=[audio_input, text_input, language_input],
        outputs=[audio_output, json_output],
    )

    submit_button_result_only.click(
        fn=align_result_only,
        inputs=[audio_input, text_input, language_input],
        outputs=[json_output],
    )

    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ["examples/example1.mp3", "我們搭上公車要回台北了", "zho - Chinese"],
            [
                "examples/example2.wav",
                "ON SATURDAY MORNINGS WHEN THE SODALITY MET IN THE CHAPEL TO RECITE THE LITTLE OFFICE HIS PLACE WAS A CUSHIONED KNEELING DESK AT THE RIGHT OF THE ALTAR FROM WHICH HE LED HIS WING OF BOYS THROUGH THE RESPONSES",
                "eng - English",
            ],
        ],
        inputs=[audio_input, text_input, language_input],
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()
