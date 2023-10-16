import gradio as gr
from utils import ASRSAInference

myasr = ASRSAInference()

def asrsa(wav_path):
    """
    Transcrit un extrait audio, puis analyse le sentiment de la transcription en utilisant des modèles pré-entraînés.

    Args:
        wav_path (str): Chemin vers le fichier audio à transcrire et analyser.

    Returns:
        dict: Un dictionnaire contenant la transcription de l'extrait audio et le sentiment prédit (positif ou négatif).
    """
    try:
        transcription = myasr.asrgradio(wav_path)
        sentiment = myasr.sa(transcription)
        return {
            "transcription" : transcription,
            "sentiment" : sentiment
        } 
    except Exception as e:
        return str(e)

        

iface = gr.Interface(
    fn=asrsa,
    inputs="file",
    outputs="json"
)
iface.launch(debug=True)

