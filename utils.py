from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, BertModel
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

class ASRSAInference:
    def __init__(self, asr_model_name='bhuang/asr-wav2vec2-french', sa_model_name='billfass/bert-base-sentiment-classification'):
        self.processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)
        self.model_sample_rate = self.processor.feature_extractor.sampling_rate

        self.tokenizer = AutoTokenizer.from_pretrained(sa_model_name)
        self.bert_model = BertModel.from_pretrained(sa_model_name)
        
    def asr(self, wav_path):
        """
        Cette méthode effectue la reconnaissance automatique de la parole (ASR) sur un fichier audio donné
        en utilisant un modèle Wav2Vec2 pré-entraîné en français.

        Args:
            wav_path (str): Le chemin vers le fichier audio à transcrire.

        Returns:
            str: La séquence de mots prédite à partir du fichier audio.
        """
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.squeeze(axis=0)  # mono

        # resample
        if sample_rate != self.model_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.model_sample_rate)
            waveform = resampler(waveform)

        # normalize
        input_dict = self.processor(waveform, sampling_rate=self.model_sample_rate, return_tensors="pt")

        with torch.inference_mode():
            logits = self.model(input_dict.input_values).logits

        # decode
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = self.processor.batch_decode(predicted_ids)[0]

        return predicted_sentence
    
    def asrgradio(self, wav_path):
        """
        Cette méthode effectue la reconnaissance automatique de la parole (ASR) sur un fichier audio donné
        en utilisant un modèle Wav2Vec2 pré-entraîné en français.

        Args:
            wav_path (str): Le chemin vers le fichier audio à transcrire.

        Returns:
            str: La séquence de mots prédite à partir du fichier audio.
        """
        waveform, sample_rate = sf.read(wav_path.name)
        waveform = torch.tensor(waveform).float()

        # resample
        if sample_rate != self.model_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.model_sample_rate)
            waveform = resampler(waveform)

        # normalize
        input_dict = self.processor(waveform, sampling_rate=self.model_sample_rate, return_tensors="pt")

        with torch.inference_mode():
            logits = self.model(input_dict.input_values).logits

        # decode
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = self.processor.batch_decode(predicted_ids)[0]

        return predicted_sentence
    
    def sa(self, sentence):
        """
        Cette méthode effectue des prédictions de sentiment en utilisant le modèle "billfass/bert-base-sentiment-classification". 
        Le modèle est pré-entraîné pour la classification des sentiments, 
        et extraie l'étiquette de sentiment (positif ou négatif) en fonction de la prédiction du modèle.

        Args:
            sentence (str): phrase dont on veut classifier le sentiment.

        Returns:
            str: Un string contenant le sentiment prédit (positif ou négatif).
        """
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            output = self.bert_model(**inputs)

        logits = output.last_hidden_state
        linear_layer = nn.Linear(768, 2)
        predict = torch.max(linear_layer(logits.view(-1, 768)), dim=1)
        label_pred = predict.indices[torch.argmax(predict.values).item()].item()

        if label_pred == 0:
            sentiment = "positive"
        else:
            sentiment = "negative"

        return sentiment
    
    def asr_sa(self, audio_path):
        try:
            transcription = self.asr(audio_path)
            sentiment = self.sa(transcription)
            return {
            "transcription" : transcription,
            "sentiment" : sentiment
            } 
        except Exception as e:
            return str(e)