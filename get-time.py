import torch
import torchaudio
import numpy as np
from datetime import timedelta
import os

def format_time(seconds):
    """Convertit les secondes en format HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def detect_speech_segments(audio_path, output_txt_path=None):
    """
    Détecte les segments de parole et de silence dans un fichier audio
    
    Args:
        audio_path: Chemin vers le fichier audio
        output_txt_path: Chemin de sortie pour le fichier texte (optionnel)
    """
    
    # Charger le modèle Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    # Lire le fichier audio
    print(f"Lecture du fichier: {audio_path}")
    wav = read_audio(audio_path, sampling_rate=16000)
    
    # Détecter les timestamps de parole
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=16000,
        threshold=0.5,          # Seuil de détection (0.1-0.9)
        min_speech_duration_ms=250,  # Durée minimale de parole (ms)
        min_silence_duration_ms=100, # Durée minimale de silence (ms)
        window_size_samples=1536,    # Taille de fenêtre
        speech_pad_ms=30            # Padding autour de la parole (ms)
    )
    
    # Préparer les résultats
    results = []
    total_duration = len(wav) / 16000  # Durée totale en secondes
    
    # Ajouter un silence initial si nécessaire
    if speech_timestamps and speech_timestamps[0]['start'] > 0:
        silence_duration = speech_timestamps[0]['start'] / 16000
        results.append({
            'type': 'SILENCE',
            'start': 0,
            'end': silence_duration,
            'duration': silence_duration
        })
    
    # Traiter chaque segment de parole et les silences entre eux
    for i, segment in enumerate(speech_timestamps):
        start_time = segment['start'] / 16000
        end_time = segment['end'] / 16000
        duration = end_time - start_time
        
        # Ajouter le segment de parole
        results.append({
            'type': 'PAROLE',
            'start': start_time,
            'end': end_time,
            'duration': duration
        })
        
        # Ajouter le silence après ce segment (s'il y en a un)
        if i < len(speech_timestamps) - 1:
            next_start = speech_timestamps[i + 1]['start'] / 16000
            if next_start > end_time:
                silence_duration = next_start - end_time
                results.append({
                    'type': 'SILENCE',
                    'start': end_time,
                    'end': next_start,
                    'duration': silence_duration
                })
    
    # Ajouter un silence final si nécessaire
    if speech_timestamps and speech_timestamps[-1]['end'] / 16000 < total_duration:
        final_silence_start = speech_timestamps[-1]['end'] / 16000
        silence_duration = total_duration - final_silence_start
        results.append({
            'type': 'SILENCE',
            'start': final_silence_start,
            'end': total_duration,
            'duration': silence_duration
        })
    
    # Générer le contenu texte
    txt_content = []
    txt_content.append("=== ANALYSE AUDIO - DÉTECTION PAROLE/SILENCE ===\n")
    txt_content.append(f"Fichier: {os.path.basename(audio_path)}")
    txt_content.append(f"Durée totale: {format_time(total_duration)}")
    txt_content.append(f"Segments détectés: {len(results)}")
    txt_content.append(f"Segments de parole: {len([r for r in results if r['type'] == 'PAROLE'])}")
    txt_content.append(f"Segments de silence: {len([r for r in results if r['type'] == 'SILENCE'])}")
    txt_content.append("\n" + "="*60 + "\n")
    
    # Statistiques
    total_speech = sum(r['duration'] for r in results if r['type'] == 'PAROLE')
    total_silence = sum(r['duration'] for r in results if r['type'] == 'SILENCE')
    
    txt_content.append("STATISTIQUES:")
    txt_content.append(f"Temps total de parole: {format_time(total_speech)} ({total_speech/total_duration*100:.1f}%)")
    txt_content.append(f"Temps total de silence: {format_time(total_silence)} ({total_silence/total_duration*100:.1f}%)")
    txt_content.append("\n" + "="*60 + "\n")
    
    # Détails des segments
    txt_content.append("SEGMENTS DÉTAILLÉS:\n")
    
    for i, segment in enumerate(results, 1):
        start_formatted = format_time(segment['start'])
        end_formatted = format_time(segment['end'])
        duration_formatted = format_time(segment['duration'])
        
        if segment['type'] == 'PAROLE':
            txt_content.append(f"{i:3d}. 🎤 PAROLE    | {start_formatted} → {end_formatted} | Durée: {duration_formatted}")
        else:
            txt_content.append(f"{i:3d}. 🔇 SILENCE   | {start_formatted} → {end_formatted} | Durée: {duration_formatted}")
    
    # Sauvegarder le fichier texte
    if output_txt_path is None:
        output_txt_path = audio_path.rsplit('.', 1)[0] + '_vad_analysis.txt'
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_content))
    
    print(f"\nAnalyse terminée!")
    print(f"Résultats sauvegardés dans: {output_txt_path}")
    print(f"Segments détectés: {len(results)}")
    print(f"Parole: {format_time(total_speech)} | Silence: {format_time(total_silence)}")
    
    return results, output_txt_path

def analyze_short_silences(results, max_silence_duration=2.0):
    """
    Analyse spécifique des courts silences (respirations, pauses)
    
    Args:
        results: Résultats de l'analyse VAD
        max_silence_duration: Durée max pour considérer comme "court silence"
    """
    short_silences = [
        r for r in results 
        if r['type'] == 'SILENCE' and r['duration'] <= max_silence_duration
    ]
    
    print(f"\n=== ANALYSE DES COURTS SILENCES (≤ {max_silence_duration}s) ===")
    print(f"Nombre de courts silences: {len(short_silences)}")
    
    if short_silences:
        avg_duration = np.mean([s['duration'] for s in short_silences])
        print(f"Durée moyenne: {avg_duration:.3f}s")
        
        print("\nCourts silences détectés:")
        for silence in short_silences:
            start_formatted = format_time(silence['start'])
            duration_formatted = f"{silence['duration']:.3f}s"
            print(f"  • {start_formatted} - {duration_formatted} (respiration/pause)")

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez par le chemin de votre fichier audio
    audio_file = "./audios/tabac.mp3"  # ou .wav, .m4a, etc.
    
    try:
        # Analyser le fichier
        segments, output_file = detect_speech_segments(audio_file)
        
        # Analyser les courts silences (respirations, pauses)
        analyze_short_silences(segments, max_silence_duration=1.5)
        
        print(f"\n✅ Fichier d'analyse créé: {output_file}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("\nVérifiez que:")
        print("1. Le fichier audio existe")
        print("2. torch et torchaudio sont installés: pip install torch torchaudio")
        print("3. Le format audio est supporté (.wav, .mp3, .m4a, .flac, etc.)")