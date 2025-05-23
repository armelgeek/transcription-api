#!/usr/bin/env python3
import torch
import torchaudio
import numpy as np
import json
import argparse
import sys
import os
from datetime import timedelta
from pathlib import Path

class AudioVADAnalyzer:
    def __init__(self):
        self.model = None
        self.utils = None
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle Silero VAD"""
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le modèle Silero VAD: {e}")
    
    @staticmethod
    def format_time(seconds):
        """Convertit les secondes en format HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def analyze_audio(self, audio_path, config=None):
        """
        Analyse un fichier audio pour détecter parole et silence
        
        Args:
            audio_path: Chemin vers le fichier audio
            config: Configuration pour l'analyse VAD
        
        Returns:
            dict: Résultats de l'analyse
        """
        if config is None:
            config = {
                'threshold': 0.5,
                'min_speech_duration_ms': 250,
                'min_silence_duration_ms': 100,
                'window_size_samples': 1536,
                'speech_pad_ms': 30
            }
        
        # Vérifier l'existence du fichier
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable: {audio_path}")
        
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = self.utils
        
        # Lire le fichier audio
        try:
            wav = read_audio(audio_path, sampling_rate=16000)
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la lecture du fichier audio: {e}")
        
        # Détecter les timestamps de parole
        speech_timestamps = get_speech_timestamps(
            wav, 
            self.model, 
            sampling_rate=16000,
            **config
        )
        
        # Calculer la durée totale
        total_duration = len(wav) / 16000
        
        # Construire les segments
        segments = self._build_segments(speech_timestamps, total_duration)
        
        # Calculer les statistiques
        stats = self._calculate_statistics(segments, total_duration)
        
        # Préparer le résultat
        result = {
            'file_info': {
                'path': os.path.abspath(audio_path),
                'filename': os.path.basename(audio_path),
                'total_duration_seconds': total_duration,
                'total_duration_formatted': self.format_time(total_duration)
            },
            'analysis_config': config,
            'statistics': stats,
            'segments': segments
        }
        
        return result
    
    def _build_segments(self, speech_timestamps, total_duration):
        """Construit la liste des segments de parole et silence"""
        segments = []
        
        # Silence initial
        if speech_timestamps and speech_timestamps[0]['start'] > 0:
            silence_duration = speech_timestamps[0]['start'] / 16000
            segments.append({
                'id': len(segments) + 1,
                'type': 'SILENCE',
                'start_seconds': 0,
                'end_seconds': silence_duration,
                'duration_seconds': silence_duration,
                'start_formatted': self.format_time(0),
                'end_formatted': self.format_time(silence_duration),
                'duration_formatted': self.format_time(silence_duration)
            })
        
        # Segments de parole et silences intermédiaires
        for i, segment in enumerate(speech_timestamps):
            start_time = segment['start'] / 16000
            end_time = segment['end'] / 16000
            duration = end_time - start_time
            
            # Segment de parole
            segments.append({
                'id': len(segments) + 1,
                'type': 'SPEECH',
                'start_seconds': start_time,
                'end_seconds': end_time,
                'duration_seconds': duration,
                'start_formatted': self.format_time(start_time),
                'end_formatted': self.format_time(end_time),
                'duration_formatted': self.format_time(duration)
            })
            
            # Silence après ce segment
            if i < len(speech_timestamps) - 1:
                next_start = speech_timestamps[i + 1]['start'] / 16000
                if next_start > end_time:
                    silence_duration = next_start - end_time
                    segments.append({
                        'id': len(segments) + 1,
                        'type': 'SILENCE',
                        'start_seconds': end_time,
                        'end_seconds': next_start,
                        'duration_seconds': silence_duration,
                        'start_formatted': self.format_time(end_time),
                        'end_formatted': self.format_time(next_start),
                        'duration_formatted': self.format_time(silence_duration)
                    })
        
        # Silence final
        if speech_timestamps and speech_timestamps[-1]['end'] / 16000 < total_duration:
            final_silence_start = speech_timestamps[-1]['end'] / 16000
            silence_duration = total_duration - final_silence_start
            segments.append({
                'id': len(segments) + 1,
                'type': 'SILENCE',
                'start_seconds': final_silence_start,
                'end_seconds': total_duration,
                'duration_seconds': silence_duration,
                'start_formatted': self.format_time(final_silence_start),
                'end_formatted': self.format_time(total_duration),
                'duration_formatted': self.format_time(silence_duration)
            })
        
        return segments
    
    def _calculate_statistics(self, segments, total_duration):
        """Calcule les statistiques de l'analyse"""
        speech_segments = [s for s in segments if s['type'] == 'SPEECH']
        silence_segments = [s for s in segments if s['type'] == 'SILENCE']
        
        total_speech = sum(s['duration_seconds'] for s in speech_segments)
        total_silence = sum(s['duration_seconds'] for s in silence_segments)
        
        # Analyse des courts silences (≤ 2 secondes)
        short_silences = [s for s in silence_segments if s['duration_seconds'] <= 2.0]
        
        stats = {
            'total_segments': len(segments),
            'speech_segments': len(speech_segments),
            'silence_segments': len(silence_segments),
            'total_speech_seconds': total_speech,
            'total_silence_seconds': total_silence,
            'total_speech_formatted': self.format_time(total_speech),
            'total_silence_formatted': self.format_time(total_silence),
            'speech_percentage': (total_speech / total_duration * 100) if total_duration > 0 else 0,
            'silence_percentage': (total_silence / total_duration * 100) if total_duration > 0 else 0,
            'short_silences': {
                'count': len(short_silences),
                'average_duration': np.mean([s['duration_seconds'] for s in short_silences]) if short_silences else 0,
                'total_duration': sum(s['duration_seconds'] for s in short_silences)
            }
        }
        
        return stats

def create_text_report(analysis_result, output_path):
    """Crée un rapport texte à partir des résultats JSON"""
    content = []
    
    # En-tête
    content.append("=== ANALYSE AUDIO - DÉTECTION PAROLE/SILENCE ===\n")
    content.append(f"Fichier: {analysis_result['file_info']['filename']}")
    content.append(f"Durée totale: {analysis_result['file_info']['total_duration_formatted']}")
    content.append(f"Segments détectés: {analysis_result['statistics']['total_segments']}")
    content.append(f"Segments de parole: {analysis_result['statistics']['speech_segments']}")
    content.append(f"Segments de silence: {analysis_result['statistics']['silence_segments']}")
    content.append("\n" + "="*60 + "\n")
    
    # Statistiques
    stats = analysis_result['statistics']
    content.append("STATISTIQUES:")
    content.append(f"Temps total de parole: {stats['total_speech_formatted']} ({stats['speech_percentage']:.1f}%)")
    content.append(f"Temps total de silence: {stats['total_silence_formatted']} ({stats['silence_percentage']:.1f}%)")
    content.append(f"Courts silences (≤2s): {stats['short_silences']['count']} segments")
    content.append("\n" + "="*60 + "\n")
    
    # Détails des segments
    content.append("SEGMENTS DÉTAILLÉS:\n")
    
    for segment in analysis_result['segments']:
        icon = "🎤" if segment['type'] == 'SPEECH' else "🔇"
        type_label = "PAROLE " if segment['type'] == 'SPEECH' else "SILENCE"
        
        content.append(
            f"{segment['id']:3d}. {icon} {type_label:8s} | "
            f"{segment['start_formatted']} → {segment['end_formatted']} | "
            f"Durée: {segment['duration_formatted']}"
        )
    
    # Sauvegarder
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Analyse audio pour détecter les segments de parole et silence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s audio.mp3                           # Analyse basique avec sortie JSON
  %(prog)s audio.mp3 --output results.json    # Spécifier le fichier de sortie
  %(prog)s audio.mp3 --format text             # Sortie au format texte
  %(prog)s audio.mp3 --format both             # Sortie JSON + texte
  %(prog)s audio.mp3 --threshold 0.3           # Seuil de détection plus sensible
  %(prog)s audio.mp3 --quiet                   # Mode silencieux
        """
    )
    
    parser.add_argument('audio_file', help='Fichier audio à analyser')
    parser.add_argument('-o', '--output', help='Fichier de sortie (défaut: [audio_file]_vad.json)')
    parser.add_argument('-f', '--format', choices=['json', 'text', 'both'], 
                       default='json', help='Format de sortie (défaut: json)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Seuil de détection VAD (0.1-0.9, défaut: 0.5)')
    parser.add_argument('--min-speech', type=int, default=250,
                       help='Durée minimale de parole en ms (défaut: 250)')
    parser.add_argument('--min-silence', type=int, default=100,
                       help='Durée minimale de silence en ms (défaut: 100)')
    parser.add_argument('--speech-pad', type=int, default=30,
                       help='Padding autour de la parole en ms (défaut: 30)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Mode silencieux (pas de sortie console)')
    parser.add_argument('--pretty', action='store_true',
                       help='Formatage JSON indenté')
    
    args = parser.parse_args()
    
    try:
        # Vérifier le fichier d'entrée
        if not os.path.exists(args.audio_file):
            print(f"❌ Erreur: Fichier introuvable: {args.audio_file}", file=sys.stderr)
            sys.exit(1)
        
        # Configuration VAD
        config = {
            'threshold': args.threshold,
            'min_speech_duration_ms': args.min_speech,
            'min_silence_duration_ms': args.min_silence,
            'window_size_samples': 1536,
            'speech_pad_ms': args.speech_pad
        }
        
        # Initialiser l'analyseur
        if not args.quiet:
            print("🔄 Chargement du modèle VAD...")
        
        analyzer = AudioVADAnalyzer()
        
        # Analyser le fichier
        if not args.quiet:
            print(f"🔄 Analyse du fichier: {args.audio_file}")
        
        result = analyzer.analyze_audio(args.audio_file, config)
        
        # Déterminer les fichiers de sortie
        base_name = Path(args.audio_file).stem
        
        if args.output:
            output_base = Path(args.output).stem
            output_dir = Path(args.output).parent
        else:
            output_base = f"{base_name}_vad"
            output_dir = Path.cwd()
        
        # Sauvegarder selon le format demandé
        if args.format in ['json', 'both']:
            json_file = output_dir / f"{output_base}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                if args.pretty:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(result, f, ensure_ascii=False)
            
            if not args.quiet:
                print(f"✅ Résultats JSON sauvegardés: {json_file}")
        
        if args.format in ['text', 'both']:
            text_file = output_dir / f"{output_base}.txt"
            create_text_report(result, text_file)
            
            if not args.quiet:
                print(f"✅ Rapport texte sauvegardé: {text_file}")
        
        # Afficher un résumé si pas en mode silencieux
        if not args.quiet:
            stats = result['statistics']
            print(f"\n📊 Résumé de l'analyse:")
            print(f"   Durée totale: {result['file_info']['total_duration_formatted']}")
            print(f"   Segments: {stats['total_segments']} (🎤 {stats['speech_segments']} parole, 🔇 {stats['silence_segments']} silence)")
            print(f"   Parole: {stats['total_speech_formatted']} ({stats['speech_percentage']:.1f}%)")
            print(f"   Silence: {stats['total_silence_formatted']} ({stats['silence_percentage']:.1f}%)")
            print(f"   Courts silences: {stats['short_silences']['count']} segments")
        
        # Sortie JSON sur stdout si format JSON uniquement
        if args.format == 'json' and not args.output:
            if args.pretty:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(json.dumps(result, ensure_ascii=False))
    
    except KeyboardInterrupt:
        print(f"\n❌ Analyse interrompue par l'utilisateur", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur: {e}", file=sys.stderr)
        if not args.quiet:
            print(f"\n💡 Vérifiez que:", file=sys.stderr)
            print(f"   1. Le fichier audio existe et est accessible", file=sys.stderr)
            print(f"   2. torch et torchaudio sont installés: pip install torch torchaudio", file=sys.stderr)
            print(f"   3. Le format audio est supporté (.wav, .mp3, .m4a, .flac, etc.)", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()