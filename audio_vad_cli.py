#!/usr/bin/env python3
"""
Audio VAD (Voice Activity Detection) CLI Tool - Version Améliorée
Analyse les fichiers audio pour détecter UNIQUEMENT la parole claire (pas les murmures/chuchotements).

Installation comme commande système:
    chmod +x audio-vad
    sudo cp audio-vad /usr/local/bin/
    # ou
    ln -s $(pwd)/audio-vad ~/.local/bin/audio-vad

Usage:
    audio-vad fichier.mp3
    audio-vad fichier.wav --format text --pretty --strict
"""
import torch
import torchaudio
import numpy as np
import json
import argparse
import sys
import os
from datetime import timedelta
from pathlib import Path
import librosa
from scipy import signal
from sklearn.preprocessing import StandardScaler

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
    
    def _extract_audio_features(self, wav_segment, sr=16000):
        """
        Extrait des caractéristiques audio pour différencier parole claire des murmures
        
        Returns:
            dict: Caractéristiques audio calculées
        """
        # Conversion en float32 si nécessaire
        if wav_segment.dtype != np.float32:
            wav_segment = wav_segment.astype(np.float32)
        
        # 1. Énergie RMS (Root Mean Square)
        rms_energy = np.sqrt(np.mean(wav_segment ** 2))
        
        # 2. Énergie spectrale dans différentes bandes
        # Calcul du spectrogramme
        f, t, Sxx = signal.spectrogram(wav_segment, fs=sr, window='hamming', 
                                      nperseg=512, noverlap=256)
        
        # Bandes de fréquences importantes pour la parole
        # Basse fréquence (80-300 Hz) - fondamentales vocales
        low_freq_mask = (f >= 80) & (f <= 300)
        low_freq_energy = np.mean(Sxx[low_freq_mask, :])
        
        # Moyenne fréquence (300-2000 Hz) - voyelles et consonnes
        mid_freq_mask = (f >= 300) & (f <= 2000)
        mid_freq_energy = np.mean(Sxx[mid_freq_mask, :])
        
        # Haute fréquence (2000-8000 Hz) - consonnes fricatives
        high_freq_mask = (f >= 2000) & (f <= 8000)
        high_freq_energy = np.mean(Sxx[high_freq_mask, :])
        
        # 3. Ratio des énergies (caractéristique importante)
        # Parole claire : plus d'énergie en moyenne fréquence
        # Murmures : énergie plus répartie ou dominante en basse fréquence
        mid_to_low_ratio = mid_freq_energy / (low_freq_energy + 1e-10)
        high_to_low_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        
        # 4. Variabilité spectrale (parole claire plus variable)
        spectral_variance = np.var(np.mean(Sxx, axis=0))
        
        # 5. Centroïde spectral (fréquence "moyenne" du son)
        spectral_centroid = np.sum(f.reshape(-1, 1) * Sxx) / (np.sum(Sxx) + 1e-10)
        
        # 6. Zero Crossing Rate (taux de passages par zéro)
        # Parole claire a généralement plus de variations
        zcr = np.sum(np.diff(np.sign(wav_segment)) != 0) / len(wav_segment)
        
        return {
            'rms_energy': float(rms_energy),
            'low_freq_energy': float(low_freq_energy),
            'mid_freq_energy': float(mid_freq_energy),
            'high_freq_energy': float(high_freq_energy),
            'mid_to_low_ratio': float(mid_to_low_ratio),
            'high_to_low_ratio': float(high_to_low_ratio),
            'spectral_variance': float(spectral_variance),
            'spectral_centroid': float(spectral_centroid),
            'zero_crossing_rate': float(zcr)
        }
    
    def _is_clear_speech(self, features, strict_mode=False):
        """
        Détermine si un segment contient de la parole claire basé sur les caractéristiques
        
        Args:
            features: Dictionnaire des caractéristiques extraites
            strict_mode: Mode strict pour une détection plus sélective
            
        Returns:
            tuple: (is_clear_speech, confidence_score, reasons)
        """
        reasons = []
        score = 0
        max_score = 0
        
        # Seuils adaptatifs selon le mode
        if strict_mode:
            # Mode strict : critères plus sévères
            rms_threshold = 0.02  # Énergie minimum plus élevée
            mid_low_ratio_min = 1.5  # Ratio plus élevé
            zcr_min = 0.05  # Plus de variations requises
            spectral_variance_min = 1e-6
        else:
            # Mode normal
            rms_threshold = 0.01
            mid_low_ratio_min = 1.0
            zcr_min = 0.03
            spectral_variance_min = 1e-7
        
        # Critère 1: Énergie RMS suffisante
        max_score += 20
        if features['rms_energy'] > rms_threshold:
            score += 20
            reasons.append(f"✓ Énergie suffisante ({features['rms_energy']:.4f})")
        else:
            reasons.append(f"✗ Énergie trop faible ({features['rms_energy']:.4f})")
        
        # Critère 2: Ratio moyenne/basse fréquence (crucial pour différencier parole claire)
        max_score += 25
        if features['mid_to_low_ratio'] > mid_low_ratio_min:
            score += 25
            reasons.append(f"✓ Bon ratio mid/low freq ({features['mid_to_low_ratio']:.2f})")
        else:
            reasons.append(f"✗ Ratio mid/low freq faible ({features['mid_to_low_ratio']:.2f})")
        
        # Critère 3: Variabilité spectrale (parole vs bruit constant)
        max_score += 20
        if features['spectral_variance'] > spectral_variance_min:
            score += 20
            reasons.append(f"✓ Variabilité spectrale présente ({features['spectral_variance']:.2e})")
        else:
            reasons.append(f"✗ Manque de variabilité spectrale ({features['spectral_variance']:.2e})")
        
        # Critère 4: Zero Crossing Rate (articulation)
        max_score += 15
        if features['zero_crossing_rate'] > zcr_min:
            score += 15
            reasons.append(f"✓ Bonne articulation (ZCR: {features['zero_crossing_rate']:.4f})")
        else:
            reasons.append(f"✗ Articulation faible (ZCR: {features['zero_crossing_rate']:.4f})")
        
        # Critère 5: Centroïde spectral dans la gamme vocale
        max_score += 10
        if 200 <= features['spectral_centroid'] <= 3000:
            score += 10
            reasons.append(f"✓ Centroïde dans gamme vocale ({features['spectral_centroid']:.0f} Hz)")
        else:
            reasons.append(f"✗ Centroïde hors gamme vocale ({features['spectral_centroid']:.0f} Hz)")
        
        # Critère 6: Énergie dans les hautes fréquences (consonnes)
        max_score += 10
        if features['high_to_low_ratio'] > 0.1:
            score += 10
            reasons.append(f"✓ Présence de consonnes (ratio high/low: {features['high_to_low_ratio']:.3f})")
        else:
            reasons.append(f"✗ Manque de consonnes (ratio high/low: {features['high_to_low_ratio']:.3f})")
        
        confidence = (score / max_score) * 100
        
        # Seuil de décision
        threshold = 70 if strict_mode else 60
        is_clear = confidence >= threshold
        
        return is_clear, confidence, reasons
    
    def analyze_audio(self, audio_path, config=None, strict_mode=False):
        """
        Analyse un fichier audio pour détecter UNIQUEMENT la parole claire
        
        Args:
            audio_path: Chemin vers le fichier audio
            config: Configuration pour l'analyse VAD
            strict_mode: Mode strict pour filtrer plus sévèrement
        
        Returns:
            dict: Résultats de l'analyse
        """
        if config is None:
            # Configuration plus stricte par défaut
            config = {
                'threshold': 0.6 if strict_mode else 0.5,  # Seuil VAD plus élevé
                'min_speech_duration_ms': 500 if strict_mode else 250,  # Durée minimum plus longue
                'min_silence_duration_ms': 200,
                'window_size_samples': 1536,
                'speech_pad_ms': 50  # Padding légèrement augmenté
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
        
        # Détecter les timestamps de parole avec Silero VAD
        speech_timestamps = get_speech_timestamps(
            wav, 
            self.model, 
            sampling_rate=16000,
            **config
        )
        
        # Calculer la durée totale
        total_duration = len(wav) / 16000
        
        # Filtrer les segments avec analyse acoustique avancée
        filtered_segments = []
        segment_details = []
        
        for i, segment in enumerate(speech_timestamps):
            start_sample = segment['start']
            end_sample = segment['end']
            
            # Extraire le segment audio
            wav_segment = wav[start_sample:end_sample].numpy()
            
            # Analyser les caractéristiques acoustiques
            features = self._extract_audio_features(wav_segment)
            
            # Déterminer si c'est de la parole claire
            is_clear, confidence, reasons = self._is_clear_speech(features, strict_mode)
            
            start_time = start_sample / 16000
            end_time = end_sample / 16000
            duration = end_time - start_time
            
            segment_detail = {
                'original_id': i + 1,
                'start_seconds': start_time,
                'end_seconds': end_time,
                'duration_seconds': duration,
                'start_formatted': self.format_time(start_time),
                'end_formatted': self.format_time(end_time),
                'duration_formatted': self.format_time(duration),
                'is_clear_speech': is_clear,
                'confidence_score': confidence,
                'acoustic_features': features,
                'analysis_reasons': reasons
            }
            
            segment_details.append(segment_detail)
            
            # Ne garder que les segments de parole claire
            if is_clear:
                filtered_segments.append({
                    'id': len(filtered_segments) + 1,
                    'type': 'CLEAR_SPEECH',
                    'start_seconds': start_time,
                    'end_seconds': end_time,
                    'duration_seconds': duration,
                    'start_formatted': self.format_time(start_time),
                    'end_formatted': self.format_time(end_time),
                    'duration_formatted': self.format_time(duration),
                    'confidence_score': confidence
                })
        
        # Calculer les statistiques
        stats = self._calculate_statistics(filtered_segments, segment_details, total_duration, strict_mode)
        
        # Préparer le résultat
        result = {
            'file_info': {
                'path': os.path.abspath(audio_path),
                'filename': os.path.basename(audio_path),
                'total_duration_seconds': total_duration,
                'total_duration_formatted': self.format_time(total_duration)
            },
            'analysis_config': {
                **config,
                'strict_mode': strict_mode,
                'acoustic_analysis_enabled': True
            },
            'statistics': stats,
            'clear_speech_segments': filtered_segments,
            'all_detected_segments': segment_details  # Pour debug/analyse
        }
        
        return result
    
    def _calculate_statistics(self, clear_segments, all_segments, total_duration, strict_mode):
        """Calcule les statistiques de l'analyse améliorée"""
        total_clear_speech = sum(s['duration_seconds'] for s in clear_segments)
        total_detected = sum(s['duration_seconds'] for s in all_segments)
        
        # Segments rejetés (probablement murmures/bruit)
        rejected_segments = [s for s in all_segments if not s['is_clear_speech']]
        total_rejected = sum(s['duration_seconds'] for s in rejected_segments)
        
        # Moyennes des scores de confiance
        if clear_segments:
            avg_confidence = np.mean([s['confidence_score'] for s in clear_segments])
        else:
            avg_confidence = 0
        
        if all_segments:
            avg_confidence_all = np.mean([s['confidence_score'] for s in all_segments])
        else:
            avg_confidence_all = 0
        
        stats = {
            'analysis_mode': 'strict' if strict_mode else 'normal',
            'total_segments_detected': len(all_segments),
            'clear_speech_segments': len(clear_segments),
            'rejected_segments': len(rejected_segments),
            'total_clear_speech_seconds': total_clear_speech,
            'total_detected_speech_seconds': total_detected,
            'total_rejected_seconds': total_rejected,
            'total_silence_seconds': total_duration - total_detected,
            'total_clear_speech_formatted': self.format_time(total_clear_speech),
            'total_detected_speech_formatted': self.format_time(total_detected),
            'total_rejected_formatted': self.format_time(total_rejected),
            'clear_speech_percentage': (total_clear_speech / total_duration * 100) if total_duration > 0 else 0,
            'detected_speech_percentage': (total_detected / total_duration * 100) if total_duration > 0 else 0,
            'rejection_rate': (total_rejected / total_detected * 100) if total_detected > 0 else 0,
            'average_confidence_clear': avg_confidence,
            'average_confidence_all': avg_confidence_all,
            'quality_metrics': {
                'segments_with_high_confidence': len([s for s in clear_segments if s['confidence_score'] >= 80]),
                'segments_with_medium_confidence': len([s for s in clear_segments if 60 <= s['confidence_score'] < 80]),
                'segments_with_low_confidence': len([s for s in clear_segments if s['confidence_score'] < 60])
            }
        }
        
        return stats

def create_enhanced_text_report(analysis_result, output_path):
    """Crée un rapport texte détaillé avec l'analyse acoustique"""
    content = []
    
    # En-tête
    content.append("=== ANALYSE AUDIO AVANCÉE - PAROLE CLAIRE UNIQUEMENT ===\n")
    content.append(f"Fichier: {analysis_result['file_info']['filename']}")
    content.append(f"Durée totale: {analysis_result['file_info']['total_duration_formatted']}")
    content.append(f"Mode d'analyse: {analysis_result['statistics']['analysis_mode'].upper()}")
    content.append("\n" + "="*70 + "\n")
    
    # Statistiques détaillées
    stats = analysis_result['statistics']
    content.append("STATISTIQUES DÉTAILLÉES:")
    content.append(f"Segments détectés par VAD: {stats['total_segments_detected']}")
    content.append(f"Segments de parole claire: {stats['clear_speech_segments']}")
    content.append(f"Segments rejetés (murmures/bruit): {stats['rejected_segments']}")
    content.append(f"Taux de rejet: {stats['rejection_rate']:.1f}%")
    content.append("")
    content.append(f"Temps de parole claire: {stats['total_clear_speech_formatted']} ({stats['clear_speech_percentage']:.1f}%)")
    content.append(f"Temps total détecté: {stats['total_detected_speech_formatted']} ({stats['detected_speech_percentage']:.1f}%)")
    content.append(f"Temps rejeté: {stats['total_rejected_formatted']}")
    content.append("")
    content.append(f"Confiance moyenne (parole claire): {stats['average_confidence_clear']:.1f}%")
    content.append(f"Confiance moyenne (tous segments): {stats['average_confidence_all']:.1f}%")
    
    # Qualité des segments
    quality = stats['quality_metrics']
    content.append("")
    content.append("QUALITÉ DES SEGMENTS DE PAROLE CLAIRE:")
    content.append(f"Haute confiance (≥80%): {quality['segments_with_high_confidence']}")
    content.append(f"Confiance moyenne (60-79%): {quality['segments_with_medium_confidence']}")
    content.append(f"Faible confiance (<60%): {quality['segments_with_low_confidence']}")
    
    content.append("\n" + "="*70 + "\n")
    
    # Segments de parole claire
    content.append("SEGMENTS DE PAROLE CLAIRE VALIDÉS:\n")
    
    for segment in analysis_result['clear_speech_segments']:
        confidence_icon = "🟢" if segment['confidence_score'] >= 80 else "🟡" if segment['confidence_score'] >= 60 else "🟠"
        content.append(
            f"{segment['id']:3d}. {confidence_icon} PAROLE CLAIRE | "
            f"{segment['start_formatted']} → {segment['end_formatted']} | "
            f"Durée: {segment['duration_formatted']} | "
            f"Confiance: {segment['confidence_score']:.1f}%"
        )
    
    # Segments rejetés (optionnel, pour debug)
    rejected = [s for s in analysis_result['all_detected_segments'] if not s['is_clear_speech']]
    if rejected:
        content.append(f"\n" + "="*70)
        content.append("SEGMENTS REJETÉS (probables murmures/bruits):\n")
        
        for i, segment in enumerate(rejected, 1):
            content.append(
                f"{i:3d}. 🔴 REJETÉ        | "
                f"{segment['start_formatted']} → {segment['end_formatted']} | "
                f"Durée: {segment['duration_formatted']} | "
                f"Confiance: {segment['confidence_score']:.1f}%"
            )
            
            # Ajouter quelques raisons du rejet
            main_reasons = [r for r in segment['analysis_reasons'] if r.startswith('✗')][:3]
            if main_reasons:
                content.append(f"     Raisons: {'; '.join(main_reasons)}")
            content.append("")
    
    # Sauvegarder
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Analyse audio pour détecter UNIQUEMENT la parole claire (pas les murmures)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s audio.mp3                              # Analyse normale avec filtrage acoustique
  %(prog)s audio.mp3 --strict                     # Mode strict (filtrage sévère)
  %(prog)s audio.mp3 --format text --pretty       # Rapport détaillé au format texte
  %(prog)s audio.mp3 --threshold 0.7 --strict     # Seuil élevé + mode strict
  %(prog)s audio.mp3 --debug                      # Inclut les segments rejetés dans l'analyse
        """
    )
    
    parser.add_argument('audio_file', help='Fichier audio à analyser')
    parser.add_argument('-o', '--output', help='Fichier de sortie (défaut: [audio_file]_clear_vad.json)')
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
    parser.add_argument('--strict', action='store_true',
                       help='Mode strict: filtrage sévère des murmures/chuchotements')
    parser.add_argument('--debug', action='store_true',
                       help='Inclut les segments rejetés dans l\'analyse')
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
        
        # Configuration VAD adaptée au mode strict
        config = {
            'threshold': args.threshold,
            'min_speech_duration_ms': args.min_speech,
            'min_silence_duration_ms': args.min_silence,
            'window_size_samples': 1536,
            'speech_pad_ms': args.speech_pad
        }
        
        # Initialiser l'analyseur
        if not args.quiet:
            mode_text = "STRICT" if args.strict else "NORMAL"
            print(f"🔄 Chargement du modèle VAD avancé (mode {mode_text})...")
        
        analyzer = AudioVADAnalyzer()
        
        # Analyser le fichier
        if not args.quiet:
            print(f"🔄 Analyse acoustique avancée: {args.audio_file}")
            print("🔍 Filtrage des murmures et chuchotements...")
        
        result = analyzer.analyze_audio(args.audio_file, config, strict_mode=args.strict)
        
        # Déterminer les fichiers de sortie
        base_name = Path(args.audio_file).stem
        
        if args.output:
            output_base = Path(args.output).stem
            output_dir = Path(args.output).parent
        else:
            suffix = "_clear_strict_vad" if args.strict else "_clear_vad"
            output_base = f"{base_name}{suffix}"
            output_dir = Path.cwd()
        
        # Nettoyer le résultat si pas en mode debug
        if not args.debug:
            # Supprimer les détails des segments rejetés pour alléger la sortie
            result['all_detected_segments'] = [
                {k: v for k, v in seg.items() if k not in ['acoustic_features', 'analysis_reasons']}
                for seg in result['all_detected_segments']
            ]
        
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
            create_enhanced_text_report(result, text_file)
            
            if not args.quiet:
                print(f"✅ Rapport détaillé sauvegardé: {text_file}")
        
        # Afficher un résumé si pas en mode silencieux
        if not args.quiet:
            stats = result['statistics']
            print(f"\n📊 Résumé de l'analyse (mode {stats['analysis_mode'].upper()}):")
            print(f"   Durée totale: {result['file_info']['total_duration_formatted']}")
            print(f"   Segments détectés: {stats['total_segments_detected']}")
            print(f"   Parole claire: {stats['clear_speech_segments']} segments")
            print(f"   Segments rejetés: {stats['rejected_segments']} (taux: {stats['rejection_rate']:.1f}%)")
            print(f"   Temps de parole claire: {stats['total_clear_speech_formatted']} ({stats['clear_speech_percentage']:.1f}%)")
            print(f"   Confiance moyenne: {stats['average_confidence_clear']:.1f}%")
            
            if stats['clear_speech_segments'] == 0:
                print(f"   ⚠️  Aucune parole claire détectée - essayez le mode normal ou vérifiez l'audio")
            elif stats['rejection_rate'] > 50:
                print(f"   💡 Taux de rejet élevé - l'audio contient beaucoup de murmures/bruits")
    
    except KeyboardInterrupt:
        print(f"\n❌ Analyse interrompue par l'utilisateur", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur: {e}", file=sys.stderr)
        if not args.quiet:
            print(f"\n💡 Vérifiez que:", file=sys.stderr)
            print(f"   1. Le fichier audio existe et est accessible", file=sys.stderr)
            print(f"   2. Les dépendances sont installées:", file=sys.stderr)
            print(f"      pip install torch torchaudio librosa scikit-learn scipy", file=sys.stderr)
            print(f"   3. Le format audio est supporté (.wav, .mp3, .m4a, .flac, etc.)", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()