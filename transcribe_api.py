#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import json
import tempfile

EXTENSIONS_AUDIO = (
    ".opus", ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".aiff", ".wma"
)

def trouver_fichiers_audio(repertoire):
    fichiers_audio = []
    try:
        for nom_fichier in os.listdir(repertoire):
            chemin_fichier = os.path.join(repertoire, nom_fichier)
            if os.path.isfile(chemin_fichier) and nom_fichier.lower().endswith(EXTENSIONS_AUDIO):
                fichiers_audio.append(chemin_fichier)
    except FileNotFoundError:
        print(f"Erreur: Répertoire '{repertoire}' introuvable.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la recherche des fichiers dans '{repertoire}': {e}", file=sys.stderr)
        sys.exit(1)
    return fichiers_audio

def detecter_parole_ffmpeg(chemin_audio, repertoire_sortie):
    """Utilise FFmpeg pour détecter les segments de silence/parole"""
    print(f"Détection de parole avec FFmpeg: {chemin_audio}")

    nom_base = os.path.splitext(os.path.basename(chemin_audio))[0]
    nom_fichier_sortie = f"{nom_base}_timestamps.json"
    chemin_fichier_sortie = os.path.join(repertoire_sortie, nom_fichier_sortie)

    if os.path.exists(chemin_fichier_sortie) and os.path.getsize(chemin_fichier_sortie) > 0:
        print(f"Le fichier '{nom_fichier_sortie}' existe déjà. Passage au suivant.")
        return True

    try:
        # Commande FFmpeg pour détecter les silences
        commande = [
            "ffmpeg",
            "-i", chemin_audio,
            "-af", "silencedetect=noise=-30dB:duration=0.5",
            "-f", "null",
            "-"
        ]
        
        resultat = subprocess.run(commande, capture_output=True, text=True, stderr=subprocess.STDOUT)
        
        if resultat.returncode != 0 and "silencedetect" not in resultat.stdout:
            print(f"ERREUR FFmpeg pour '{os.path.basename(chemin_audio)}'")
            return False
        
        # Parser la sortie FFmpeg pour extraire les silences
        lignes = resultat.stdout.split('\n')
        silences = []
        
        for ligne in lignes:
            if "silence_start" in ligne:
                try:
                    debut_silence = float(ligne.split("silence_start: ")[1].split()[0])
                    silences.append({"debut": debut_silence, "fin": None})
                except:
                    continue
            elif "silence_end" in ligne and silences:
                try:
                    fin_silence = float(ligne.split("silence_end: ")[1].split()[0])
                    if silences[-1]["fin"] is None:
                        silences[-1]["fin"] = fin_silence
                except:
                    continue
        
        # Obtenir la durée totale du fichier
        commande_duree = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            chemin_audio
        ]
        
        try:
            duree_result = subprocess.run(commande_duree, capture_output=True, text=True)
            duree_totale = float(duree_result.stdout.strip())
        except:
            duree_totale = 0
            print("Attention: Impossible d'obtenir la durée totale")
        
        # Convertir les silences en segments de parole
        segments_parole = []
        temps_precedent = 0
        
        for silence in silences:
            if silence["fin"] is not None:
                # Segment de parole avant ce silence
                if silence["debut"] > temps_precedent + 0.5:  # Au moins 0.5s de parole
                    segments_parole.append({
                        "debut": round(temps_precedent, 2),
                        "fin": round(silence["debut"], 2),
                        "duree": round(silence["debut"] - temps_precedent, 2)
                    })
                temps_precedent = silence["fin"]
        
        # Dernier segment de parole après le dernier silence
        if duree_totale > temps_precedent + 0.5:
            segments_parole.append({
                "debut": round(temps_precedent, 2),
                "fin": round(duree_totale, 2),
                "duree": round(duree_totale - temps_precedent, 2)
            })
        
        # Si aucun silence détecté, considérer tout comme de la parole
        if not segments_parole and duree_totale > 0:
            segments_parole.append({
                "debut": 0,
                "fin": round(duree_totale, 2),
                "duree": round(duree_totale, 2)
            })
        
        # Créer le fichier de résultat
        timestamps_data = {
            "fichier_audio": os.path.basename(chemin_audio),
            "duree_totale_fichier": round(duree_totale, 2),
            "duree_totale_parole": round(sum(s["duree"] for s in segments_parole), 2),
            "nombre_segments": len(segments_parole),
            "segments_parole": segments_parole,
            "methode": "FFmpeg silencedetect"
        }
        
        with open(chemin_fichier_sortie, 'w', encoding='utf-8') as f:
            json.dump(timestamps_data, f, indent=2, ensure_ascii=False)
        
        print(f"Détection terminée: {len(segments_parole)} segments de parole")
        print(f"Durée de parole: {timestamps_data['duree_totale_parole']:.1f}s / {duree_totale:.1f}s")
        return True
        
    except FileNotFoundError:
        print("ERREUR: FFmpeg n'est pas installé. Installez-le avec:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Téléchargez depuis https://ffmpeg.org/")
        return False
    except Exception as e:
        print(f"Erreur lors de la détection pour '{chemin_audio}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Détecte les segments de parole en utilisant FFmpeg (pas de dépendances Python complexes).",
        usage="%(prog)s <repertoire_contenant_fichiers_audio>"
    )
    parser.add_argument(
        "repertoire_cible",
        metavar="repertoire_contenant_fichiers_audio", 
        help="Chemin vers le répertoire contenant les fichiers audio à analyser."
    )

    args = parser.parse_args()
    repertoire_cible = args.repertoire_cible
    repertoire_sortie = repertoire_cible

    if not os.path.isdir(repertoire_cible):
        print(f"Erreur: Répertoire '{repertoire_cible}' introuvable.", file=sys.stderr)
        sys.exit(1)

    print(f"Recherche de fichiers audio dans: {repertoire_cible}")

    fichiers_audio = trouver_fichiers_audio(repertoire_cible)

    if not fichiers_audio:
        print(f"Aucun fichier audio trouvé dans '{repertoire_cible}'.")
        sys.exit(0)

    print(f"\nDétection de parole pour {len(fichiers_audio)} fichiers...")
    print(f"Méthode: FFmpeg silencedetect")
    print(f"Seuil de bruit: -30dB")
    print(f"Durée minimale de silence: 0.5s")
    print("-" * 60)

    nombre_succes = 0
    nombre_erreurs = 0

    for fichier_audio in fichiers_audio:
        if detecter_parole_ffmpeg(fichier_audio, repertoire_sortie):
            nombre_succes += 1
        else:
            nombre_erreurs += 1
        print("-" * 60)

    print("Processus terminé.")
    print(f"Fichiers traités: {nombre_succes}")
    if nombre_erreurs > 0:
        print(f"Erreurs: {nombre_erreurs}")

    sys.exit(0 if nombre_erreurs == 0 else 1)

if __name__ == "__main__":
    main()