#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import json

MODELE = "base"
FORMAT_SORTIE = "json"

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

def detecter_parole(chemin_audio, repertoire_sortie):
    print(f"Détection de parole: {chemin_audio}")

    nom_base = os.path.splitext(os.path.basename(chemin_audio))[0]
    nom_fichier_sortie = f"{nom_base}_timestamps.{FORMAT_SORTIE}"
    chemin_fichier_sortie = os.path.join(repertoire_sortie, nom_fichier_sortie)

    if os.path.exists(chemin_fichier_sortie) and os.path.getsize(chemin_fichier_sortie) > 0:
        print(f"Le fichier '{nom_fichier_sortie}' existe déjà et n'est pas vide. Passage au suivant.")
        return True

    # Commande Whisper pour détecter seulement les segments de parole
    commande = [
        sys.executable,
        "-m",
        "whisper",
        chemin_audio,
        "--model", MODELE,
        "--output_format", "json",
        "--output_dir", repertoire_sortie,
        "--language", "fr",  # On utilise français comme base pour la détection
        "--task", "transcribe"
    ]

    try:
        resultat = subprocess.run(commande, check=False, capture_output=True, text=True)

        if resultat.returncode == 0:
            # Lire le fichier JSON généré par Whisper
            fichier_whisper = os.path.join(repertoire_sortie, f"{nom_base}.json")
            
            if os.path.exists(fichier_whisper):
                with open(fichier_whisper, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extraire seulement les timestamps
                segments_parole = []
                for segment in data.get('segments', []):
                    segments_parole.append({
                        "debut": segment.get('start', 0),
                        "fin": segment.get('end', 0),
                        "duree": segment.get('end', 0) - segment.get('start', 0),
                        "confiance": segment.get('avg_logprob', 0)  # Niveau de confiance
                    })
                
                # Sauvegarder les timestamps seulement
                timestamps_data = {
                    "fichier_audio": os.path.basename(chemin_audio),
                    "duree_totale": sum(s["duree"] for s in segments_parole),
                    "nombre_segments": len(segments_parole),
                    "segments_parole": segments_parole
                }
                
                with open(chemin_fichier_sortie, 'w', encoding='utf-8') as f:
                    json.dump(timestamps_data, f, indent=2, ensure_ascii=False)
                
                # Supprimer le fichier Whisper original (contient le texte)
                os.remove(fichier_whisper)
                
                print(f"Détection terminée: {len(segments_parole)} segments de parole trouvés")
                print(f"Durée totale de parole: {timestamps_data['duree_totale']:.2f} secondes")
                return True
            else:
                print(f"ERREUR: Fichier de sortie Whisper non trouvé pour '{os.path.basename(chemin_audio)}'")
                return False
        else:
            print(f"ERREUR lors de la détection pour '{os.path.basename(chemin_audio)}'. Code de retour: {resultat.returncode}", file=sys.stderr)
            if resultat.stderr:
                print(f"Détails de l'erreur: {resultat.stderr}", file=sys.stderr)
            return False

    except ModuleNotFoundError:
        print(f"Erreur: Le module 'whisper' ne semble pas être installé.", file=sys.stderr)
        print(f"Exécutez: pip install openai-whisper", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Erreur inattendue lors de la détection pour '{chemin_audio}': {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description=f"Détecte les segments de parole et extrait les timestamps (pas de transcription du texte).",
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
        print(f"Erreur: Répertoire '{repertoire_cible}' introuvable ou n'est pas un répertoire valide.", file=sys.stderr)
        sys.exit(1)

    print(f"Recherche de fichiers audio ({', '.join(EXTENSIONS_AUDIO)}) dans: {repertoire_cible}")

    fichiers_audio = trouver_fichiers_audio(repertoire_cible)

    if not fichiers_audio:
        print(f"Aucun fichier audio dans les formats spécifiés n'a été trouvé dans '{repertoire_cible}'.")
        sys.exit(0)

    print(f"\nDébut de la détection de parole pour {len(fichiers_audio)} fichiers audio...")
    print(f"Modèle Whisper: {MODELE}")
    print(f"Mode: Détection de segments de parole uniquement (pas de transcription)")
    print(f"Répertoire de sortie: {repertoire_sortie}")
    print("-" * 60)

    nombre_succes = 0
    nombre_erreurs = 0

    for fichier_audio in fichiers_audio:
        if detecter_parole(fichier_audio, repertoire_sortie):
            nombre_succes += 1
        else:
            nombre_erreurs += 1
        print("-" * 60)

    print("Processus de détection terminé.")
    print(f"Fichiers traités avec succès: {nombre_succes}")
    if nombre_erreurs > 0:
        print(f"Fichiers avec erreur: {nombre_erreurs}")

    if nombre_erreurs > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()