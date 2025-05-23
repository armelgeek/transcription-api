#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

MODELE = "base"
LANGUE = "de"
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

def transcrire_fichier(chemin_audio, repertoire_sortie):
    print(f"Traitement: {chemin_audio}")

    nom_base = os.path.splitext(os.path.basename(chemin_audio))[0]
    nom_fichier_sortie = f"{nom_base}.{FORMAT_SORTIE}"
    chemin_fichier_sortie = os.path.join(repertoire_sortie, nom_fichier_sortie)

    if os.path.exists(chemin_fichier_sortie) and os.path.getsize(chemin_fichier_sortie) > 0:
        print(f"Le fichier '{nom_fichier_sortie}' existe déjà et n'est pas vide. Passage au suivant.")
        return True

    commande = [
        sys.executable,
        "-m",
        "whisperx",
        chemin_audio,
        "--model", MODELE,
        "--language", LANGUE,
        "--output_format", FORMAT_SORTIE,
        "--output_dir", repertoire_sortie,
        "--segment_resolution", "chunk",
        "--max_line_count", "1",
        "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "--compute_type", "float32",
        "--max_line_width", "-50"
    ]

    try:
        resultat = subprocess.run(commande, check=False)

        if resultat.returncode == 0:
            print(f"Transcription de '{os.path.basename(chemin_audio)}' terminée.")
            return True
        else:
            print(f"ERREUR lors de la transcription de '{os.path.basename(chemin_audio)}'. Code de retour: {resultat.returncode}", file=sys.stderr)
            if os.path.exists(chemin_fichier_sortie):
                try:
                    os.remove(chemin_fichier_sortie)
                    print(f"Fichier de sortie incomplet '{nom_fichier_sortie}' supprimé.")
                except OSError as e:
                    print(f"Avertissement: Impossible de supprimer le fichier de sortie incomplet '{nom_fichier_sortie}': {e}", file=sys.stderr)
            return False

    except ModuleNotFoundError:
        print(f"Erreur: Le module 'whisper' ne semble pas être installé dans l'environnement Python actuel: {sys.executable}", file=sys.stderr)
        print(f"Exécutez: pip install -U openai-whisper", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Erreur inattendue lors de l'exécution de Whisper pour '{chemin_audio}': {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description=f"Transcrit les fichiers audio dans un répertoire en utilisant Whisper (Modèle: {MODELE}).",
        usage="%(prog)s <repertoire_contenant_fichiers_audio>"
    )
    parser.add_argument(
        "repertoire_cible",
        metavar="repertoire_contenant_fichiers_audio",
        help="Chemin vers le répertoire contenant les fichiers audio à transcrire."
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

    print(f"\nDébut de la transcription de {len(fichiers_audio)} fichiers audio trouvés...")
    print(f"Modèle Whisper: {MODELE}")
    print(f"Langue: {LANGUE}")
    print(f"Format de sortie: {FORMAT_SORTIE}")
    print(f"Répertoire de sortie: {repertoire_sortie}")
    if MODELE.lower() == "large":
        print("ATTENTION: Le modèle 'large' nécessite des ressources informatiques importantes (RAM/VRAM) et prendra du temps.")
    print("-" * 40)

    nombre_succes = 0
    nombre_erreurs = 0

    for fichier_audio in fichiers_audio:
        if transcrire_fichier(fichier_audio, repertoire_sortie):
            nombre_succes += 1
        else:
            nombre_erreurs += 1
        print("-" * 40)

    print("Processus de transcription terminé.")
    print(f"Fichiers traités avec succès: {nombre_succes}")
    if nombre_erreurs > 0:
        print(f"Fichiers avec erreur de transcription: {nombre_erreurs}")

    if nombre_erreurs > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()