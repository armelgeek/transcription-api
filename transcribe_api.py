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

def detecter_parole_whisper(chemin_audio, repertoire_sortie):
    print(f"Analyse avec Whisper: {chemin_audio}")

    nom_base = os.path.splitext(os.path.basename(chemin_audio))[0]
    nom_fichier_sortie = f"{nom_base}_segments.json"
    chemin_fichier_sortie = os.path.join(repertoire_sortie, nom_fichier_sortie)

    if os.path.exists(chemin_fichier_sortie) and os.path.getsize(chemin_fichier_sortie) > 0:
        print(f"Le fichier '{nom_fichier_sortie}' existe déjà. Passage au suivant.")
        return True

    # Commande Whisper - on utilise français comme langue de base pour la détection
    commande = [
        "whisper",
        chemin_audio,
        "--model", MODELE,
        "--output_format", "json",
        "--output_dir", repertoire_sortie,
        "--language", "fr",
        "--task", "transcribe"
    ]

    try:
        resultat = subprocess.run(commande, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if resultat.returncode == 0:
            # Lire le fichier JSON généré par Whisper
            fichier_whisper = os.path.join(repertoire_sortie, f"{nom_base}.json")
            
            if os.path.exists(fichier_whisper):
                with open(fichier_whisper, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Traiter les segments et anonymiser le texte
                segments_anonymes = []
                for i, segment in enumerate(data.get('segments', []), 1):
                    # Analyser la confiance pour déterminer si c'est vraiment du français
                    confiance = segment.get('avg_logprob', -1)
                    texte_original = segment.get('text', '').strip()
                    
                    # Si la confiance est faible ou le texte semble bizarre, on anonymise
                    texte_final = anonymiser_texte_si_necessaire(texte_original, confiance)
                    
                    segments_anonymes.append({
                        "segment": i,
                        "debut": round(segment.get('start', 0), 2),
                        "fin": round(segment.get('end', 0), 2),
                        "duree": round(segment.get('end', 0) - segment.get('start', 0), 2),
                        "texte": texte_final,
                        "confiance": round(confiance, 3),
                        "langue_detectee": detecter_langue_probable(texte_original)
                    })
                
                # Créer le fichier de résultat final
                result_data = {
                    "fichier_audio": os.path.basename(chemin_audio),
                    "duree_totale": round(data.get('segments', [{}])[-1].get('end', 0), 2) if data.get('segments') else 0,
                    "nombre_segments": len(segments_anonymes),
                    "segments": segments_anonymes,
                    "modele_whisper": MODELE,
                    "note": "Texte anonymisé pour les langues non reconnues"
                }
                
                with open(chemin_fichier_sortie, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                # Supprimer le fichier Whisper original
                os.remove(fichier_whisper)
                
                print(f"Analyse terminée: {len(segments_anonymes)} segments détectés")
                print(f"Durée totale: {result_data['duree_totale']:.1f} secondes")
                
                # Statistiques sur l'anonymisation
                textes_anonymes = sum(1 for s in segments_anonymes if s['texte'].startswith('[PAROLE'))
                if textes_anonymes > 0:
                    print(f"  {textes_anonymes} segments anonymisés (langue non reconnue)")
                
                return True
            else:
                print(f"ERREUR: Fichier Whisper non trouvé pour '{os.path.basename(chemin_audio)}'")
                return False
        else:
            print(f"ERREUR Whisper pour '{os.path.basename(chemin_audio)}'. Code: {resultat.returncode}")
            if resultat.stderr:
                print(f"Détails: {resultat.stderr.strip()}")
            return False

    except FileNotFoundError:
        print(f"Erreur: 'whisper' n'est pas installé.", file=sys.stderr)
        print(f"Installez avec: pip install openai-whisper", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Erreur inattendue pour '{chemin_audio}': {e}", file=sys.stderr)
        return False

def anonymiser_texte_si_necessaire(texte, confiance):
    """Détermine si le texte doit être anonymisé basé sur la confiance et le contenu"""
    
    # Si confiance très faible, c'est probablement pas du français
    if confiance < -0.8:
        return f"[PAROLE INCONNUE - {len(texte.split())} mots]"
    
    # Vérifier si ça ressemble à du français
    mots_francais_communs = [
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 'ce', 'que', 'qui',
        'il', 'elle', 'nous', 'vous', 'ils', 'avec', 'pour', 'dans', 'sur', 'par', 'pas',
        'tout', 'tous', 'bien', 'aussi', 'plus', 'très', 'comme', 'alors', 'mais', 'ou'
    ]
    
    mots_texte = texte.lower().split()
    if len(mots_texte) >= 3:
        mots_francais_trouves = sum(1 for mot in mots_texte if any(fr in mot for fr in mots_francais_communs))
        ratio_francais = mots_francais_trouves / len(mots_texte)
        
        # Si moins de 20% des mots ressemblent au français, anonymiser
        if ratio_francais < 0.2:
            return f"[PAROLE AUTRE LANGUE - {len(mots_texte)} mots]"
    
    # Vérifier des patterns suspects (beaucoup de caractères répétés, etc.)
    if len(set(texte.replace(' ', ''))) < len(texte) / 4:  # Trop de répétitions
        return f"[PAROLE INDÉCHIFFRABLE - {len(mots_texte)} mots]"
    
    # Si tout semble normal, garder le texte original
    return texte

def detecter_langue_probable(texte):
    """Essaie de deviner la langue basée sur des indices simples"""
    
    if not texte.strip():
        return "silence"
    
    # Indices français
    mots_francais = ['le', 'la', 'de', 'et', 'est', 'que', 'qui', 'pour', 'avec', 'dans']
    if any(mot in texte.lower() for mot in mots_francais):
        return "français_probable"
    
    # Indices anglais
    mots_anglais = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but']
    if any(mot in texte.lower() for mot in mots_anglais):
        return "anglais_probable"
    
    # Caractères non-latins
    if any(ord(c) > 127 for c in texte):
        return "langue_non_latine"
    
    return "langue_inconnue"

def main():
    parser = argparse.ArgumentParser(
        description=f"Utilise Whisper pour détecter les segments de parole et anonymise le texte non reconnu.",
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

    print(f"\nAnalyse Whisper pour {len(fichiers_audio)} fichiers...")
    print(f"Modèle: {MODELE}")
    print(f"Mode: Détection + texte anonymisé si langue non reconnue")
    print("-" * 60)

    nombre_succes = 0
    nombre_erreurs = 0

    for fichier_audio in fichiers_audio:
        if detecter_parole_whisper(fichier_audio, repertoire_sortie):
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