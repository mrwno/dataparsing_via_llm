"""
Script de test pour valider l'optimisation GEPA avec le modèle local.
"""
import time
import torch
from standar_gepa import load_standardized_dataset, LOCAL_MODEL_ID

def test_local_gepa_pipeline():
    # 1. Configuration du test
    dataset_name = "glue"
    config_name = "sst2"
    
    print(f"\n{'='*60}")
    print(f"🧬 TEST GEPA: Démarrage de l'optimisation sur {dataset_name}/{config_name}")
    print(f"🤖 Modèle Local: {LOCAL_MODEL_ID}")
    print(f"⚙️  GPU Disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Nom GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        # 2. Appel de la fonction avec use_local_llm=True
        # Cela va déclencher :
        # - Le chargement du modèle (peut prendre 10-30s)
        # - La boucle GEPA (Evaluation -> Réflexion -> Mutation)
        result = load_standardized_dataset(
            dataset_name=dataset_name, 
            config=config_name, 
            instruction="This is a sentiment analysis dataset.",
            use_local_llm=True 
        )

        end_time = time.time()
        duration = end_time - start_time

        # 3. Affichage des résultats
        print(f"\n{'='*60}")
        print("✅ RÉSULTAT DU TEST")
        print(f"{'='*60}")
        print(f"⏱️  Durée totale: {duration:.2f} secondes")
        
        # Le score interne calculé sur les 5 premiers samples du vrai dataset
        print(f"📊 Score (Validation interne): {result['score']:.2f} / 1.0") 
        
        print("\n🗺️  MAPPING DÉDUIT :")
        print(result['mapping'])
        
        print("\n💻 CODE GÉNÉRÉ (Unitxt) :")
        print(result['code'])
        
        # Vérification basique
        if result['score'] > 0.5:
            print("\n🎉 SUCCÈS : Le modèle a réussi à mapper les colonnes correctement.")
        else:
            print("\n⚠️  ATTENTION : Le score est faible. L'optimisation a peut-être échoué.")

    except ImportError as e:
        print(f"\n❌ ERREUR D'IMPORT : {e}")
        print("Assurez-vous d'avoir installé : transformers, torch, accelerate, bitsandbytes")
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE PENDANT LE TEST :")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_gepa_pipeline()