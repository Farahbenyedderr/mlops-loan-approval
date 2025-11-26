import pandas as pd
import numpy as np
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def test_prepare_data():
    """Test individuel de la fonction prepare_data()"""
    print("Test 1: prepare_data()")
    
    # Chargement des données
    df = pd.read_csv("loan_data.csv")
    print(f"   Données brutes: {df.shape}")
    
    # Test de la fonction
    df_clean = prepare_data(df)
    
    # Vérifications
    assert isinstance(df_clean, pd.DataFrame), "Doit retourner un DataFrame"
    assert not df_clean.empty, "Le DataFrame ne doit pas être vide"
    assert 'loan_status' in df_clean.columns, "Doit contenir la colonne cible"
    
    print("   ✅ prepare_data() - SUCCÈS")
    return df_clean

def test_train_model(df_clean):
    """Test individuel de la fonction train_model()"""
    print("Test 2: train_model()")
    
    # Test de la fonction
    model, X_test, y_test, X_train, y_train = train_model(df_clean)
    
    # Vérifications
    assert model is not None, "Le modèle ne doit pas être None"
    assert hasattr(model, 'predict'), "Le modèle doit avoir une méthode predict"
    assert len(X_test) > 0, "Les données de test ne doivent pas être vides"
    assert len(y_test) > 0, "Les labels de test ne doivent pas être vides"
    
    print("   ✅ train_model() - SUCCÈS")
    return model, X_test, y_test

def test_evaluate_model(model, X_test, y_test):
    """Test individuel de la fonction evaluate_model()"""
    print("Test 3: evaluate_model()")
    
    # Test de la fonction
    metrics = evaluate_model(model, X_test, y_test)
    
    # Vérifications
    required_keys = ['accuracy', 'auc_score', 'classification_report', 'confusion_matrix']
    for key in required_keys:
        assert key in metrics, f"Les métriques doivent contenir: {key}"
    
    assert 0 <= metrics['accuracy'] <= 1, "L'accuracy doit être entre 0 et 1"
    assert 0 <= metrics['auc_score'] <= 1, "L'AUC doit être entre 0 et 1"
    
    print("   ✅ evaluate_model() - SUCCÈS")
    return metrics

def test_save_load_model(model):
    """Test individuel des fonctions save_model() et load_model()"""
    print("Test 4: save_model() et load_model()")
    
    # Test sauvegarde
    save_path = save_model(model, "models/test_model.pkl")
    assert os.path.exists(save_path), "Le fichier modèle doit être créé"
    
    # Test chargement
    loaded_model = load_model(save_path)
    assert loaded_model is not None, "Le modèle chargé ne doit pas être None"
    
    print("   ✅ save_model() et load_model() - SUCCÈS")

def test_all_functions():
    """Test complet de toutes les fonctions du pipeline"""
    print("DÉMARRAGE DES TESTS COMPLETS\n")
    
    try:
        # Test 1: Préparation des données
        df_clean = test_prepare_data()
        
        # Test 2: Entraînement du modèle
        model, X_test, y_test = test_train_model(df_clean)
        
        # Test 3: Évaluation du modèle
        metrics = test_evaluate_model(model, X_test, y_test)
        
        # Test 4: Sauvegarde/Chargement
        test_save_load_model(model)
        
        print(f"\nTOUS LES TESTS RÉUSSIS!")
        print(f"Performance du modèle:")
        print(f"   - Précision: {metrics['accuracy']:.2%}")
        print(f"   - Score AUC: {metrics['auc_score']:.3f}")
        
    except Exception as e:
        print(f"ÉCHEC DU TEST: {e}")
        raise

if __name__ == "__main__":
    test_all_functions()