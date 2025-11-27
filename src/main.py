import pandas as pd
import argparse
import sys
import numpy as np
import os
import glob
from model import LoanRiskModel, get_latest_prepared_file, check_prepared_data_exists

def setup_directories():
    """Crée la structure de dossiers nécessaire"""
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

def clear_results_folder():
    """Vide le dossier results avant une nouvelle préparation"""
    results_dir = 'results'
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            file_path = os.path.join(results_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                print(f"Supprimé: {file_path}")
            except Exception as e:
                print(f"Erreur suppression {file_path}: {e}")
    else:
        os.makedirs(results_dir)

def prepare_data(data_path='data/loan_data.csv', output_filename='prepared_data.csv'):
    """
    Fonction pour la préparation des données seulement
    Utilise le chemin relatif par défaut
    """
    print("=" * 50)
    print(" PRÉPARATION DES DONNÉES SEULEMENT")
    print("=" * 50)
    
    try:
        # Setup des dossiers
        setup_directories()
        
        # Vérification que le fichier source existe
        if not os.path.exists(data_path):
            print(f" Fichier source non trouvé: {data_path}")
            print(" Placez votre fichier 'loan_data.csv' dans le dossier 'data/'")
            sys.exit(1)
        
        # Nettoyage du dossier results
        print(" Nettoyage du dossier results...")
        clear_results_folder()
        
        # Chargement des données
        print(f" Chargement des données: {data_path}")
        df = pd.read_csv(data_path)
        print(f" Données brutes: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Initialisation du modèle
        model = LoanRiskModel()
        
        # Préparation des données
        df_processed = model.prepare_data(df)
        
        # Sauvegarde dans results
        output_path = os.path.join('results', output_filename)
        df_processed.to_csv(output_path, index=False)
        print(f" Données préparées sauvegardées: {output_path}")
        
        # Affichage des informations
        print("\n RÉSUMÉ DE LA PRÉPARATION:")
        print(f"  • Shape initiale: {df.shape}")
        print(f"  • Shape finale: {df_processed.shape}")
        print(f"  • Colonnes numériques: {len(df_processed.select_dtypes(include=[np.number]).columns)}")
        print(f"  • Colonnes totales: {len(df_processed.columns)}")
        print(f"  • Fichier sauvegardé: {output_path}")
        
        return df_processed
        
    except Exception as e:
        print(f" Erreur lors de la préparation: {e}")
        sys.exit(1)

def train_model(data_path=None, model_path='models/loan_risk_model.pkl', optimize=True):
    """
    Fonction pour l'entraînement seulement 
    Utilise le fichier préparé du dossier results si data_path n'est pas spécifié
    """
    print("=" * 50)
    print(" ENTRAÎNEMENT DU MODÈLE")
    print("=" * 50)
    
    try:
        # Détermination du fichier d'entrée
        if data_path is None:
            # Utilisation du fichier préparé dans results
            data_path = get_latest_prepared_file()
            if data_path is None:
                print(" Aucun fichier préparé trouvé dans 'results/'")
                print(" Exécutez d'abord: python main.py --mode prepare")
                print(" Ou: make prepare")
                sys.exit(1)
            print(f" Utilisation du fichier préparé: {data_path}")
        else:
            print(f" Utilisation du fichier spécifié: {data_path}")
        
        # Chargement des données
        df = pd.read_csv(data_path)
        print(f" Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Vérification de la colonne cible
        if 'loan_status' not in df.columns:
            print(" Colonne cible 'loan_status' non trouvée dans les données")
            sys.exit(1)
        
        # Initialisation du modèle
        model = LoanRiskModel()
        
        # Split des données
        X_train, X_test, y_train, y_test = model.split_data(df)
        
        # Entraînement du modèle
        best_params = model.train_model(X_train, y_train, optimization=optimize)
        print(f" Modèle entraîné avec paramètres: {best_params}")
        
        # Évaluation
        evaluation = model.evaluate_model(X_test, y_test)
        
        # Affichage des résultats
        print("\n RÉSULTATS DE L'ÉVALUATION")
        print("-" * 30)
        for metric, value in evaluation['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        # Feature importance
        if evaluation['feature_importance']:
            print("\n TOP 5 FEATURES IMPORTANTES")
            print("-" * 30)
            top_features = sorted(evaluation['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                print(f"  {feature}: {importance:.4f}")
        
        # Sauvegarde du modèle
        model.save_model(model_path)
        
        print(f"\n ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f" Modèle sauvegardé: {model_path}")
        
        return model, evaluation
        
    except Exception as e:
        print(f" Erreur lors de l'entraînement: {e}")
        sys.exit(1)

def evaluate_model(data_path=None, model_path='models/loan_risk_model.pkl'):
    """
    Évaluation seulement du modèle
    """
    print("=" * 50)
    print(" ÉVALUATION DU MODÈLE")
    print("=" * 50)
    
    try:
        # Détermination du fichier d'entrée
        if data_path is None:
            data_path = get_latest_prepared_file()
            if data_path is None:
                print(" Aucun fichier préparé trouvé dans 'results/'")
                print(" Exécutez d'abord: python main.py --mode prepare")
                sys.exit(1)
            print(f" Utilisation du fichier préparé: {data_path}")
        
        # Chargement des données et du modèle
        df = pd.read_csv(data_path)
        model = LoanRiskModel()
        model.load_model(model_path)
        
        # Split et évaluation
        X_train, X_test, y_train, y_test = model.split_data(df)
        evaluation = model.evaluate_model(X_test, y_test)
        
        # Affichage détaillé
        print("\n RAPPORT DÉTAILLÉ")
        print("-" * 30)
        for metric, value in evaluation['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        return evaluation
        
    except Exception as e:
        print(f"Erreur lors de l'évaluation: {e}")
        sys.exit(1)

def predict_new_data(model_path='models/loan_risk_model.pkl', input_data_path=None):
    """
    Prédiction sur de nouvelles données
    """
    print("=" * 50)
    print(" PRÉDICTION SUR NOUVELLES DONNÉES")
    print("=" * 50)
    
    try:
        # Vérification du modèle
        if not os.path.exists(model_path):
            print(f" Modèle non trouvé: {model_path}")
            print("Exécutez d'abord l'entraînement: python main.py --mode train")
            sys.exit(1)
        
        # Vérification des données d'entrée
        if input_data_path is None:
            print(" Chemin des données d'entrée non spécifié")
            sys.exit(1)
        
        if not os.path.exists(input_data_path):
            print(f" Fichier de données non trouvé: {input_data_path}")
            sys.exit(1)
        
        # Chargement du modèle et des données
        model = LoanRiskModel()
        model.load_model(model_path)
        
        df_new = pd.read_csv(input_data_path)
        print(f" Données à prédire: {df_new.shape[0]} lignes")
        
        # Préparation des nouvelles données (sans la target)
        df_processed = model.prepare_data(df_new)
        
        # Prédiction
        predictions, probabilities = model.predict(df_processed)
        
        # Création des résultats
        results = df_new.copy()
        results['prediction'] = predictions
        results['probability_risk'] = probabilities[:, 1]
        results['risk_category'] = results['probability_risk'].apply(
            lambda x: 'High Risk' if x > 0.7 else 'Medium Risk' if x > 0.3 else 'Low Risk'
        )
        
        # Sauvegarde des résultats
        output_path = 'results/predictions_results.csv'
        results.to_csv(output_path, index=False)
        
        # Statistiques des prédictions
        print(f"\n STATISTIQUES DES PRÉDICTIONS")
        print("-" * 30)
        print(f"  High Risk: {(results['risk_category'] == 'High Risk').sum()}")
        print(f"  Medium Risk: {(results['risk_category'] == 'Medium Risk').sum()}")
        print(f"  Low Risk: {(results['risk_category'] == 'Low Risk').sum()}")
        print(f"  Taux de risque moyen: {results['probability_risk'].mean():.2%}")
        
        print(f"\n Prédictions sauvegardées: {output_path}")
        
        return results
        
    except Exception as e:
        print(f" Erreur lors de la prédiction: {e}")
        sys.exit(1)

def main():
    """Fonction principale avec interface en ligne de commande"""
    parser = argparse.ArgumentParser(description='Loan Risk Prediction Model')
    parser.add_argument('--data', type=str, default='data/loan_data.csv',
                       help='Chemin vers le fichier de données CSV (défaut: data/loan_data.csv)')
    parser.add_argument('--mode', type=str, 
                       choices=['prepare', 'train', 'evaluate', 'predict', 'full'], 
                       required=True, 
                       help='Mode: prepare, train, evaluate, predict, full')
    parser.add_argument('--model_path', type=str, default='models/loan_risk_model.pkl',
                       help='Chemin pour sauvegarder/charger le modèle')
    parser.add_argument('--output_name', type=str, default='prepared_data.csv',
                       help='Nom du fichier de sortie pour les données préparées')
    parser.add_argument('--optimize', action='store_true',
                       help='Activer l\'optimisation des hyperparamètres')
    parser.add_argument('--input_data', type=str,
                       help='Chemin vers les données à prédire (mode predict seulement)')
    
    args = parser.parse_args()
    
    # Setup des dossiers
    setup_directories()
    
    if args.mode == 'prepare':
        prepare_data(args.data, args.output_name)
        
    elif args.mode == 'train':
        # Si --data est fourni, l'utiliser, sinon utiliser le fichier préparé
        train_model(args.data if args.data != 'data/loan_data.csv' else None, 
                        args.model_path, args.optimize)
        
    elif args.mode == 'evaluate':
        evaluate_model(args.data if args.data != 'data/loan_data.csv' else None, 
                           args.model_path)
        
    elif args.mode == 'predict':
        if not args.input_data:
            print(" Argument --input_data requis pour le mode predict")
            sys.exit(1)
        predict_new_data(args.model_path, args.input_data)
        
    elif args.mode == 'full':
        # Mode complet: préparation + entraînement
        print(" MODE COMPLET: PRÉPARATION + ENTRAÎNEMENT")
        
        # Préparation des données
        df_processed = prepare_data(args.data, args.output_name)
        
        # Entraînement du modèle (utilise automatiquement le fichier dans results)
        train_model(None, args.model_path, args.optimize)
            
        print(" Pipeline complet terminé avec succès!")

if __name__ == "__main__":
    main()