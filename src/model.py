import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import shutil

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report , log_loss ,average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.outliers import OutlierTrimmer
import warnings
warnings.filterwarnings("ignore")

class LoanRiskModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def prepare_data(self, df, target_column='loan_status'):
        """
        Préparation des données avec gestion des outliers et encodage
        """
        print("🔧 Préparation des données...")
        
        # Créer une copie pour éviter les modifications sur l'original
        df_processed = df.copy()
        
        # Vérification de la colonne cible
        if target_column not in df_processed.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée")
        
        # Conversion de l'âge en entier
        df_processed['person_age'] = df_processed['person_age'].astype('int')
        
        # Colonnes pour scaling
        skewed_cols = ['person_age', 'person_income', 'person_emp_exp',
                      'loan_amnt', 'loan_percent_income',
                      'cb_person_cred_hist_length', 'credit_score']
        
        norm_cols = ['loan_int_rate']
        
        # Vérification que les colonnes existent
        skewed_cols = [col for col in skewed_cols if col in df_processed.columns]
        norm_cols = [col for col in norm_cols if col in df_processed.columns]
        
        # Scaling des données
        if skewed_cols:
            df_processed[skewed_cols] = self.scaler.fit_transform(df_processed[skewed_cols])
        
        if norm_cols:
            df_processed[norm_cols] = self.min_max_scaler.fit_transform(df_processed[norm_cols])
        
        # Encodage des variables catégorielles
        df_processed = self._encode_categorical_features(df_processed)
        
        # Gestion des outliers (variables numériques seulement)
        numerical_vars_for_outliers = [
            'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
            'credit_score'
        ]
        
        numerical_vars_for_outliers = [var for var in numerical_vars_for_outliers if var in df_processed.columns]
        
        if numerical_vars_for_outliers:
            trimmer = OutlierTrimmer(
                capping_method='iqr', 
                tail='both',
                variables=numerical_vars_for_outliers
            )
            df_processed = trimmer.fit_transform(df_processed)
            print(f"Outliers traités sur {len(numerical_vars_for_outliers)} variables")
        
        # Sauvegarde des noms de features
        self.feature_names = [col for col in df_processed.columns if col != target_column]
        
        print(f" Préparation terminée. Shape finale: {df_processed.shape}")
        return df_processed
    
    def _encode_categorical_features(self, df):
        """Encodage des features catégorielles"""
        df_encoded = df.copy()
        
        # Encodage de l'éducation
        education_mapping = {
            'High School': 0,
            'Associate': 1, 
            'Bachelor': 2,
            'Master': 3,
            'Doctorate': 4
        }
        
        if 'person_education' in df_encoded.columns:
            df_encoded['person_education'] = df_encoded['person_education'].map(education_mapping)
        
        # Encodage des autres variables catégorielles
        gender_mapping = {'male': 0, 'female': 1}
        home_ownership_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
        loan_intent_mapping = {
            'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 
            'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5
        }
        previous_loan_defaults_mapping = {'No': 0, 'Yes': 1}
        
        # Application des mappings
        if 'person_gender' in df_encoded.columns:
            df_encoded['person_gender'] = df_encoded['person_gender'].map(gender_mapping)
        
        if 'person_home_ownership' in df_encoded.columns:
            df_encoded['person_home_ownership'] = df_encoded['person_home_ownership'].map(home_ownership_mapping)
        
        if 'loan_intent' in df_encoded.columns:
            df_encoded['loan_intent'] = df_encoded['loan_intent'].map(loan_intent_mapping)
        
        if 'previous_loan_defaults_on_file' in df_encoded.columns:
            df_encoded['previous_loan_defaults_on_file'] = df_encoded['previous_loan_defaults_on_file'].map(previous_loan_defaults_mapping)
        
        return df_encoded
    
    def split_data(self, df, target_column='loan_status', test_size=0.2, random_state=42):
        """Split des données avec stratification"""
        if target_column not in df.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f" Split des données: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, optimization=True):
        """Entraînement LightGBM avec optimisation"""
        print(" Entraînement du modèle LightGBM...")
        
        if optimization:
            best_params = self._optimize_hyperparameters(X_train, y_train)
            # CORRECTION: Vérifier si best_params est None APRÈS l'optimisation
            if best_params is None:
                print("  Optimisation échouée, utilisation des paramètres par défaut")
                best_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'num_leaves': 31,
                    'random_state': 42
                }
        else:
            best_params = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 7,
                'num_leaves': 50,
                'subsample': 0.9,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
    
        # AJOUT: Paramètres pour réduire les warnings LightGBM
        best_params.update({
            'verbosity': -1,
            'force_col_wise': True,
        })
        
        print(f" Paramètres utilisés: {best_params}")
        
        self.model = lgb.LGBMClassifier(**best_params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        if self.feature_names is None:
            self.feature_names = X_train.columns.tolist()
            print(f"✅ Feature names définis: {len(self.feature_names)} features")
        print(" Modèle entraîné avec succès")
        return best_params
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """Optimisation robuste avec Grid Search"""
        print("🎯 Optimisation des hyperparamètres...")
        
        # Grid de paramètres robuste
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'num_leaves': [15, 31, 50],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        best_score = -1
        best_params = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Compteur pour suivre le progrès
        total_combinations = (len(param_grid['n_estimators']) * 
                            len(param_grid['learning_rate']) * 
                            len(param_grid['max_depth']) * 
                            len(param_grid['num_leaves']) * 
                            len(param_grid['subsample']) * 
                            len(param_grid['colsample_bytree']))
        current_combination = 0
        
        print(f" Test de {total_combinations} combinaisons...")
        
        # Grid Search manuel
        for n_est in param_grid['n_estimators']:
            for lr in param_grid['learning_rate']:
                for depth in param_grid['max_depth']:
                    for leaves in param_grid['num_leaves']:
                        for subsample in param_grid['subsample']:
                            for colsample in param_grid['colsample_bytree']:
                                current_combination += 1
                                
                                # Paramètres avec configuration robuste
                                params = {
                                    'n_estimators': n_est,
                                    'learning_rate': lr,
                                    'max_depth': depth,
                                    'num_leaves': leaves,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample,
                                    'random_state': 42,
                                    'verbosity': -1,
                                    'force_col_wise': True
                                }
                                
                                try:
                                    model = lgb.LGBMClassifier(**params)
                                    # Utiliser n_jobs=1 pour éviter les problèmes de parallélisme
                                    scores = cross_val_score(model, X_train, y_train, 
                                                           cv=cv, scoring='roc_auc', n_jobs=1)
                                    mean_score = np.mean(scores)
                                    
                                    if current_combination % 10 == 0:
                                        print(f" Progression: {current_combination}/{total_combinations} | Meilleur score: {best_score:.4f}")
                                    
                                    if mean_score > best_score:
                                        best_score = mean_score
                                        best_params = params.copy()  # Important: faire une copie
                                        print(f" Nouveau meilleur score: {best_score:.4f}")
                                        print(f"   Paramètres: {best_params}")
                                        
                                except Exception as e:
                                    # Continuer avec la combinaison suivante en cas d'erreur
                                    continue
        
        # Vérifier si on a trouvé des paramètres valides
        if not best_params:
            print(" Aucune combinaison de paramètres n'a fonctionné")
            return None
        
        print(f" Optimisation terminée - Meilleur score: {best_score:.4f}")
        print(f" Meilleurs paramètres: {best_params}")
        return best_params
    
    def evaluate_model(self, X_test, y_test):
        """Évaluation complète du modèle"""
        if not self.is_fitted:
            raise ValueError("❌ Modèle non entraîné!")
        
        print("📊 Évaluation du modèle...")
        
        # VÉRIFICATION CRITIQUE : S'assurer que feature_names existe
        if self.feature_names is None:
            print("⚠️  Feature names non définis, utilisation des noms de colonnes")
            self.feature_names = X_test.columns.tolist()
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'log_loss': log_loss(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        # Feature importance avec vérification renforcée
        feature_importance = {}
        if (hasattr(self.model, 'feature_importances_') and 
            self.feature_names is not None and 
            len(self.feature_names) == len(self.model.feature_importances_)):
            
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            print("⚠️  Impossible de calculer feature importance")
        
        return {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_model(self, filepath='models/loan_risk_model.pkl'):
        """Sauvegarde du modèle et des preprocesseurs"""
        os.makedirs('models', exist_ok=True)
        
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'min_max_scaler': self.min_max_scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_artifacts, filepath)
        print(f" Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath='models/loan_risk_model.pkl'):
        """Chargement du modèle et des preprocesseurs"""
        model_artifacts = joblib.load(filepath)
        
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.min_max_scaler = model_artifacts['min_max_scaler']
        self.feature_names = model_artifacts['feature_names']
        self.is_fitted = model_artifacts['is_fitted']
        
        print(f" Modèle chargé: {filepath}")
    
    def predict(self, X):
        """Prédiction sur de nouvelles données"""
        if not self.is_fitted:
            raise ValueError(" Modèle non chargé!")
        
        return self.model.predict(X), self.model.predict_proba(X)

def get_latest_prepared_file():
    """Récupère le dernier fichier préparé dans le dossier results"""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        return None
    
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not csv_files:
        return None
    
    return os.path.join(results_dir, csv_files[0])

def check_prepared_data_exists():
    """Vérifie si des données préparées existent"""
    prepared_file = get_latest_prepared_file()
    if prepared_file is None:
        print("  ATTENTION: Aucun fichier préparé trouvé dans le dossier 'results/'")
        print(" Conseil: Exécutez d'abord 'make prepare' ou 'python main.py --mode prepare'")
        return False
    return True