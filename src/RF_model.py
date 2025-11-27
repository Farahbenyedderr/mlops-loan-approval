from sklearn.ensemble import RandomForestClassifier



def train_random_forest(x_train, y_train):
    
    """Train Random Forest optimized for catching defaults"""
    print("\n🌲 Training Random Forest Model...")
    print("="*50)
    
    # Calculate class weights to prioritize catching defaults (class 0)
    class_0_count = (y_train == 0).sum()
    class_1_count = (y_train == 1).sum()
    
    # Custom weights: penalize missing defaults more heavily
    class_weights = {
        0: 1.5,  # Higher weight on defaults
        1: 0.5   # Lower weight on approvals
    }
    
    print(f"📊 Training set class distribution:")
    print(f"   Defaults (0): {class_0_count} ({class_0_count/len(y_train)*100:.1f}%)")
    print(f"   Approvals (1): {class_1_count} ({class_1_count/len(y_train)*100:.1f}%)")
    print(f"   Using class weights: {class_weights}")
    
    # Initialize Random Forest with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,           # Number of trees
        max_depth=15,                # Prevent overfitting
        min_samples_split=20,        # Minimum samples to split
        min_samples_leaf=10,         # Minimum samples per leaf
        max_features='sqrt',         # Features per tree
        class_weight=class_weights,  # Prioritize catching defaults
        random_state=42,
        n_jobs=-1,                   # Use all CPU cores
        verbose=0
    )
    
    # Train model
    print("\n🔄 Training in progress...")
    rf_model.fit(x_train, y_train)
    print("✅ Model trained successfully!")
    
    
    
    return rf_model