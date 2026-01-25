from sklearn.ensemble import RandomForestClassifier


def train_random_forest(x_train, y_train, params=None):
    """Train Random Forest optimized for catching defaults

    Args:
        x_train: Training features.
        y_train: Training labels.
        params: Dictionary of parameters for RandomForestClassifier.
                If None, uses a default set.

    Returns:
        Trained RandomForestClassifier model.
    """
    print("\n Training Random Forest Model...")
    print("=" * 50)

    # Calculate class weights to prioritize catching defaults (class 0)
    class_0_count = (y_train == 0).sum()
    class_1_count = (y_train == 1).sum()

    print(" Training set class distribution:")
    print(
        f"   Defaults (0): {class_0_count} ({class_0_count / len(y_train) * 100:.1f}%)"
    )
    print(
        f"   Approvals (1): {class_1_count} ({class_1_count / len(y_train) * 100:.1f}%)"
    )

    # Define default parameters if none are provided
    if params is None:
        class_weights = {0: 1.5, 1: 0.5}
        print(f"   Using default class weights: {class_weights}")

        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "class_weight": class_weights,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0,
        }
    else:
        print("   Using provided parameters.")
        # Ensure class weights are handled if provided in params
        if "class_weight" not in params:
            class_weights = {0: 1.5, 1: 0.5}  # Or your preferred default
            params["class_weight"] = class_weights
            print(f"   Using default class weights: {class_weights}")
        else:
            print(f"   Using class weights: {params['class_weight']}")

    # Initialize Random Forest with the chosen parameters
    rf_model = RandomForestClassifier(**params)

    # Train model
    print(" Training in progress...")
    rf_model.fit(x_train, y_train)
    print(" Model trained successfully!")
    return rf_model
