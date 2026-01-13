import lime
import lime.lime_tabular

def explain_prediction(clf, X_train, feature_names, instance):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Human', 'AI'],
        discretize_continuous=True
    )
    exp = explainer.explain_instance(instance, clf.predict_proba, num_features=10)
    return exp
