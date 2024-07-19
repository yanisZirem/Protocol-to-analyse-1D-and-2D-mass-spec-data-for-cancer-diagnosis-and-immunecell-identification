# Necessary libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.discriminant_analysis import *
import importlib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
import joblib
import eli5

# Function to plot average spectra for each class
def plot_average_spectra(data, class_column='Class', threshold=None, colors=None):
    import plotly.graph_objects as go
    # Create a new plotly figure
    fig = go.Figure()
    
    # Get the unique class labels from the specified class column
    unique_classes = data[class_column].unique()
    
    # Generate default colors if none are provided
    if colors is None:
        colors = {class_label: f'rgb({i * 10}, {255 - i * 40}, {i * 20})'
                  for i, class_label in enumerate(unique_classes)}
    # Iterate over each unique class to compute and plot the mean spectrum
    for class_label in unique_classes:
        # Select data for the current class and drop the class column
        class_data = data[data[class_column] == class_label].drop(class_column, axis=1) 
        # Compute the mean spectrum for the current class
        mean_spectrum = class_data.mean()
        # Add the mean spectrum as a trace to the plotly figure
        fig.add_trace(go.Scatter(
            x=mean_spectrum.index, 
            y=mean_spectrum.values,
            mode='lines', 
            name=f'Class {class_label}',
            line=dict(color=colors.get(class_label, 'blue'))
        ))
    
    # Update the layout of the figure with titles and dimensions
    fig.update_layout(width=1000, xaxis_title='m/z', yaxis_title='Relative Intensities')
    # Update the x-axis to adjust tick angle and font size
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))  
    # Return the completed figure
    return fig

# Function to plot specific sample using its index
def plot_sample_spectrum(data, sample_index, color='orange'):
    import plotly.graph_objects as go
    # Check if the sample index is within the valid range
    if sample_index < 0 or sample_index >= len(data):
        raise ValueError(f"Sample index {sample_index} is out of bounds. It should be between 0 and {len(data)-1}.")

    # Extract the spectrum data for the specified sample
    sample_spectrum = data.iloc[sample_index]

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.columns, y=sample_spectrum.values, mode='lines', line=dict(color=color)))

    # Update the layout of the plot
    fig.update_layout(width=1000, xaxis_title='m/z', yaxis_title='relative intensities', showlegend=False)

    # Display the plot
    fig.show()


def Train_models(data, target_column='Class', test_size=0.2, random_state=1):
    from sklearn.model_selection import train_test_split
    from lazypredict.Supervised import LazyClassifier
    
    # Separate features and target
    y = data[target_column]
    X = data.drop([target_column], axis=1)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    
    # Initialize LazyClassifier
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    
    # Fit LazyClassifier to training data
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    return models, predictions

# Function to find and build the best model based on F1 score
def Find_and_build_best_model(data, models, specific_model=None):
    from sklearn.model_selection import train_test_split

    # Define the model mappings
    model_mapping = {
        'Perceptron': 'linear_model',
        'LGBMClassifier': 'lightgbm',
        'PassiveAggressiveClassifier': 'linear_model',
        'LinearSVC': 'svm',
        'RidgeClassifierCV': 'linear_model',
        'RidgeClassifier': 'linear_model',
        'ExtraTreeClassifier': 'tree',
        'ExtraTreesClassifier': 'ensemble',
        'SGDClassifier': 'linear_model',
        'CalibratedClassifierCV': 'calibration',
        'RandomForestClassifier': 'ensemble',
        'BaggingClassifier': 'ensemble',
        'KNeighborsClassifier': 'neighbors',
        'DecisionTreeClassifier': 'tree',
        'LogisticRegression': 'linear_model',
        'LinearDiscriminantAnalysis': 'discriminant_analysis',
        'NuSVC': 'svm',
        'SVC': 'svm',
        'GaussianNB': 'naive_bayes',
        'NearestCentroid': 'neighbors',
        'BernoulliNB': 'naive_bayes',
        'AdaBoostClassifier': 'ensemble',
        'QuadraticDiscriminantAnalysis': 'discriminant_analysis',
        'DummyClassifier': 'dummy',
        'LabelSpreading': 'semi_supervised',
        'LabelPropagation': 'semi_supervised'
    }

    y = data['Class']
    X = data.drop(['Class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)

    best_model_name = specific_model
    best_f1_score = -1

    if not specific_model:
        for model_name in models.index:
            f1_score = models.at[model_name, 'F1 Score']
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model_name = model_name

    if best_model_name:
        print("Best Classifier:", best_model_name)
        module_name = model_mapping.get(best_model_name)

        if module_name:
            try:
                model_module = importlib.import_module(f'sklearn.{module_name}')
                if hasattr(model_module, best_model_name):
                    best_model = getattr(model_module, best_model_name)()
                else:
                    raise ImportError
            except ImportError:
                print(f"Model {best_model_name} could not be imported from sklearn.")
                return None, None
        else:
            if best_model_name.startswith("LGBM"):
                import lightgbm as lgb
                best_model = getattr(lgb, best_model_name)()
            elif best_model_name.startswith("XGB"):
                import xgboost as xgb
                best_model = getattr(xgb, best_model_name)()
            else:
                print("Best Classifier not found.")
                return None, None

        pipeline = Pipeline([('scaler', StandardScaler()), (best_model_name, best_model)])
        pipeline.fit(X_train, y_train)
        return best_model_name, pipeline
    else:
        print("Best Classifier not found.")
        return None, None

# Function to display confusion matrix, scores, and classification report
def confusion_matrix_scores_classification_report(pipeline,data):
    from sklearn.model_selection import train_test_split

    y = data['Class']
    X = data.drop(['Class'], axis=1)
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)
    # Generate predictions on the test set
    y_pred = pipeline.predict(X_test)
    # Calculate the accuracy score
    score = pipeline.score(X_test, y_test)
    print('Accuracy:', score)
    # Print the classification report
    print(classification_report(y_test, y_pred))
    # Display the confusion matrix
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
    plt.figure(figsize=(10, 7))
    # Set the figure size for better visualization
    plt.show()
    
# Function for cross-validation and reporting results
def cross_validate_and_report(pipeline,data):
    import seaborn as sns

    y = data['Class']
    X = data.drop(['Class'], axis=1)

    # Define KFold cross-validator with 5 splits, shuffling enabled, and a fixed random state for reproducibility
    kfold = KFold(n_splits=20, shuffle=True, random_state=1)
    
    # Perform cross-validation and obtain scores
    cv_scores = cross_val_score(pipeline, X, y, cv=kfold)
    
    # Print cross-validation scores
    print('CV Scores:', cv_scores)
    
    # Print the mean of the cross-validation scores
    print('Mean CV Score:', cv_scores.mean())
    
    # Print the standard deviation of the cross-validation scores
    print('Std CV Score:', cv_scores.std())
    
    # Generate cross-validated predictions
    y_pred = cross_val_predict(pipeline, X, y, cv=kfold)
    
    # Print the classification report
    print(classification_report(y, y_pred))
    ConfusionMatrixDisplay.from_estimator(pipeline,X, y_pred)
    plt.figure(figsize=(10, 7))
    plt.show()

def eli5_feature_importance(pipeline, data, top_features=40):
    from sklearn.model_selection import train_test_split

    y = data['Class']
    X = data.drop(['Class'], axis=1)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)
    # Extract the model from the final step of the pipeline
    model = pipeline.named_steps[pipeline.steps[-1][0]]
    # Generate and display the feature importance using eli5
    sample_contribution = eli5.show_weights(
        model, 
        feature_names=X_train.columns.tolist(),  # List of feature names
        top=top_features,  # Number of top features to display
        feature_re='^.*$'  # Regular expression to match all feature names
    )
    
    # Return the HTML object containing the feature importance
    return sample_contribution

def save_contributions(csv_name, pipeline, data):
    from sklearn.model_selection import train_test_split

    y = data['Class']
    X = data.drop(['Class'], axis=1)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)
    # Extract the model from the final step of the pipeline
    model = pipeline.named_steps[pipeline.steps[-1][0]]
    # Initialize a list to store the contributions for each sample
    sample_contributions = []
    # Loop through each sample in the training data
    for idx in range(len(X_train.index)):
        # Generate the feature contribution for the current sample
        sample_contribution_df = eli5.explain_weights_df(
            model, 
            feature_names=X_train.columns.tolist(),  # List of feature names
            feature_re='^.*$'  # Regular expression to match all feature names
        )
        # Add the contribution dataframe to the list
        sample_contributions.append(sample_contribution_df)
    # Concatenate all the contributions into a single dataframe
    import pandas as pd
    all_contributions_df = pd.concat(sample_contributions)
    # Save the dataframe to a CSV file
    all_contributions_df.to_csv(csv_name, index=False)
