from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1,
    use_label_encoder=False,
    random_state=42
)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            try:
                df = pd.read_csv(file)
                
                X = df.drop(columns=['subject_id', 'label'])
                y = df['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                model.fit(X_train, y_train)

                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred_binary = (y_prob > 0.5).astype(int)
           
                auroc = roc_auc_score(y_test, y_prob)
                precision_values, recall_values, _ = precision_recall_curve(y_test, y_prob)
                auprc = auc(recall_values, precision_values)
        
                precision = precision_score(y_test, y_pred_binary)
                recall = recall_score(y_test, y_pred_binary)

                results = {
                    'Precision': f'{precision:.4f}',
                    'Recall': f'{recall:.4f}',
                    'AUROC': f'{auroc:.4f}',
                    'AUPRC': f'{auprc:.4f}'
                }

                return render_template('index.html', results=results)
            except Exception as e:
                return f"An error occurred: {e}"
    
    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)