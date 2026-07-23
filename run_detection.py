import pandas as pd
import numpy as np
import os
import glob
import argparse
import scipy.io as sio
from datetime import datetime
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from src.MSCVAE import MSCVAE
from src.MSCVAE_v2 import MSCVAE_v2
from src.PCA import PCA
from src.ANN_AE import ANN_AE
from src.CNN_AE import CNN_AE
from src.LSTM_AE import LSTM_AE
from src.MSCRED import MSCRED
from src.OmniAnomaly import OmniAnomaly
from src.TranAD import TranAD
from src.USAD import USAD

def plot_predict(predictions, threshold, save_path):
    df = pd.DataFrame({
        "timestamp": predictions['timestamp'],
        "phi": predictions['phi'] / threshold
    })
    df["threshold"] = 1.0

    layout = go.Layout(plot_bgcolor="white", paper_bgcolor='white')
    fig = go.Figure(layout=layout)
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["phi"], mode="lines", 
        name="Índice (Phi)", fill="tozeroy", line_color="#0F293A"
    ))
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["threshold"], mode="lines", 
        name="Limiar", line_color="#FB8102"
    ))
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#CECFD1')
    
    fig.update_layout(
        hovermode='x unified', 
        legend=dict(orientation="h"),
        yaxis_range=[0, 4] 
    )
    fig.write_image(save_path, scale=2)

def plot_reconstruction(df_original, df_reconstruction, df_contribution, save_path):
    num_vars = len(df_contribution)
    if num_vars == 0:
        return
        
    vars_to_plot = list(df_contribution['VARIAVEL'])
    percentages = list(df_contribution['%'])
    
    subplot_titles = [f"{pct:.1f}% - Variável {var}" for var, pct in zip(vars_to_plot, percentages)]
    
    fig = make_subplots(
        rows=num_vars, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=subplot_titles
    )
    
    for i, var in enumerate(vars_to_plot):
        row_idx = i + 1
        
        fig.add_trace(go.Scatter(
            x=df_original.index, y=df_original[var], mode="lines", 
            line_color="#1f77b4", name=f"{var} (Original)"
        ), row=row_idx, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_reconstruction.index, y=df_reconstruction[var], mode="lines", 
            line_color="#d62728", line_dash="dot", name=f"{var} (Reconst)"
        ), row=row_idx, col=1)

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor='white', hovermode='x unified',
        height=250 * num_vars,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#CECFD1')
    
    fig.write_image(save_path, scale=2)

def compute_metrics(true_labels, pred_labels, nome_base, dataset):
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0,1]).ravel()
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "dataset": dataset,
        "fault": nome_base,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

def run_experiment(dataset, gain, epochs, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = f"output/{dataset}_{timestamp}_{model_name}"
    os.makedirs(main_output_dir, exist_ok=True)
    
    metrics_list = []
    if model_name == "MSCVAE":
        model = MSCVAE()
    elif model_name == "MSCVAE_v2":
        model = MSCVAE_v2()
    elif model_name == "PCA":
        model = PCA()
    elif model_name == "ANN_AE":
        model = ANN_AE()
    elif model_name == "CNN_AE":
        model = CNN_AE()
    elif model_name == "LSTM_AE":
        model = LSTM_AE()
    elif model_name == "MSCRED":
        model = MSCRED()
    elif model_name == "OmniAnomaly":
        model = OmniAnomaly()
    elif model_name == "TranAD":
        model = TranAD()
    elif model_name == "USAD":
        model = USAD()
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    if dataset == "CSTR":
        dir_data = "data/CSTR_data/"
        
        # Load Normal Data
        mat_normal_path = os.path.join(dir_data, "normal.mat")
        csv_normal_path = os.path.join(dir_data, "normal.csv")
        
        if os.path.exists(mat_normal_path):
            data_normal = sio.loadmat(mat_normal_path)
            df_normal = pd.DataFrame(data_normal["testdata"], columns=[c.strip() for c in data_normal["columns"]])
        else:
            df_normal = pd.read_csv(csv_normal_path, sep=";", decimal=".")
            if "CLASS" in df_normal.columns:
                df_normal = df_normal.drop(columns=["CLASS"])
            
        print(f"Training {model_name} on CSTR normal data with gain {gain} for {epochs} epochs...")
        model.fit([df_normal], gain=gain, epochs=epochs, verbose=True)
        
        df_sistema = pd.read_csv(os.path.join(dir_data, "CSTR_subsistema.csv"), sep=";")
        
        # Look for fault files (.mat preferred)
        fault_files = glob.glob(os.path.join(dir_data, "falha*.mat"))
        if not fault_files:
            fault_files = glob.glob(os.path.join(dir_data, "falha*.csv"))
        
        for file in fault_files:
            nome_base = os.path.splitext(os.path.basename(file))[0]
            print(f"Processing {nome_base}...")
            
            if file.endswith('.mat'):
                data = sio.loadmat(file)
                df_falha = pd.DataFrame(data["testdata"], columns=[c.strip() for c in data["columns"]])
                raw_labels = data["labels"]
                # raw_labels may be an array of strings like 'normal', 'normal', 'S17', ...
                true_labels = np.array([0 if 'normal' in str(l) else 1 for l in raw_labels])
            else:
                df = pd.read_csv(file, sep=";", decimal=".")
                true_labels = np.where(df["CLASS"] == 'normal', 0, 1)
                df_falha = df.drop(columns=["CLASS"])
            
            predictions = model.predict(df_falha)
            threshold = model.threshold
            pred_labels = (predictions['phi'] > threshold).astype(int)
            
            true_labels = true_labels[-len(pred_labels):]
            metrics = compute_metrics(true_labels, pred_labels, nome_base, "CSTR")
            metrics_list.append(metrics)
            
            plot_predict(predictions, threshold, os.path.join(main_output_dir, f"{nome_base}_predict.png"))
            
            contribution, df_reconstruction = model.contribution(df_falha, df_sistema, top_k=None)
            df_contribution = pd.DataFrame().from_dict(contribution).head(5) # Top 5 to save space/time
            plot_reconstruction(df_falha, df_reconstruction, df_contribution, os.path.join(main_output_dir, f"{nome_base}_reconstruction.png"))
            
    elif dataset == "TE":
        dir_data = "data/TE_data/"
        data_normal = sio.loadmat(os.path.join(dir_data, "fault_00.mat"))
        df_normal = pd.DataFrame(data_normal["trainingdata"])
        
        print(f"Training {model_name} on TE normal data with gain {gain} for {epochs} epochs...")
        model.fit([df_normal], gain=gain, epochs=epochs, verbose=True)
        
        df_sistema = pd.DataFrame({
            "VARIAVEL": list(range(52)),
            "DESC": [f"Var {i}" for i in range(52)],
            "SISTEMA": ["TE"] * 52
        })
        
        fault_files = glob.glob(os.path.join(dir_data, "fault_*.mat"))
        
        for file in fault_files:
            nome_base = os.path.splitext(os.path.basename(file))[0]
            if "00" in nome_base:
                continue # fault_00 is normal data for TE usually
            
            print(f"Processing {nome_base}...")
            
            data = sio.loadmat(file)
            df_falha = pd.DataFrame(data["testdata"])
            
            true_labels = np.ones(len(df_falha))
            if len(df_falha) > 160:
                true_labels[:160] = 0
                    
            predictions = model.predict(df_falha)
            threshold = model.threshold
            pred_labels = (predictions['phi'] > threshold).astype(int)
            
            true_labels = true_labels[-len(pred_labels):]
            metrics = compute_metrics(true_labels, pred_labels, nome_base, "TE")
            metrics_list.append(metrics)
            
            plot_predict(predictions, threshold, os.path.join(main_output_dir, f"{nome_base}_predict.png"))
            
            contribution, df_reconstruction = model.contribution(df_falha, df_sistema, top_k=None)
            df_contribution = pd.DataFrame().from_dict(contribution).head(5) # Top 5
            plot_reconstruction(df_falha, df_reconstruction, df_contribution, os.path.join(main_output_dir, f"{nome_base}_reconstruction.png"))
    
    metrics_path = os.path.join(main_output_dir, f"metrics_{dataset}.json")
    
    experiment_data = {
        "execution_params": {
            "dataset": dataset,
            "model": model_name,
            "gain": gain,
            "epochs": epochs,
            "timestamp": datetime.now().isoformat()
        },
        "results": metrics_list
    }
    
    with open(metrics_path, "w") as f:
        json.dump(experiment_data, f, indent=4)
        
    print(f"Processamento concluído. Resultados salvos em {main_output_dir}")
    print(f"Métricas salvas em: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run anomaly detection on CSTR or TE dataset.')
    parser.add_argument('--dataset', type=str, choices=['CSTR', 'TE'], required=True, help='Dataset to use (CSTR or TE)')
    parser.add_argument('--gain', type=float, default=1.0, help='Threshold gain multiplier')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--model', type=str, choices=['MSCVAE', 'MSCVAE_v2', 'PCA', 'ANN_AE', 'CNN_AE', 'LSTM_AE', 'MSCRED', 'OmniAnomaly', 'TranAD', 'USAD'], default='MSCVAE', help='Model to use')
    args = parser.parse_args()
    
    gain = args.gain
    epochs = args.epochs if args.epochs is not None else (50 if args.dataset == 'CSTR' else 100)
    
    run_experiment(args.dataset, gain, epochs, args.model)
