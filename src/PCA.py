import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

class PCA:
    """
    Baseline PCA model for Multivariate Time Series Anomaly Detection.
    Implements a compatible interface with MSCVAE.
    """
    def __init__(self, n_components=0.95):
        # n_components can be a float to explain a variance ratio, or an int for exact components
        self.n_components = n_components
        self.pca = SklearnPCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.threshold = None
        self.gain = 1.0

    def fit(self, train_dataframes, gain=1.0, epochs=None, verbose=True):
        self.gain = gain
        if isinstance(train_dataframes, pd.DataFrame):
            train_dataframes = [train_dataframes]
            
        full_train_df = pd.concat(train_dataframes, ignore_index=True)
        
        # Scale data
        X = self.scaler.fit_transform(full_train_df)
        
        # Fit PCA
        self.pca.fit(X)
        
        # Calculate SPE (Sum of Squared Prediction Errors)
        X_pred = self.pca.inverse_transform(self.pca.transform(X))
        reconstruction_error = np.sum((X - X_pred)**2, axis=1)
        
        # Compute threshold based on residual eigenvalues (Q-statistic / SPE limit)
        cov_matrix = np.cov(X, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        l = self.pca.n_components_
        lambda_rem = eigenvalues[l:]
        
        theta1 = np.sum(lambda_rem)
        theta2 = np.sum(np.square(lambda_rem))
        
        if theta1 > 1e-8 and theta2 > 1e-8:
            g_SPE = theta2 / theta1
            h_SPE = (theta1**2) / theta2
            alpha = 0.05
            base_threshold = g_SPE * chi2.ppf(1 - alpha, h_SPE)
        else:
            base_threshold = np.percentile(reconstruction_error, 99.9)
            
        self.threshold = base_threshold * self.gain
        
        if verbose:
            print(f"PCA fitted with {self.pca.n_components_} components.")
            print(f"Base Threshold: {base_threshold:.6f}")
            print(f"Gain: {self.gain}")
            print(f"Final Threshold: {self.threshold:.6f}")

    def predict(self, df_test, timestamps=None, batch_size=None):
        X = self.scaler.transform(df_test)
        X_pred = self.pca.inverse_transform(self.pca.transform(X))
        reconstruction_error = np.sum((X - X_pred)**2, axis=1)
        
        ts = timestamps if timestamps is not None else np.arange(len(df_test))
        if hasattr(ts, 'values'):
            ts = ts.values
            
        return {
            'timestamp': ts,
            'phi': reconstruction_error
        }

    def contribution(self, df_test, df_sistema, top_k=None):
        # Transform and reconstruct
        X = self.scaler.transform(df_test)
        X_pred = self.pca.inverse_transform(self.pca.transform(X))
        
        # Reconstruction in original scale
        X_pred_original = self.scaler.inverse_transform(X_pred)
        df_reconstruction = pd.DataFrame(X_pred_original, columns=df_test.columns, index=df_test.index)
        
        # Anomaly score per sample (SPE)
        phi = np.sum((X - X_pred)**2, axis=1)
        anom_mask = phi > self.threshold
        
        if not anom_mask.any():
            anom_mask[np.argmax(phi)] = True
            
        # Reconstruction-Based Contribution (RBC) for SPE
        P = self.pca.components_.T
        C = P @ P.T
        Ctil = np.eye(C.shape[0]) - C
        cii = np.diag(Ctil).copy()
        
        # Avoid division by zero
        cii[cii == 0] = 1e-10
        
        var_errors = (X - X_pred)**2
        rbc_errors = var_errors / cii
        
        # Sum errors over anomalous windows
        contrib = rbc_errors[anom_mask].sum(axis=0)
        peak_z = rbc_errors[anom_mask].max(axis=0)
        
        variable_names = df_test.columns
        
        df_contrib = pd.DataFrame({
            'VARIAVEL': variable_names,
            'score': contrib,
            'peak_z': peak_z,
        })
        
        df_contrib = pd.merge(df_contrib, df_sistema[['VARIAVEL', 'DESC', 'SISTEMA']], on='VARIAVEL', how='left')
        df_contrib['DESC'] = df_contrib['DESC'].fillna('NoDesc')
        df_contrib['SISTEMA'] = df_contrib['SISTEMA'].fillna('NoSystem')
        
        df_contrib = df_contrib.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        if top_k is not None:
            df_contrib = df_contrib.head(int(top_k)).copy()
            
        total = df_contrib['score'].sum()
        df_contrib['%'] = (df_contrib['score'] / total * 100) if total > 0 else 0.0
        
        df_contrib.index = df_contrib.index.astype(str)
        df_contrib = df_contrib[['VARIAVEL', 'DESC', 'SISTEMA', 'score', 'peak_z', '%']]
        
        return df_contrib.to_dict(), df_reconstruction
