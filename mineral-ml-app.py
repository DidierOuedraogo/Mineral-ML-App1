import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import io

# Configuration de la page
st.set_page_config(
    page_title="Application ML Mining - Didier Ouedraogo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= AUTHENTIFICATION ======================= #
def check_login(username, password):
    """Vérifie les identifiants de connexion"""
    users = {
        "Didier": "Gloria",
        "student": "E3MG25"
    }
    return users.get(username) == password

# Initialisation de la session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Page de connexion
if not st.session_state.logged_in:
    st.markdown("""
    <style>
        .login-header {
            background: linear-gradient(90deg, #4b5563 0%, #374151 100%);
            padding: 3rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        .login-box {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            border: 2px solid #e5e7eb;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-header">
        <h1>🎯 Application Machine Learning Pédagogique</h1>
        <h2>Pour l'Industrie Minière</h2>
        <p style="font-size: 1.1rem; margin-top: 1rem;">Auteur: Didier Ouedraogo, P.Geo</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("### 🔐 Authentification")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("👤 Nom d'utilisateur", key="login_username")
            password = st.text_input("🔒 Mot de passe", type="password", key="login_password")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                submit = st.form_submit_button("🔓 Se connecter", use_container_width=True)
            
            if submit:
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Connexion réussie!")
                    st.rerun()
                else:
                    st.error("❌ Nom d'utilisateur ou mot de passe incorrect")
        
        st.markdown("---")
        st.info("""
        **📋 Information:**
        
        Contactez l'auteur pour obtenir vos identifiants d'accès.
        
        **Contact:** Didier Ouedraogo, P.Geo
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# ======================= APPLICATION PRINCIPALE ======================= #

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4b5563 0%, #374151 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #8b5cf6;
    }
    .step-header {
        background: linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937;
        color: #d1d5db;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
    }
    div[data-testid="stExpander"] {
        background-color: #f9fafb;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
    }
    .footer {
        background: #4b5563;
        color: white;
        text-align: center;
        padding: 1rem;
        margin-top: 3rem;
        border-radius: 8px;
    }
    .user-badge {
        background: linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar avec informations utilisateur
st.sidebar.markdown(f'<div class="user-badge">👤 {st.session_state.username}</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

if st.sidebar.button("🚪 Déconnexion", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Navigation")
st.sidebar.info("Plateforme d'apprentissage du Machine Learning appliqué à l'industrie minière")

# En-tête principal
st.markdown(f"""
<div class="main-header">
    <h1>🎯 Application Machine Learning Pédagogique</h1>
    <h2>Pour l'Industrie Minière</h2>
    <p>Auteur: Didier Ouedraogo, P.Geo | Expérimentez avec des algorithmes ML appliqués au secteur minier</p>
    <p style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.9;">Connecté en tant que: <strong>{st.session_state.username}</strong></p>
</div>
""", unsafe_allow_html=True)

# Fonction pour télécharger les données
def download_data(df, filename):
    csv = df.to_csv(index=False)
    return csv

# Initialisation des variables de session
if 'classif_data' not in st.session_state:
    st.session_state.classif_data = None
if 'reg_data' not in st.session_state:
    st.session_state.reg_data = None
if 'optim_data' not in st.session_state:
    st.session_state.optim_data = None
if 'maint_data' not in st.session_state:
    st.session_state.maint_data = None

# Onglets principaux
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Classification Minerai",
    "📈 Régression Teneurs",
    "⚙️ Optimisation Process",
    "🔧 Maintenance Prédictive"
])

# ======================= ONGLET 1: CLASSIFICATION ======================= #
with tab1:
    st.markdown("## Classification de Minerai: Réfractaire vs Non-Réfractaire")
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Objectif:</strong><br>
        Classifier automatiquement le minerai aurifère en deux catégories (Réfractaire / Non-Réfractaire) 
        à partir d'analyses géochimiques et minéralogiques pour optimiser le choix du traitement métallurgique.
    </div>
    """, unsafe_allow_html=True)
    
    # ÉTAPE 1: Génération des données
    st.markdown('<div class="step-header">📊 Étape 1: Génération des Données</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Paramètres de Génération")
        classif_samples = st.slider("Nombre d'échantillons", 100, 5000, 1000, 100)
        classif_prop = st.slider("Proportion Réfractaire (%)", 20, 60, 40, 5)
        classif_noise = st.slider("Bruit dans les données (%)", 0, 50, 10, 5)
        
        if st.button("🎲 Générer les Données", key="gen_classif"):
            n_refract = int(classif_samples * classif_prop / 100)
            n_non_refract = classif_samples - n_refract
            noise_factor = classif_noise / 100
            
            data = []
            
            # Minerai Réfractaire
            for i in range(n_refract):
                pyrite = 8 + np.random.random() * 12 + (np.random.random() - 0.5) * noise_factor * 10
                arsenopyrite = 3 + np.random.random() * 7 + (np.random.random() - 0.5) * noise_factor * 5
                carbone = 0.8 + np.random.random() * 2.5 + (np.random.random() - 0.5) * noise_factor * 2
                oxydation = 10 + np.random.random() * 30 + (np.random.random() - 0.5) * noise_factor * 20
                taille_grain = 2 + np.random.random() * 15 + (np.random.random() - 0.5) * noise_factor * 10
                sulfures_totaux = pyrite + arsenopyrite + np.random.random() * 5
                antimoine = 200 + np.random.random() * 800 + (np.random.random() - 0.5) * noise_factor * 400
                recup_grav = 5 + np.random.random() * 25 + (np.random.random() - 0.5) * noise_factor * 15
                
                data.append({
                    'Pyrite_%': round(pyrite, 2),
                    'Arsenopyrite_%': round(arsenopyrite, 2),
                    'Carbone_org_%': round(carbone, 2),
                    'Oxydation': round(oxydation, 1),
                    'Taille_grain_um': round(taille_grain, 1),
                    'Sulfures_totaux_%': round(sulfures_totaux, 2),
                    'Antimoine_ppm': round(antimoine, 0),
                    'Recup_grav_%': round(recup_grav, 1),
                    'Classe': 'Réfractaire'
                })
            
            # Minerai Non-Réfractaire
            for i in range(n_non_refract):
                pyrite = 0.5 + np.random.random() * 4 + (np.random.random() - 0.5) * noise_factor * 3
                arsenopyrite = 0.1 + np.random.random() * 1.5 + (np.random.random() - 0.5) * noise_factor * 1
                carbone = 0.05 + np.random.random() * 0.5 + (np.random.random() - 0.5) * noise_factor * 0.3
                oxydation = 60 + np.random.random() * 35 + (np.random.random() - 0.5) * noise_factor * 20
                taille_grain = 30 + np.random.random() * 100 + (np.random.random() - 0.5) * noise_factor * 50
                sulfures_totaux = pyrite + arsenopyrite + np.random.random() * 2
                antimoine = 10 + np.random.random() * 150 + (np.random.random() - 0.5) * noise_factor * 100
                recup_grav = 55 + np.random.random() * 40 + (np.random.random() - 0.5) * noise_factor * 20
                
                data.append({
                    'Pyrite_%': round(pyrite, 2),
                    'Arsenopyrite_%': round(arsenopyrite, 2),
                    'Carbone_org_%': round(carbone, 2),
                    'Oxydation': round(oxydation, 1),
                    'Taille_grain_um': round(taille_grain, 1),
                    'Sulfures_totaux_%': round(sulfures_totaux, 2),
                    'Antimoine_ppm': round(antimoine, 0),
                    'Recup_grav_%': round(recup_grav, 1),
                    'Classe': 'Non-Réfractaire'
                })
            
            df = pd.DataFrame(data)
            df = df.sample(frac=1).reset_index(drop=True)
            st.session_state.classif_data = df
            st.success(f"✅ {classif_samples} échantillons générés avec succès!")
    
    with col2:
        st.markdown("### Variables Prédictives")
        st.markdown("""
        <div style="background: #fee2e2; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; border-left: 4px solid #ef4444;">
            <strong style="color: #991b1b;">🔴 Minerai Réfractaire</strong><br>
            <small>Sulfures élevés, carbone organique, encapsulation Au</small>
        </div>
        <div style="background: #d1fae5; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #10b981;">
            <strong style="color: #065f46;">🟢 Minerai Non-Réfractaire</strong><br>
            <small>Or libre, oxydation élevée, cyanuration facile</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Variables utilisées:**
        - 🔹 Pyrite (%) - Sulfure principal
        - 🔹 Arsénopyrite (%) - Minéral réfractaire
        - 🔹 Carbone organique (%) - Preg-robbing
        - 🔹 Degré d'oxydation - État minéraux
        - 🔹 Taille grain Au (μm) - Encapsulation
        - 🔹 Sulfures totaux (%) - S total
        - 🔹 Antimoine (ppm) - Élément délétère
        - 🔹 Récupération gravimétrique (%) - Or libre
        """)
    
    # Affichage des données générées
    if st.session_state.classif_data is not None:
        st.markdown("### 📊 Aperçu des Données Générées")
        
        col1, col2, col3 = st.columns(3)
        refract_count = len(st.session_state.classif_data[st.session_state.classif_data['Classe'] == 'Réfractaire'])
        non_refract_count = len(st.session_state.classif_data[st.session_state.classif_data['Classe'] == 'Non-Réfractaire'])
        
        col1.metric("Total Échantillons", len(st.session_state.classif_data))
        col2.metric("Réfractaire", refract_count)
        col3.metric("Non-Réfractaire", non_refract_count)
        
        st.dataframe(st.session_state.classif_data.head(10), use_container_width=True)
        
        csv = download_data(st.session_state.classif_data, "classification_minerai.csv")
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="classification_minerai.csv",
            mime="text/csv"
        )
    
    # ÉTAPE 2: Configuration du modèle
    if st.session_state.classif_data is not None:
        st.markdown('<div class="step-header">⚙️ Étape 2: Configuration du Modèle</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            classif_algo = st.selectbox(
                "Algorithme de Classification",
                ["Random Forest", "SVM", "XGBoost", "Logistic Regression"],
                key="classif_algo_select"
            )
            
            algo_info = {
                "Random Forest": "Ensemble d'arbres de décision - Robuste et précis",
                "SVM": "Séparation par hyperplan - Données complexes",
                "XGBoost": "Boosting par gradient - Performance optimale",
                "Logistic Regression": "Régression logistique - Simple et interprétable"
            }
            st.info(algo_info[classif_algo])
        
        with col2:
            st.markdown("### Paramètres du Modèle")
            
            n_estimators_classif = 200
            max_depth_classif = 15
            min_samples_classif = 5
            C_classif = 1.0
            kernel_classif = 'rbf'
            learning_rate_classif = 0.1
            max_iter_classif = 100
            
            if classif_algo == "Random Forest":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_classif = st.number_input("n_estimators", 10, 1000, 200, 10, key="rf_classif_n")
                    st.caption("Plus d'arbres = meilleure performance mais calcul plus lent")
                with col_b:
                    max_depth_classif = st.number_input("max_depth", 3, 50, 15, key="rf_classif_depth")
                    st.caption("Profondeur maximale. Trop élevé = surapprentissage")
                min_samples_classif = st.number_input("min_samples_split", 2, 20, 5, key="rf_classif_min")
                st.caption("Nombre minimum pour diviser un noeud")
                
            elif classif_algo == "SVM":
                col_a, col_b = st.columns(2)
                with col_a:
                    C_classif = st.number_input("C (Régularisation)", 0.01, 100.0, 1.0, 0.1, key="svm_c")
                    st.caption("Compromis entre marge maximale et erreurs")
                with col_b:
                    kernel_classif = st.selectbox("kernel", ["rbf", "linear", "poly"], key="svm_kernel")
                    st.caption("RBF: données non-linéaires")
                    
            elif classif_algo == "XGBoost":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_classif = st.number_input("n_estimators", 10, 1000, 100, 10, key="xgb_classif_n")
                    st.caption("Nombre d'arbres de boosting")
                with col_b:
                    learning_rate_classif = st.number_input("learning_rate", 0.01, 1.0, 0.1, 0.01, key="xgb_classif_lr")
                    st.caption("Contribution de chaque arbre")
                max_depth_classif = st.number_input("max_depth", 3, 20, 6, key="xgb_classif_depth")
                st.caption("Profondeur max des arbres")
                
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    C_classif = st.number_input("C (Inverse régularisation)", 0.001, 100.0, 1.0, 0.1, key="lr_classif_c")
                    st.caption("Plus petit C = régularisation plus forte")
                with col_b:
                    max_iter_classif = st.number_input("max_iter", 50, 1000, 100, key="lr_classif_iter")
                    st.caption("Nombre max d'itérations")
        
        # ÉTAPE 3: Entraînement et résultats
        st.markdown('<div class="step-header">📊 Étape 3: Résultats et Métriques</div>', unsafe_allow_html=True)
        
        if st.button("▶ Entraîner le Modèle", key="train_classif"):
            with st.spinner("⏳ Entraînement en cours..."):
                df = st.session_state.classif_data
                
                X = df.drop('Classe', axis=1)
                y = df['Classe'].map({'Réfractaire': 1, 'Non-Réfractaire': 0})
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                if classif_algo == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators_classif,
                        max_depth=max_depth_classif,
                        min_samples_split=min_samples_classif,
                        class_weight='balanced',
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                elif classif_algo == "SVM":
                    model = SVC(C=C_classif, kernel=kernel_classif, class_weight='balanced', random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                elif classif_algo == "XGBoost":
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        n_estimators=n_estimators_classif,
                        learning_rate=learning_rate_classif,
                        max_depth=max_depth_classif,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                else:
                    model = LogisticRegression(C=C_classif, max_iter=max_iter_classif, class_weight='balanced', random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                
                st.markdown("### 🎯 Métriques de Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.3f}", f"{accuracy*100:.1f}%")
                col2.metric("Precision", f"{precision:.3f}", "Vrais + / Total +")
                col3.metric("Recall", f"{recall:.3f}", "Vrais + / Réels +")
                col4.metric("F1-Score", f"{f1:.3f}", "Moyenne P & R")
                
                st.markdown("### 📊 Matrice de Confusion")
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Non-Réfractaire', 'Réfractaire'],
                    y=['Non-Réfractaire', 'Réfractaire'],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                    colorscale='Blues'
                ))
                fig.update_layout(
                    title="Matrice de Confusion",
                    xaxis_title="Prédiction",
                    yaxis_title="Réalité",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Interprétation:</strong><br>
                    • <strong>Accuracy = {accuracy*100:.1f}%:</strong> Identifie correctement le type dans {accuracy*100:.1f}% des cas<br>
                    • <strong>Recall = {recall*100:.1f}%:</strong> {recall*100:.1f}% des minerais réfractaires sont identifiés<br>
                    • <strong>Impact:</strong> Évite coûts traitement inadapté et pertes métallurgiques
                </div>
                """, unsafe_allow_html=True)
                
                if classif_algo in ["Random Forest", "XGBoost"]:
                    st.markdown("### 📊 Importance des Variables")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Importance des Variables dans la Classification"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ======================= ONGLET 2: RÉGRESSION ======================= #
with tab2:
    st.markdown("## Prédiction des Teneurs en Or (g/t)")
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Objectif:</strong><br>
        Prédire la teneur en or à partir de données géochimiques (As, Cu, Sb) et géospatiales 
        (distance à faille, profondeur) en utilisant des algorithmes de régression.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="step-header">📊 Étape 1: Génération des Données</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Paramètres de Génération")
        reg_samples = st.slider("Nombre d'échantillons", 100, 5000, 800, 100, key="reg_samples")
        reg_complexity = st.select_slider(
            "Complexité des relations",
            options=["Faible", "Moyenne", "Élevée"],
            value="Moyenne",
            key="reg_complexity"
        )
        
        if st.button("🎲 Générer les Données", key="gen_reg"):
            complexity_map = {"Faible": 0.5, "Moyenne": 1.0, "Élevée": 1.5}
            c_factor = complexity_map[reg_complexity]
            
            data = []
            for i in range(reg_samples):
                As = np.random.random() * 200 + 50
                Cu = np.random.random() * 500 + 100
                Sb = np.random.random() * 100 + 10
                dist_faille = np.random.random() * 500
                profondeur = np.random.random() * 300 + 50
                alteration = np.random.random() * 10
                
                teneur = (
                    0.5 +
                    (As / 100) * 0.3 * c_factor +
                    (Cu / 200) * 0.2 * c_factor +
                    (Sb / 50) * 0.25 * c_factor -
                    (dist_faille / 200) * 0.4 * c_factor +
                    (alteration / 10) * 0.5 * c_factor +
                    (np.random.random() - 0.5) * 0.8
                )
                teneur = max(0.1, min(10, teneur))
                
                data.append({
                    'As_ppm': round(As, 1),
                    'Cu_ppm': round(Cu, 1),
                    'Sb_ppm': round(Sb, 1),
                    'Dist_faille_m': round(dist_faille, 1),
                    'Profondeur_m': round(profondeur, 1),
                    'Alteration': round(alteration, 2),
                    'Teneur_Au_g_t': round(teneur, 3)
                })
            
            df = pd.DataFrame(data)
            st.session_state.reg_data = df
            st.success(f"✅ {reg_samples} échantillons générés avec succès!")
    
    with col2:
        st.markdown("### Variables Prédictives")
        st.markdown("""
        **Variables utilisées:**
        - 🔹 **Teneur Arsenic (As)** - Élément pathfinder pour l'or
        - 🔹 **Teneur Cuivre (Cu)** - Associé aux minéralisations
        - 🔹 **Teneur Antimoine (Sb)** - Indicateur hydrothermal
        - 🔹 **Distance à faille (m)** - Contrôle structural
        - 🔹 **Profondeur (m)** - Niveau hydrothermal
        - 🔹 **Index d'altération** - Intensité altération
        
        ➜ **Variable cible:** Teneur Au (g/t)
        """)
    
    if st.session_state.reg_data is not None:
        st.markdown("### 📊 Aperçu des Données Générées")
        
        col1, col2, col3 = st.columns(3)
        avg_teneur = st.session_state.reg_data['Teneur_Au_g_t'].mean()
        max_teneur = st.session_state.reg_data['Teneur_Au_g_t'].max()
        min_teneur = st.session_state.reg_data['Teneur_Au_g_t'].min()
        
        col1.metric("Moy Au (g/t)", f"{avg_teneur:.3f}")
        col2.metric("Max Au (g/t)", f"{max_teneur:.3f}")
        col3.metric("Min Au (g/t)", f"{min_teneur:.3f}")
        
        st.dataframe(st.session_state.reg_data.head(10), use_container_width=True)
        
        csv = download_data(st.session_state.reg_data, "regression_teneurs.csv")
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="regression_teneurs.csv",
            mime="text/csv",
            key="download_reg"
        )
        
        fig = px.histogram(
            st.session_state.reg_data,
            x='Teneur_Au_g_t',
            nbins=50,
            title="Distribution des Teneurs en Or"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.reg_data is not None:
        st.markdown('<div class="step-header">⚙️ Étape 2: Configuration du Modèle</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            reg_algo = st.selectbox(
                "Algorithme de Régression",
                ["Random Forest", "Gradient Boosting", "Linear Regression"],
                key="reg_algo_select"
            )
            
            algo_info = {
                "Random Forest": "Moyenne d'arbres - Robuste aux outliers",
                "Gradient Boosting": "Optimisation séquentielle - Très performant",
                "Linear Regression": "Régression linéaire - Simple et rapide"
            }
            st.info(algo_info[reg_algo])
        
        with col2:
            st.markdown("### Paramètres du Modèle")
            
            n_estimators_reg = 200
            max_depth_reg = 12
            min_samples_reg = 2
            learning_rate_reg = 0.05
            fit_intercept = True
            
            if reg_algo == "Random Forest":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_reg = st.number_input("n_estimators", 10, 1000, 200, 10, key="rf_reg_n")
                with col_b:
                    max_depth_reg = st.number_input("max_depth", 3, 50, 12, key="rf_reg_depth")
                min_samples_reg = st.number_input("min_samples_leaf", 1, 20, 2, key="rf_reg_samples")
                
            elif reg_algo == "Gradient Boosting":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_reg = st.number_input("n_estimators", 10, 1000, 300, 10, key="gb_reg_n")
                with col_b:
                    learning_rate_reg = st.number_input("learning_rate", 0.001, 1.0, 0.05, 0.01, key="gb_reg_lr")
                max_depth_reg = st.number_input("max_depth", 3, 20, 8, key="gb_reg_depth")
                
            else:
                fit_intercept = st.checkbox("fit_intercept", value=True, key="lr_intercept")
        
        st.markdown('<div class="step-header">📊 Étape 3: Résultats et Métriques</div>', unsafe_allow_html=True)
        
        if st.button("▶ Entraîner le Modèle", key="train_reg"):
            with st.spinner("⏳ Entraînement en cours..."):
                df = st.session_state.reg_data
                
                X = df.drop('Teneur_Au_g_t', axis=1)
                y = df['Teneur_Au_g_t']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                if reg_algo == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=n_estimators_reg,
                        max_depth=max_depth_reg,
                        min_samples_leaf=min_samples_reg,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                elif reg_algo == "Gradient Boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators_reg,
                        learning_rate=learning_rate_reg,
                        max_depth=max_depth_reg,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                else:
                    model = LinearRegression(fit_intercept=fit_intercept)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
                
                st.markdown("### 🎯 Métriques de Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{r2:.3f}", "Variance expliquée")
                col2.metric("MAE", f"{mae:.3f} g/t", "Erreur absolue")
                col3.metric("RMSE", f"{rmse:.3f} g/t", "Erreur quadratique")
                col4.metric("MAPE", f"{mape:.1f}%", "Erreur % moyenne")
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Interprétation:</strong><br>
                    • <strong>R² = {r2:.3f}:</strong> Le modèle explique {r2*100:.1f}% de la variance<br>
                    • <strong>MAE = {mae:.3f} g/t:</strong> Erreur moyenne absolue<br>
                    • <strong>RMSE = {rmse:.3f} g/t:</strong> Pénalise grandes erreurs
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📊 Prédictions vs Réalité")
                
                results_df = pd.DataFrame({
                    'Réel': y_test,
                    'Prédit': y_pred
                })
                
                fig = px.scatter(
                    results_df,
                    x='Réel',
                    y='Prédit',
                    title="Teneurs Prédites vs Teneurs Réelles",
                    labels={'Réel': 'Teneur Réelle (g/t)', 'Prédit': 'Teneur Prédite (g/t)'}
                )
                
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Prédiction parfaite',
                    line=dict(color='red', dash='dash')
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                if reg_algo in ["Random Forest", "Gradient Boosting"]:
                    st.markdown("### 📊 Importance des Variables")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Importance des Variables dans la Prédiction"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ======================= ONGLET 3: OPTIMISATION ======================= #
with tab3:
    st.markdown("## Optimisation du Process Métallurgique")
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Objectif:</strong><br>
        Optimiser les paramètres opérationnels (pH, temps de lixiviation, cyanure) pour maximiser 
        la récupération de l'or tout en minimisant les coûts de traitement.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="step-header">📊 Étape 1: Génération des Données de Process</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Paramètres de Génération")
        optim_samples = st.slider("Nombre d'essais", 100, 2000, 500, 50, key="optim_samples")
        optim_noise = st.slider("Bruit dans les données (%)", 0, 30, 5, 5, key="optim_noise")
        
        if st.button("🎲 Générer les Données", key="gen_optim"):
            noise_factor = optim_noise / 100
            
            data = []
            for i in range(optim_samples):
                pH = 10.0 + np.random.random() * 2.5
                temps = 12 + np.random.random() * 36
                cyanure = 200 + np.random.random() * 600
                temperature = 20 + np.random.random() * 15
                granulo_p80 = 50 + np.random.random() * 100
                
                recup = (
                    65 +
                    (pH - 10.5) * 8 -
                    ((pH - 11.0) ** 2) * 3 +
                    (temps / 48) * 15 -
                    ((temps - 24) ** 2) / 100 +
                    (cyanure / 800) * 10 -
                    ((cyanure - 400) ** 2) / 5000 +
                    (temperature / 35) * 5 -
                    (granulo_p80 / 150) * 8 +
                    (np.random.random() - 0.5) * noise_factor * 20
                )
                recup = max(50, min(98, recup))
                
                cout = (
                    50 +
                    (cyanure / 10) +
                    (temps * 2) +
                    (temperature - 20) * 3 +
                    (np.random.random() - 0.5) * noise_factor * 30
                )
                cout = max(30, cout)
                
                data.append({
                    'pH': round(pH, 2),
                    'Temps_h': round(temps, 1),
                    'Cyanure_ppm': round(cyanure, 0),
                    'Temperature_C': round(temperature, 1),
                    'Granulo_P80_um': round(granulo_p80, 0),
                    'Recup_%': round(recup, 2),
                    'Cout_$/t': round(cout, 2)
                })
            
            df = pd.DataFrame(data)
            st.session_state.optim_data = df
            st.success(f"✅ {optim_samples} essais générés avec succès!")
    
    with col2:
        st.markdown("### Variables d'Optimisation")
        st.markdown("""
        **Variables de contrôle:**
        - 🔹 **pH** - Optimum 10.5-11.5 pour dissolution Au
        - 🔹 **Temps lixiviation (h)** - Trade-off recup/coût
        - 🔹 **Cyanure (ppm)** - Consommation vs efficacité
        - 🔹 **Température (°C)** - Cinétique réaction
        - 🔹 **Granulométrie P80 (μm)** - Libération Au
        
        **Objectifs:**
        - ✅ Maximiser Récupération (%)
        - ✅ Minimiser Coût ($/t)
        """)
    
    if st.session_state.optim_data is not None:
        st.markdown("### 📊 Aperçu des Données de Process")
        
        col1, col2, col3, col4 = st.columns(4)
        avg_recup = st.session_state.optim_data['Recup_%'].mean()
        max_recup = st.session_state.optim_data['Recup_%'].max()
        avg_cout = st.session_state.optim_data['Cout_$/t'].mean()
        min_cout = st.session_state.optim_data['Cout_$/t'].min()
        
        col1.metric("Récup. Moy.", f"{avg_recup:.1f}%")
        col2.metric("Récup. Max.", f"{max_recup:.1f}%")
        col3.metric("Coût Moy.", f"{avg_cout:.1f} $/t")
        col4.metric("Coût Min.", f"{min_cout:.1f} $/t")
        
        st.dataframe(st.session_state.optim_data.head(10), use_container_width=True)
        
        csv = download_data(st.session_state.optim_data, "optimisation_process.csv")
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="optimisation_process.csv",
            mime="text/csv",
            key="download_optim"
        )
        
        fig = px.scatter(
            st.session_state.optim_data,
            x='Cout_$/t',
            y='Recup_%',
            color='pH',
            title="Trade-off Récupération vs Coût",
            labels={'Cout_$/t': 'Coût ($/t)', 'Recup_%': 'Récupération (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.optim_data is not None:
        st.markdown('<div class="step-header">⚙️ Étape 2: Modèle Prédictif de Récupération</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            optim_algo = st.selectbox(
                "Algorithme",
                ["Random Forest", "Gradient Boosting"],
                key="optim_algo"
            )
        
        with col2:
            st.markdown("### Paramètres du Modèle")
            
            if optim_algo == "Random Forest":
                n_est_opt = st.number_input("n_estimators", 50, 500, 150, 50, key="opt_rf_n")
                max_d_opt = st.number_input("max_depth", 5, 30, 10, key="opt_rf_d")
            else:
                n_est_opt = st.number_input("n_estimators", 50, 500, 200, 50, key="opt_gb_n")
                lr_opt = st.number_input("learning_rate", 0.01, 0.5, 0.1, 0.01, key="opt_gb_lr")
        
        if st.button("▶ Entraîner Modèle Prédictif", key="train_optim"):
            with st.spinner("⏳ Entraînement en cours..."):
                df = st.session_state.optim_data
                
                X = df[['pH', 'Temps_h', 'Cyanure_ppm', 'Temperature_C', 'Granulo_P80_um']]
                y = df['Recup_%']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if optim_algo == "Random Forest":
                    model = RandomForestRegressor(n_estimators=n_est_opt, max_depth=max_d_opt, random_state=42)
                else:
                    model = GradientBoostingRegressor(n_estimators=n_est_opt, learning_rate=lr_opt, random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                col1, col2 = st.columns(2)
                col1.metric("R² Score", f"{r2:.3f}")
                col2.metric("MAE", f"{mae:.2f}%")
                
                st.session_state['optim_model'] = model
                st.success("✅ Modèle entraîné avec succès!")
        
        if 'optim_model' in st.session_state:
            st.markdown('<div class="step-header">🎯 Étape 3: Optimisation des Paramètres</div>', unsafe_allow_html=True)
            
            st.markdown("### Stratégie d'Optimisation")
            
            objectif = st.radio(
                "Objectif principal",
                ["Maximiser Récupération", "Minimiser Coût", "Équilibre Récup/Coût"],
                key="objectif_optim"
            )
            
            if st.button("🚀 Lancer Optimisation", key="launch_optim"):
                with st.spinner("⏳ Optimisation en cours..."):
                    model = st.session_state['optim_model']
                    
                    def objective_function(params):
                        pH, temps, cyanure, temp, granulo = params
                        
                        X_pred = np.array([[pH, temps, cyanure, temp, granulo]])
                        recup_pred = model.predict(X_pred)[0]
                        
                        cout_pred = (
                            50 +
                            (cyanure / 10) +
                            (temps * 2) +
                            (temp - 20) * 3
                        )
                        
                        if objectif == "Maximiser Récupération":
                            return -recup_pred
                        elif objectif == "Minimiser Coût":
                            return cout_pred
                        else:
                            return -recup_pred * 0.6 + cout_pred * 0.4
                    
                    bounds = [
                        (10.0, 12.5),
                        (12, 48),
                        (200, 800),
                        (20, 35),
                        (50, 150)
                    ]
                    
                    x0 = [11.0, 24, 400, 25, 75]
                    
                    result = minimize(
                        objective_function,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds
                    )
                    
                    optimal_params = result.x
                    pH_opt, temps_opt, cyan_opt, temp_opt, gran_opt = optimal_params
                    
                    X_opt = np.array([[pH_opt, temps_opt, cyan_opt, temp_opt, gran_opt]])
                    recup_opt = model.predict(X_opt)[0]
                    
                    cout_opt = (
                        50 +
                        (cyan_opt / 10) +
                        (temps_opt * 2) +
                        (temp_opt - 20) * 3
                    )
                    
                    st.markdown("### 🎯 Paramètres Optimaux")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        **Paramètres Process:**
                        - pH: **{:.2f}**
                        - Temps: **{:.1f} h**
                        """.format(pH_opt, temps_opt))
                    
                    with col2:
                        st.markdown("""
                        **Réactifs & Conditions:**
                        - Cyanure: **{:.0f} ppm**
                        - Température: **{:.1f} °C**
                        """.format(cyan_opt, temp_opt))
                    
                    with col3:
                        st.markdown("""
                        **Granulométrie:**
                        - P80: **{:.0f} μm**
                        """.format(gran_opt))
                    
                    st.markdown("### 📊 Performances Attendues")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Récupération Optimale", f"{recup_opt:.2f}%", "Prédiction modèle")
                    col2.metric("Coût Estimé", f"{cout_opt:.2f} $/t", "Calcul direct")
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>💡 Recommandations:</strong><br>
                        • pH optimal à {pH_opt:.2f} pour maximiser dissolution Au<br>
                        • Temps de {temps_opt:.1f}h équilibre recup/coût<br>
                        • Dose cyanure à {cyan_opt:.0f} ppm évite surconsommation<br>
                        • Récupération attendue: {recup_opt:.1f}% avec coût de {cout_opt:.1f} $/t
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=['pH', 'Temps (h/10)', 'CN (ppm/100)', 'Temp (°C)', 'P80 (μm/10)'],
                        y=[pH_opt, temps_opt/10, cyan_opt/100, temp_opt, gran_opt/10],
                        marker_color='#8b5cf6'
                    ))
                    
                    fig.update_layout(
                        title="Paramètres Optimaux (normalisés)",
                        yaxis_title="Valeur",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# ======================= ONGLET 4: MAINTENANCE ======================= #
with tab4:
    st.markdown("## Maintenance Prédictive des Équipements")
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Objectif:</strong><br>
        Prédire les pannes d'équipements (broyeurs, pompes) à partir de données de capteurs 
        (vibrations, température, pression) pour planifier la maintenance et éviter les arrêts imprévus.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="step-header">📊 Étape 1: Génération des Données de Capteurs</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Paramètres de Génération")
        maint_samples = st.slider("Nombre de mesures", 500, 5000, 2000, 100, key="maint_samples")
        maint_failure_rate = st.slider("Taux de pannes (%)", 5, 30, 15, 5, key="maint_failure")
        
        if st.button("🎲 Générer les Données", key="gen_maint"):
            n_failures = int(maint_samples * maint_failure_rate / 100)
            n_normal = maint_samples - n_failures
            
            data = []
            
            # Équipements normaux
            for i in range(n_normal):
                vibration = 1.0 + np.random.random() * 3.0
                temperature = 40 + np.random.random() * 30
                pression = 2.0 + np.random.random() * 2.0
                courant = 15 + np.random.random() * 10
                hours_operation = np.random.random() * 8000
                
                data.append({
                    'Vibration_mm_s': round(vibration, 2),
                    'Temperature_C': round(temperature, 1),
                    'Pression_bar': round(pression, 2),
                    'Courant_A': round(courant, 1),
                    'Heures_operation': round(hours_operation, 0),
                    'Panne': 'Non'
                })
            
            # Équipements en panne
            for i in range(n_failures):
                vibration = 4.5 + np.random.random() * 4.0
                temperature = 65 + np.random.random() * 30
                pression = 1.0 + np.random.random() * 1.5
                courant = 25 + np.random.random() * 15
                hours_operation = 5000 + np.random.random() * 5000
                
                data.append({
                    'Vibration_mm_s': round(vibration, 2),
                    'Temperature_C': round(temperature, 1),
                    'Pression_bar': round(pression, 2),
                    'Courant_A': round(courant, 1),
                    'Heures_operation': round(hours_operation, 0),
                    'Panne': 'Oui'
                })
            
            df = pd.DataFrame(data)
            df = df.sample(frac=1).reset_index(drop=True)
            st.session_state.maint_data = df
            st.success(f"✅ {maint_samples} mesures générées avec succès!")
    
    with col2:
        st.markdown("### Variables de Surveillance")
        st.markdown("""
        **Capteurs utilisés:**
        - 🔹 **Vibration (mm/s)** - Détection déséquilibre/usure
        - 🔹 **Température (°C)** - Surchauffe roulements
        - 🔹 **Pression (bar)** - Anomalies circuit hydraulique
        - 🔹 **Courant (A)** - Surcharge moteur
        - 🔹 **Heures opération** - Usure cumulative
        
        **Seuils d'alerte:**
        - ⚠️ Vibration > 4 mm/s
        - ⚠️ Température > 65°C
        - ⚠️ Pression < 1.5 bar
        """)
    
    if st.session_state.maint_data is not None:
        st.markdown("### 📊 Aperçu des Données de Capteurs")
        
        col1, col2, col3 = st.columns(3)
        total = len(st.session_state.maint_data)
        pannes = len(st.session_state.maint_data[st.session_state.maint_data['Panne'] == 'Oui'])
        normal = total - pannes
        
        col1.metric("Total Mesures", total)
        col2.metric("Pannes", pannes, f"{pannes/total*100:.1f}%")
        col3.metric("Normaux", normal, f"{normal/total*100:.1f}%")
        
        st.dataframe(st.session_state.maint_data.head(10), use_container_width=True)
        
        csv = download_data(st.session_state.maint_data, "maintenance_predictive.csv")
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="maintenance_predictive.csv",
            mime="text/csv",
            key="download_maint"
        )
        
        fig = px.box(
            st.session_state.maint_data,
            x='Panne',
            y='Vibration_mm_s',
            title="Distribution Vibrations par Statut",
            color='Panne',
            color_discrete_map={'Oui': '#ef4444', 'Non': '#10b981'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.maint_data is not None:
        st.markdown('<div class="step-header">⚙️ Étape 2: Modèle de Prédiction de Pannes</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            maint_algo = st.selectbox(
                "Algorithme",
                ["Random Forest", "XGBoost"],
                key="maint_algo"
            )
        
        with col2:
            st.markdown("### Paramètres du Modèle")
            
            if maint_algo == "Random Forest":
                n_est_maint = st.number_input("n_estimators", 50, 500, 200, 50, key="maint_rf_n")
                max_d_maint = st.number_input("max_depth", 5, 30, 12, key="maint_rf_d")
            else:
                n_est_maint = st.number_input("n_estimators", 50, 500, 150, 50, key="maint_xgb_n")
                lr_maint = st.number_input("learning_rate", 0.01, 0.5, 0.1, 0.01, key="maint_xgb_lr")
        
        if st.button("▶ Entraîner Modèle Prédictif", key="train_maint"):
            with st.spinner("⏳ Entraînement en cours..."):
                df = st.session_state.maint_data
                
                X = df.drop('Panne', axis=1)
                y = df['Panne'].map({'Oui': 1, 'Non': 0})
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                if maint_algo == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=n_est_maint,
                        max_depth=max_d_maint,
                        class_weight='balanced',
                        random_state=42
                    )
                else:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        n_estimators=n_est_maint,
                        learning_rate=lr_maint,
                        random_state=42
                    )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                
                st.markdown("### 🎯 Performance du Modèle")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.3f}", f"{accuracy*100:.1f}%")
                col2.metric("Precision", f"{precision:.3f}", "Fiabilité alertes")
                col3.metric("Recall", f"{recall:.3f}", "Détection pannes")
                col4.metric("F1-Score", f"{f1:.3f}")
                
                st.markdown("### 📊 Matrice de Confusion")
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Normal', 'Panne'],
                    y=['Normal', 'Panne'],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                    colorscale='Reds'
                ))
                fig.update_layout(
                    title="Prédictions de Maintenance",
                    xaxis_title="Prédiction",
                    yaxis_title="Réalité",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                <div class="success-box">
                    <strong>💡 Impact Business:</strong><br>
                    • <strong>Recall = {recall*100:.1f}%:</strong> Détecte {recall*100:.1f}% des pannes avant qu'elles surviennent<br>
                    • <strong>Precision = {precision*100:.1f}%:</strong> {precision*100:.1f}% des alertes sont justifiées<br>
                    • <strong>Économies estimées:</strong> Réduction arrêts imprévus de 60-80%<br>
                    • <strong>ROI:</strong> Planification maintenance = -40% coûts réparations
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📊 Importance des Variables")
                
                feature_importance = pd.DataFrame({
                    'Capteur': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Capteur',
                    orientation='h',
                    title="Capteurs les Plus Prédictifs",
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="step-header">🔍 Simulation de Diagnostic en Temps Réel</div>', unsafe_allow_html=True)
                
                st.markdown("### Entrez les Valeurs des Capteurs")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    vib_sim = st.number_input("Vibration (mm/s)", 0.0, 10.0, 2.5, 0.1, key="vib_sim")
                    temp_sim = st.number_input("Température (°C)", 20.0, 100.0, 50.0, 1.0, key="temp_sim")
                
                with col2:
                    press_sim = st.number_input("Pression (bar)", 0.0, 5.0, 2.5, 0.1, key="press_sim")
                    curr_sim = st.number_input("Courant (A)", 0.0, 50.0, 20.0, 1.0, key="curr_sim")
                
                with col3:
                    hours_sim = st.number_input("Heures opération", 0, 10000, 4000, 100, key="hours_sim")
                
                if st.button("🔍 Diagnostiquer", key="diagnose"):
                    X_sim = np.array([[vib_sim, temp_sim, press_sim, curr_sim, hours_sim]])
                    pred = model.predict(X_sim)[0]
                    proba = model.predict_proba(X_sim)[0]
                    
                    if pred == 1:
                        st.markdown(f"""
                        <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 1.5rem; border-radius: 0.5rem;">
                            <h3 style="color: #991b1b; margin: 0;">⚠️ ALERTE PANNE IMMINENTE</h3>
                            <p style="font-size: 1.2rem; margin: 0.5rem 0;">Probabilité de panne: <strong>{proba[1]*100:.1f}%</strong></p>
                            <p style="margin: 0.5rem 0;"><strong>Action recommandée:</strong> Planifier maintenance préventive dans les 24-48h</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: #d1fae5; border-left: 4px solid #10b981; padding: 1.5rem; border-radius: 0.5rem;">
                            <h3 style="color: #065f46; margin: 0;">✅ ÉQUIPEMENT NORMAL</h3>
                            <p style="font-size: 1.2rem; margin: 0.5rem 0;">Probabilité état normal: <strong>{proba[0]*100:.1f}%</strong></p>
                            <p style="margin: 0.5rem 0;"><strong>Action:</strong> Continuer surveillance régulière</p>
                        </div>
                        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-weight: 600;">© 2025 Application ML Mining - Didier Ouedraogo, P.Geo</p>
    <p style="color: #9ca3af;">Simulateur pédagogique pour l'industrie minière | Données simulées à des fins didactiques</p>
</div>
""", unsafe_allow_html=True)