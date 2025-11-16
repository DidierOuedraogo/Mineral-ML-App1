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
        **📋 Comptes de test disponibles:**
        
        - **Compte 1:** Didier / Gloria
        - **Compte 2:** student / E3MG25
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
            # Génération des données
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
        
        # Téléchargement
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
            
            if classif_algo == "Random Forest":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators = st.number_input("n_estimators", 10, 1000, 200, 10)
                    st.caption("Plus d'arbres = meilleure performance mais calcul plus lent")
                with col_b:
                    max_depth = st.number_input("max_depth", 3, 50, 15)
                    st.caption("Profondeur maximale. Trop élevé = surapprentissage")
                min_samples = st.number_input("min_samples_split", 2, 20, 5)
                st.caption("Nombre minimum pour diviser un noeud")
                
            elif classif_algo == "SVM":
                col_a, col_b = st.columns(2)
                with col_a:
                    C = st.number_input("C (Régularisation)", 0.01, 100.0, 1.0, 0.1)
                    st.caption("Compromis entre marge maximale et erreurs")
                with col_b:
                    kernel = st.selectbox("kernel", ["rbf", "linear", "poly"])
                    st.caption("RBF: données non-linéaires")
                    
            elif classif_algo == "XGBoost":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators = st.number_input("n_estimators", 10, 1000, 100, 10)
                    st.caption("Nombre d'arbres de boosting")
                with col_b:
                    learning_rate = st.number_input("learning_rate", 0.01, 1.0, 0.1, 0.01)
                    st.caption("Contribution de chaque arbre")
                max_depth = st.number_input("max_depth", 3, 20, 6)
                st.caption("Profondeur max des arbres")
                
            else:  # Logistic Regression
                col_a, col_b = st.columns(2)
                with col_a:
                    C = st.number_input("C (Inverse régularisation)", 0.001, 100.0, 1.0, 0.1)
                    st.caption("Plus petit C = régularisation plus forte")
                with col_b:
                    max_iter = st.number_input("max_iter", 50, 1000, 100)
                    st.caption("Nombre max d'itérations")
        
        # ÉTAPE 3: Entraînement et résultats
        st.markdown('<div class="step-header">📊 Étape 3: Résultats et Métriques</div>', unsafe_allow_html=True)
        
        if st.button("▶ Entraîner le Modèle", key="train_classif"):
            with st.spinner("⏳ Entraînement en cours..."):
                df = st.session_state.classif_data
                
                # Préparation des données
                X = df.drop('Classe', axis=1)
                y = df['Classe'].map({'Réfractaire': 1, 'Non-Réfractaire': 0})
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                # Standardisation
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement du modèle
                if classif_algo == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples,
                        class_weight='balanced',
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                elif classif_algo == "SVM":
                    model = SVC(C=C, kernel=kernel, class_weight='balanced', random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                elif classif_algo == "XGBoost":
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                else:  # Logistic Regression
                    model = LogisticRegression(C=C, max_iter=max_iter, class_weight='balanced', random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Calcul des métriques
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                # Affichage des résultats
                st.markdown("### 🎯 Métriques de Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.3f}", f"{accuracy*100:.1f}%")
                col2.metric("Precision", f"{precision:.3f}", "Vrais + / Total +")
                col3.metric("Recall", f"{recall:.3f}", "Vrais + / Réels +")
                col4.metric("F1-Score", f"{f1:.3f}", "Moyenne P & R")
                
                # Matrice de confusion
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
                
                # Interprétation
                st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Interprétation:</strong><br>
                    • <strong>Accuracy = {accuracy*100:.1f}%:</strong> Identifie correctement le type dans {accuracy*100:.1f}% des cas<br>
                    • <strong>Recall = {recall*100:.1f}%:</strong> {recall*100:.1f}% des minerais réfractaires sont identifiés<br>
                    • <strong>Impact:</strong> Évite coûts traitement inadapté et pertes métallurgiques
                </div>
                """, unsafe_allow_html=True)
                
                # Importance des features (pour Random Forest et XGBoost)
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
                
                # Code Python équivalent
                st.markdown("### 💻 Code Python Équivalent")
                
                code_map = {
                    "Random Forest": f"RandomForestClassifier(n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples})",
                    "SVM": f"SVC(C={C}, kernel='{kernel}')",
                    "XGBoost": f"XGBClassifier(n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth})",
                    "Logistic Regression": f"LogisticRegression(C={C}, max_iter={max_iter})"
                }
                
                code = f"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = {code_map[classif_algo]}
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1-Score: {f1:.3f}")
"""
                st.code(code, language="python")

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
    
    # ÉTAPE 1: Génération des données
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
                
                # Modèle de teneur avec relations complexes
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
    
    # Affichage des données générées
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
        
        # Téléchargement
        csv = download_data(st.session_state.reg_data, "regression_teneurs.csv")
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="regression_teneurs.csv",
            mime="text/csv",
            key="download_reg"
        )
        
        # Visualisation de la distribution
        fig = px.histogram(
            st.session_state.reg_data,
            x='Teneur_Au_g_t',
            nbins=50,
            title="Distribution des Teneurs en Or"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ÉTAPE 2: Configuration du modèle
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
            
            if reg_algo == "Random Forest":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_reg = st.number_input("n_estimators", 10, 1000, 200, 10, key="rf_reg_n")
                    st.caption("Nombre d'arbres. Plus = prédictions stables")
                with col_b:
                    max_depth_reg = st.number_input("max_depth", 3, 50, 12, key="rf_reg_depth")
                    st.caption("Profondeur maximale")
                min_samples_reg = st.number_input("min_samples_leaf", 1, 20, 2, key="rf_reg_samples")
                st.caption("Minimum d'échantillons dans feuille")
                
            elif reg_algo == "Gradient Boosting":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_reg = st.number_input("n_estimators", 10, 1000, 300, 10, key="gb_reg_n")
                    st.caption("Nombre d'étapes de boosting")
                with col_b:
                    learning_rate_reg = st.number_input("learning_rate", 0.001, 1.0, 0.05, 0.01, key="gb_reg_lr")
                    st.caption("Pondération de chaque arbre")
                max_depth_reg = st.number_input("max_depth", 3, 20, 8, key="gb_reg_depth")
                st.caption("Profondeur des arbres")
                
            else:  # Linear Regression
                fit_intercept = st.checkbox("fit_intercept", value=True, key="lr_intercept")
                st.caption("Calculer l'ordonnée à l'origine")
        
        # ÉTAPE 3: Entraînement et résultats
        st.markdown('<div class="step-header">📊 Étape 3: Résultats et Métriques</div>', unsafe_allow_html=True)
        
        if st.button("▶ Entraîner le Modèle", key="train_reg"):
            with st.spinner("⏳ Entraînement en cours..."):
                df = st.session_state.reg_data
                
                # Préparation des données
                X = df.drop('Teneur_Au_g_t', axis=1)
                y = df['Teneur_Au_g_t']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Standardisation
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement du modèle
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
                    
                else:  # Linear Regression
                    model = LinearRegression(fit_intercept=fit_intercept)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Calcul des métriques
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Affichage des résultats
                st.markdown("### 🎯 Métriques de Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Score", f"{r2:.3f}", "Variance expliquée")
                col2.metric("MAE", f"{mae:.3f} g/t", "Erreur absolue")
                col3.metric("RMSE", f"{rmse:.3f} g/t", "Erreur quadratique")
                col4.metric("MAPE", f"{mape:.1f}%", "Erreur % moyenne")
                
                # Interprétation
                st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Interprétation:</strong><br>
                    • <strong>R² = {r2:.3f}:</strong> Le modèle explique {r2*100:.1f}% de la variance<br>
                    • <strong>MAE = {mae:.3f} g/t:</strong> Erreur moyenne absolue<br>
                    • <strong>RMSE = {rmse:.3f} g/t:</strong> Pénalise grandes erreurs
                </div>
                """, unsafe_allow_html=True)
                
                # Graphique Prédictions vs Réalité
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
                
                # Ligne parfaite
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Prédiction parfaite',
                    line=dict(color='red', dash='dash')
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Importance des features
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
                
                # Code Python équivalent
                st.markdown("### 💻 Code Python Équivalent")
                
                if reg_algo == "Random Forest":
                    params = f"n_estimators={n_estimators_reg}, max_depth={max_depth_reg}, min_samples_leaf={min_samples_reg}"
                    model_name = "RandomForestRegressor"
                elif reg_algo == "Gradient Boosting":
                    params = f"n_estimators={n_estimators_reg}, learning_rate={learning_rate_reg}, max_depth={max_depth_reg}"
                    model_name = "GradientBoostingRegressor"
                else:
                    params = f"fit_intercept={fit_intercept}"
                    model_name = "LinearRegression"
                
                code = f"""
from sklearn.ensemble import {model_name}
from sklearn.metrics import r2_score, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = {model_name}({params})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.3f} g/t")
"""
                st.code(code, language="python")

# ======================= ONGLET 3: OPTIMISATION ======================= #
with tab3:
    st.markdown("## Optimisation du Process Métallurgique")
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Objectif:</strong><br>
        Optimiser les paramètres du circuit de lixiviation (pH, concentration CN⁻, temps, température) 
        pour maximiser la récupération d'or en utilisant des algorithmes de régression multivariée.
    </div>
    """, unsafe_allow_html=True)
    
    # ÉTAPE 1: Génération des données
    st.markdown('<div class="step-header">📊 Étape 1: Génération des Données Process</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Paramètres de Génération")
        optim_samples = st.slider("Nombre d'essais", 100, 5000, 600, 100, key="optim_samples")
        optim_variability = st.select_slider(
            "Variabilité du minerai",
            options=["Faible", "Moyenne", "Élevée"],
            value="Moyenne",
            key="optim_variability"
        )
        
        if st.button("🎲 Générer les Données", key="gen_optim"):
            variability_map = {"Faible": 0.6, "Moyenne": 1.0, "Élevée": 1.4}
            v_factor = variability_map[optim_variability]
            
            data = []
            for i in range(optim_samples):
                pH = 9.5 + np.random.random() * 2
                CN = 200 + np.random.random() * 800
                temps = 12 + np.random.random() * 36
                temp = 20 + np.random.random() * 30
                solides = 35 + np.random.random() * 15
                O2 = 4 + np.random.random() * 6
                P80 = 50 + np.random.random() * 100
                
                # Modèle de récupération
                recovery = (
                    70 +
                    (pH - 10.5) * 3 +
                    (CN / 600) * 8 +
                    (temps / 30) * 5 -
                    (temp - 35) * 0.3 +
                    (O2 / 7) * 4 -
                    (P80 / 100) * 3 +
                    (np.random.random() - 0.5) * 10 * v_factor
                )
                recovery = max(60, min(98, recovery))
                
                data.append({
                    'pH': round(pH, 2),
                    'CN_ppm': round(CN, 0),
                    'Temps_h': round(temps, 1),
                    'Temperature_C': round(temp, 1),
                    'Solides_%': round(solides, 1),
                    'O2_ppm': round(O2, 1),
                    'P80_um': round(P80, 0),
                    'Recovery_%': round(recovery, 2)
                })
            
            df = pd.DataFrame(data)
            st.session_state.optim_data = df
            st.success(f"✅ {optim_samples} essais générés avec succès!")
    
    with col2:
        st.markdown("### Paramètres Process")
        st.markdown("""
        **Variables contrôlables:**
        - 🔹 **pH solution** - 9.5 à 11.5
        - 🔹 **[CN⁻] (ppm)** - 200 à 1000 ppm
        - 🔹 **Temps séjour (h)** - 12 à 48 heures
        - 🔹 **Température (°C)** - 20 à 50°C
        - 🔹 **% solides pulpe** - 35 à 50%
        - 🔹 **[O₂] dissous (ppm)** - 4 à 10 ppm
        - 🔹 **Granulométrie P80 (μm)** - 50 à 150 μm
        
        ➜ **Variable cible:** Récupération Au (%)
        """)
    
    # Affichage des données générées
    if st.session_state.optim_data is not None:
        st.markdown("### 📊 Aperçu des Données Générées")
        
        col1, col2, col3 = st.columns(3)
        avg_recovery = st.session_state.optim_data['Recovery_%'].mean()
        max_recovery = st.session_state.optim_data['Recovery_%'].max()
        min_recovery = st.session_state.optim_data['Recovery_%'].min()
        
        col1.metric("Moy Récup (%)", f"{avg_recovery:.2f}")
        col2.metric("Max Récup (%)", f"{max_recovery:.2f}")
        col3.metric("Min Récup (%)", f"{min_recovery:.2f}")
        
        st.dataframe(st.session_state.optim_data.head(10), use_container_width=True)
        
        # Téléchargement
        csv = download_data(st.session_state.optim_data, "optimisation_process.csv")
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="optimisation_process.csv",
            mime="text/csv",
            key="download_optim"
        )
    
    # ÉTAPE 2: Configuration du modèle
    if st.session_state.optim_data is not None:
        st.markdown('<div class="step-header">⚙️ Étape 2: Configuration du Modèle d\'Optimisation</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            optim_algo = st.selectbox(
                "Algorithme d'Optimisation",
                ["Random Forest", "Gradient Boosting", "Neural Network (MLP)"],
                key="optim_algo_select"
            )
            
            algo_info = {
                "Random Forest": "Forêt aléatoire - Capture interactions complexes",
                "Gradient Boosting": "Gradient Boosting - Très précis",
                "Neural Network (MLP)": "Réseau neuronal - Relations non-linéaires"
            }
            st.info(algo_info[optim_algo])
        
        with col2:
            st.markdown("### Paramètres du Modèle")
            
            if optim_algo == "Random Forest":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_optim = st.number_input("n_estimators", 10, 1000, 300, 10, key="rf_optim_n")
                    st.caption("Plus d'arbres pour capturer interactions")
                with col_b:
                    max_depth_optim = st.number_input("max_depth", 3, 50, 12, key="rf_optim_depth")
                    st.caption("Profondeur pour capturer non-linéarités")
                    
            elif optim_algo == "Gradient Boosting":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_optim = st.number_input("n_estimators", 10, 1000, 200, 10, key="gb_optim_n")
                    st.caption("Nombre d'arbres de boosting")
                with col_b:
                    learning_rate_optim = st.number_input("learning_rate", 0.01, 1.0, 0.1, 0.01, key="gb_optim_lr")
                    st.caption("Taux d'apprentissage")
                    
            else:  # Neural Network
                hidden_layers = st.text_input("hidden_layers (ex: 100,50)", "100,50", key="mlp_hidden")
                st.caption("Architecture réseau neuronal")
                learning_rate_mlp = st.number_input("learning_rate", 0.0001, 0.1, 0.001, 0.001, key="mlp_lr")
                st.caption("Taux pour descente de gradient")
                max_iter_mlp = st.number_input("max_iter", 100, 2000, 500, key="mlp_iter")
                st.caption("Nombre max d'époques")
        
        # ÉTAPE 3: Entraînement et résultats
        st.markdown('<div class="step-header">📊 Étape 3: Résultats et Paramètres Optimaux</div>', unsafe_allow_html=True)
        
        if st.button("▶ Entraîner et Optimiser", key="train_optim"):
            with st.spinner("⏳ Optimisation en cours..."):
                df = st.session_state.optim_data
                
                # Préparation des données
                X = df.drop('Recovery_%', axis=1)
                y = df['Recovery_%']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Standardisation
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement du modèle
                if optim_algo == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=n_estimators_optim,
                        max_depth=max_depth_optim,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                elif optim_algo == "Gradient Boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators_optim,
                        learning_rate=learning_rate_optim,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                else:  # Neural Network
                    from sklearn.neural_network import MLPRegressor
                    layers = tuple(map(int, hidden_layers.split(',')))
                    model = MLPRegressor(
                        hidden_layer_sizes=layers,
                        learning_rate_init=learning_rate_mlp,
                        max_iter=max_iter_mlp,
                        random_state=42
                    )
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Calcul des métriques
                r2 = r2_score(y_test, y_pred)
                
                # Trouver les paramètres optimaux (via prédiction sur grille)
                from scipy.optimize import differential_evolution
                
                def objective(params):
                    # Transformation des params
                    if optim_algo == "Neural Network":
                        params_scaled = scaler.transform([params])
                        return -model.predict(params_scaled)[0]
                    else:
                        return -model.predict([params])[0]
                
                # Bornes pour l'optimisation
                bounds = [
                    (9.5, 11.5),   # pH
                    (200, 1000),   # CN
                    (12, 48),      # Temps
                    (20, 50),      # Température
                    (35, 50),      # Solides
                    (4, 10),       # O2
                    (50, 150)      # P80
                ]
                
                result = differential_evolution(objective, bounds, seed=42, maxiter=100)
                optimal_params = result.x
                optimal_recovery = -result.fun
                
                baseline_recovery = y.mean()
                improvement = ((optimal_recovery - baseline_recovery) / baseline_recovery) * 100
                
                # Affichage des résultats
                st.markdown("### 🎯 Métriques de Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("R² Modèle", f"{r2:.3f}", "Qualité prédiction")
                col2.metric("Récup. Optimale", f"{optimal_recovery:.2f}%", "Récupération Au max")
                col3.metric("Récup. Baseline", f"{baseline_recovery:.2f}%", "Avant optimisation")
                col4.metric("Amélioration", f"+{improvement:.1f}%", "Gain relatif")
                
                # Paramètres optimaux
                st.markdown(f"""
                <div class="success-box">
                    <strong>🎯 Paramètres Optimaux:</strong><br><br>
                    • pH: {optimal_params[0]:.2f}<br>
                    • [CN⁻]: {optimal_params[1]:.0f} ppm<br>
                    • Temps: {optimal_params[2]:.1f} h<br>
                    • Température: {optimal_params[3]:.1f}°C<br>
                    • % Solides: {optimal_params[4]:.1f}%<br>
                    • [O₂]: {optimal_params[5]:.1f} ppm<br>
                    • P80: {optimal_params[6]:.0f} μm<br><br>
                    <strong>Gain: +{improvement:.1f}%</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Importance des features
                if optim_algo in ["Random Forest", "Gradient Boosting"]:
                    st.markdown("### 📊 Importance des Paramètres")
                    
                    feature_importance = pd.DataFrame({
                        'Paramètre': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Paramètre',
                        orientation='h',
                        title="Importance des Paramètres Process"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Graphique de surface 3D pour pH vs CN vs Recovery
                st.markdown("### 📊 Surface de Réponse: pH vs CN")
                
                # Créer une grille pour pH et CN
                pH_range = np.linspace(9.5, 11.5, 30)
                CN_range = np.linspace(200, 1000, 30)
                pH_grid, CN_grid = np.meshgrid(pH_range, CN_range)
                
                # Fixer les autres paramètres à leurs valeurs optimales
                recovery_grid = np.zeros_like(pH_grid)
                for i in range(pH_grid.shape[0]):
                    for j in range(pH_grid.shape[1]):
                        params = [
                            pH_grid[i, j],
                            CN_grid[i, j],
                            optimal_params[2],
                            optimal_params[3],
                            optimal_params[4],
                            optimal_params[5],
                            optimal_params[6]
                        ]
                        if optim_algo == "Neural Network":
                            params_scaled = scaler.transform([params])
                            recovery_grid[i, j] = model.predict(params_scaled)[0]
                        else:
                            recovery_grid[i, j] = model.predict([params])[0]
                
                fig = go.Figure(data=[go.Surface(
                    x=pH_range,
                    y=CN_range,
                    z=recovery_grid,
                    colorscale='Viridis'
                )])
                
                fig.update_layout(
                    title='Surface de Réponse: Récupération vs pH et CN',
                    scene=dict(
                        xaxis_title='pH',
                        yaxis_title='CN (ppm)',
                        zaxis_title='Récupération (%)'
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Code Python équivalent
                st.markdown("### 💻 Code Python Équivalent")
                
                if optim_algo == "Random Forest":
                    params = f"n_estimators={n_estimators_optim}, max_depth={max_depth_optim}"
                    model_name = "RandomForestRegressor"
                elif optim_algo == "Gradient Boosting":
                    params = f"n_estimators={n_estimators_optim}, learning_rate={learning_rate_optim}"
                    model_name = "GradientBoostingRegressor"
                else:
                    params = f"hidden_layer_sizes={hidden_layers}, learning_rate_init={learning_rate_mlp}, max_iter={max_iter_mlp}"
                    model_name = "MLPRegressor"
                
                code = f"""
from sklearn.ensemble import {model_name}
from scipy.optimize import differential_evolution

model = {model_name}({params})
model.fit(X_train, y_train)

def objective(params):
    return -model.predict([params])[0]

bounds = [(9.5, 11.5), (200, 1000), (12, 48), (20, 50), (35, 50), (4, 10), (50, 150)]
result = differential_evolution(objective, bounds)

print(f"Récupération optimale: {optimal_recovery:.2f}%")
print(f"Paramètres: pH={optimal_params[0]:.2f}, CN={optimal_params[1]:.0f}")
"""
                st.code(code, language="python")

# ======================= ONGLET 4: MAINTENANCE ======================= #
with tab4:
    st.markdown("## Maintenance Prédictive des Équipements")
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Objectif:</strong><br>
        Prédire les pannes d'équipements critiques (broyeurs, pompes, convoyeurs) à partir de données 
        de capteurs IoT pour anticiper la maintenance et minimiser les arrêts non planifiés.
    </div>
    """, unsafe_allow_html=True)
    
    # ÉTAPE 1: Génération des données
    st.markdown('<div class="step-header">📊 Étape 1: Génération des Données Capteurs</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Paramètres de Génération")
        maint_samples = st.slider("Nombre de mesures", 100, 5000, 1200, 100, key="maint_samples")
        maint_failure = st.slider("Taux de pannes (%)", 5, 30, 15, 5, key="maint_failure")
        
        if st.button("🎲 Générer les Données", key="gen_maint"):
            n_failures = int(maint_samples * maint_failure / 100)
            n_normal = maint_samples - n_failures
            
            data = []
            
            # Données de panne
            for i in range(n_failures):
                vibration = 8 + np.random.random() * 12
                temperature = 75 + np.random.random() * 25
                courant = 180 + np.random.random() * 70
                pression_huile = 1.5 + np.random.random() * 2
                vitesse = 600 + np.random.random() * 400
                bruit = 95 + np.random.random() * 20
                heures = 4000 + np.random.random() * 6000
                
                data.append({
                    'Vibration_mm_s': round(vibration, 2),
                    'Temperature_C': round(temperature, 1),
                    'Courant_A': round(courant, 1),
                    'Pression_huile_bar': round(pression_huile, 2),
                    'Vitesse_RPM': round(vitesse, 0),
                    'Bruit_dB': round(bruit, 1),
                    'Heures_fonct': round(heures, 0),
                    'Etat': 1  # Panne
                })
            
            # Données normales
            for i in range(n_normal):
                vibration = 2 + np.random.random() * 4
                temperature = 50 + np.random.random() * 15
                courant = 120 + np.random.random() * 40
                pression_huile = 4 + np.random.random() * 2
                vitesse = 900 + np.random.random() * 200
                bruit = 75 + np.random.random() * 12
                heures = 500 + np.random.random() * 3000
                
                data.append({
                    'Vibration_mm_s': round(vibration, 2),
                    'Temperature_C': round(temperature, 1),
                    'Courant_A': round(courant, 1),
                    'Pression_huile_bar': round(pression_huile, 2),
                    'Vitesse_RPM': round(vitesse, 0),
                    'Bruit_dB': round(bruit, 1),
                    'Heures_fonct': round(heures, 0),
                    'Etat': 0  # Normal
                })
            
            df = pd.DataFrame(data)
            df = df.sample(frac=1).reset_index(drop=True)
            st.session_state.maint_data = df
            st.success(f"✅ {maint_samples} mesures générées avec succès!")
    
    with col2:
        st.markdown("### Signaux Capteurs")
        st.markdown("""
        **Variables mesurées:**
        - 🔹 **Vibration (mm/s)** - Accéléromètres
        - 🔹 **Température (°C)** - Thermocouples
        - 🔹 **Courant moteur (A)** - Charge électrique
        - 🔹 **Pression huile (bar)** - Système lubrifiant
        - 🔹 **Vitesse rotation (RPM)** - Encodeur
        - 🔹 **Niveau bruit (dB)** - Acoustique
        - 🔹 **Heures fonctionnement** - Temps cumulé
        
        ➜ **État (0=Normal, 1=Panne)** - Variable cible
        """)
    
    # Affichage des données générées
    if st.session_state.maint_data is not None:
        st.markdown("### 📊 Aperçu des Données Générées")
        
        col1, col2, col3 = st.columns(3)
        total_samples = len(st.session_state.maint_data)
        pannes = len(st.session_state.maint_data[st.session_state.maint_data['Etat'] == 1])
        normal = total_samples - pannes
        
        col1.metric("Total Mesures", total_samples)
        col2.metric("Pannes", pannes)
        col3.metric("Normal", normal)
        
        # Afficher avec labels textuels
        df_display = st.session_state.maint_data.copy()
        df_display['Etat_Text'] = df_display['Etat'].map({0: 'Normal', 1: 'Panne'})
        st.dataframe(df_display.head(10), use_container_width=True)
        
        # Téléchargement
        csv = download_data(st.session_state.maint_data, "maintenance_predictive.csv")
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="maintenance_predictive.csv",
            mime="text/csv",
            key="download_maint"
        )
    
    # ÉTAPE 2: Configuration du modèle
    if st.session_state.maint_data is not None:
        st.markdown('<div class="step-header">⚙️ Étape 2: Configuration du Modèle Prédictif</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            maint_algo = st.selectbox(
                "Algorithme de Prédiction",
                ["Random Forest", "XGBoost", "Logistic Regression"],
                key="maint_algo_select"
            )
            
            algo_info = {
                "Random Forest": "Forêt aléatoire - Équilibré précision/rappel",
                "XGBoost": "XGBoost - Excellent pour déséquilibre classes",
                "Logistic Regression": "Régression logistique - Probabilités interprétables"
            }
            st.info(algo_info[maint_algo])
        
        with col2:
            st.markdown("### Paramètres du Modèle")
            
            if maint_algo == "Random Forest":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_maint = st.number_input("n_estimators", 10, 1000, 200, 10, key="rf_maint_n")
                    st.caption("Plus d'arbres = prédictions robustes")
                with col_b:
                    max_depth_maint = st.number_input("max_depth", 3, 50, 15, key="rf_maint_depth")
                    st.caption("Profondeur pour interactions")
                class_weight = st.selectbox("class_weight", ["balanced", "none"], key="rf_class_weight")
                st.caption("Balanced compense déséquilibre. CRUCIAL!")
                
            elif maint_algo == "XGBoost":
                col_a, col_b = st.columns(2)
                with col_a:
                    n_estimators_maint = st.number_input("n_estimators", 10, 1000, 150, 10, key="xgb_maint_n")
                    st.caption("Arbres de boosting séquentiels")
                with col_b:
                    learning_rate_maint = st.number_input("learning_rate", 0.01, 1.0, 0.1, 0.01, key="xgb_maint_lr")
                    st.caption("Taux d'apprentissage")
                scale_pos_weight = st.number_input("scale_pos_weight", 1, 20, 5, key="xgb_scale")
                st.caption("Poids pour pannes. Plus élevé = sensibilité")
                
            else:  # Logistic Regression
                col_a, col_b = st.columns(2)
                with col_a:
                    C_maint = st.number_input("C (Régularisation)", 0.01, 100.0, 1.0, 0.1, key="lr_maint_c")
                    st.caption("Inverse régularisation")
                with col_b:
                    class_weight = st.selectbox("class_weight", ["balanced", "none"], key="lr_class_weight")
                    st.caption("Équilibrer poids classes")
        
        # ÉTAPE 3: Entraînement et résultats
        st.markdown('<div class="step-header">📊 Étape 3: Résultats et Alertes Prédictives</div>', unsafe_allow_html=True)
        
        if st.button("▶ Entraîner le Modèle", key="train_maint"):
            with st.spinner("⏳ Entraînement en cours..."):
                df = st.session_state.maint_data
                
                # Préparation des données
                X = df.drop('Etat', axis=1)
                y = df['Etat']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                # Standardisation
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement du modèle
                if maint_algo == "Random Forest":
                    cw = 'balanced' if class_weight == 'balanced' else None
                    model = RandomForestClassifier(
                        n_estimators=n_estimators_maint,
                        max_depth=max_depth_maint,
                        class_weight=cw,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                elif maint_algo == "XGBoost":
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        n_estimators=n_estimators_maint,
                        learning_rate=learning_rate_maint,
                        scale_pos_weight=scale_pos_weight,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                else:  # Logistic Regression
                    cw = 'balanced' if class_weight == 'balanced' else None
                    model = LogisticRegression(C=C_maint, class_weight=cw, max_iter=1000, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Calcul des métriques
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                # Affichage des résultats
                st.markdown("### 🎯 Métriques de Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.3f}", "Précision globale")
                col2.metric("Recall (Pannes)", f"{recall:.3f}", f"{recall*100:.1f}% détectées")
                col3.metric("Precision", f"{precision:.3f}", "Fiabilité alertes")
                col4.metric("F1-Score", f"{f1:.3f}", "Score équilibré")
                
                # Interprétation
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ Interprétation:</strong><br>
                    • <strong>Recall = {recall*100:.1f}%:</strong> Détecte {recall*100:.1f}% des pannes<br>
                    • <strong>Precision = {precision*100:.1f}%:</strong> {precision*100:.1f}% des alertes justifiées<br>
                    • <strong>Impact:</strong> Maintenance proactive et réduction temps d'arrêt
                </div>
                """, unsafe_allow_html=True)
                
                # Matrice de confusion
                st.markdown("### 📊 Matrice de Confusion")
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Normal', 'Panne'],
                    y=['Normal', 'Panne'],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                    colorscale='RdYlGn_r'
                ))
                fig.update_layout(
                    title="Matrice de Confusion",
                    xaxis_title="Prédiction",
                    yaxis_title="Réalité",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Importance des features
                if maint_algo in ["Random Forest", "XGBoost"]:
                    st.markdown("### 📊 Importance des Signaux Capteurs")
                    
                    feature_importance = pd.DataFrame({
                        'Signal': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    # Ajouter interprétation
                    interpretations = {
                        'Vibration_mm_s': 'Indicateur principal usure',
                        'Temperature_C': 'Surchauffe roulements',
                        'Heures_fonct': 'Fatigue matériaux',
                        'Courant_A': 'Charge anormale',
                        'Pression_huile_bar': 'Problème lubrification',
                        'Vitesse_RPM': 'Désalignement',
                        'Bruit_dB': 'Anomalie acoustique'
                    }
                    
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Signal',
                        orientation='h',
                        title="Importance Relative des Signaux"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau d'interprétation
                    st.markdown("### 🔍 Interprétation des Signaux")
                    
                    interp_df = feature_importance.copy()
                    interp_df['Interprétation'] = interp_df['Signal'].map(interpretations)
                    st.dataframe(interp_df[['Signal', 'Importance', 'Interprétation']], use_container_width=True)
                
                # Code Python équivalent
                st.markdown("### 💻 Code Python Équivalent")
                
                if maint_algo == "Random Forest":
                    cw_str = "'balanced'" if class_weight == 'balanced' else "None"
                    params = f"n_estimators={n_estimators_maint}, max_depth={max_depth_maint}, class_weight={cw_str}"
                    model_name = "RandomForestClassifier"
                elif maint_algo == "XGBoost":
                    params = f"n_estimators={n_estimators_maint}, learning_rate={learning_rate_maint}, scale_pos_weight={scale_pos_weight}"
                    model_name = "XGBClassifier"
                else:
                    cw_str = "'balanced'" if class_weight == 'balanced' else "None"
                    params = f"C={C_maint}, class_weight={cw_str}"
                    model_name = "LogisticRegression"
                
                code = f"""
from sklearn.ensemble import {model_name}

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model = {model_name}({params})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Recall: {recall:.3f}")
print(f"Accuracy: {accuracy:.3f}")
"""
                st.code(code, language="python")

# Footer
st.markdown("""
<div class="footer">
    <p style="font-weight: 600;">© 2025 Application ML Mining - Didier Ouedraogo, P.Geo</p>
    <p style="color: #9ca3af;">Simulateur pédagogique pour l'industrie minière | Données simulées à des fins didactiques</p>
</div>
""", unsafe_allow_html=True)