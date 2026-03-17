import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import shap
import io
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# ── DATABASE ────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('churn.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id TEXT, tenure INTEGER,
        monthly_charges REAL, total_charges REAL,
        contract TEXT, prediction TEXT,
        probability REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit(); conn.close()

def save_prediction(cid, tenure, mc, tc, contract, pred, prob):
    conn = sqlite3.connect('churn.db')
    conn.execute('INSERT INTO predictions (customer_id,tenure,monthly_charges,total_charges,contract,prediction,probability) VALUES (?,?,?,?,?,?,?)',
                 (cid, tenure, mc, tc, contract, pred, prob))
    conn.commit(); conn.close()

def get_predictions():
    conn = sqlite3.connect('churn.db')
    df = pd.read_sql('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100', conn)
    conn.close()
    return df

init_db()

# ── DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

@st.cache_data
def preprocess(df):
    df2 = df.copy()
    le = LabelEncoder()
    cats = [c for c in df2.select_dtypes('object').columns if c != 'customerID']
    for col in cats:
        df2[col] = le.fit_transform(df2[col])
    return df2

@st.cache_resource
def train_models(df):
    X = df.drop(['customerID','Churn'], axis=1)
    y = df['Churn']
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    models = {
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
        'Logistic Regression': LogisticRegression(C=0.5, max_iter=1000)
    }
    results = {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        proba = model.predict_proba(Xte)[:,1]
        cv    = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy')
        fpr, tpr, _ = roc_curve(y_test, proba)
        results[name] = {
            'model':    model,
            'accuracy': accuracy_score(y_test, preds),
            'cv_mean':  cv.mean(),
            'cv_std':   cv.std(),
            'report':   classification_report(y_test, preds),
            'cm':       confusion_matrix(y_test, preds),
            'fpr': fpr, 'tpr': tpr,
            'auc': auc(fpr, tpr),
            'X_test': Xte, 'y_test': y_test
        }
    return results, scaler, X.columns.tolist()

df_raw  = load_data()
df_proc = preprocess(df_raw)
model_results, scaler, feature_cols = train_models(df_proc)

# ── SIDEBAR ─────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Overview","📈 EDA","🤖 ML Models","🔮 Predict","📤 Bulk Predict","📋 History"])

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📊 Customer Churn Analysis Dashboard")
    st.markdown("**Telecom Customer Churn Prediction using Machine Learning**")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Customers",    f"{len(df_raw):,}")
    c2.metric("Churned",            f"{df_raw['Churn'].sum():,}")
    c3.metric("Churn Rate",         f"{df_raw['Churn'].mean()*100:.1f}%")
    c4.metric("Avg Monthly Charges",f"${df_raw['MonthlyCharges'].mean():.2f}")

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.pie(df_raw, names=df_raw['Churn'].map({1:'Churned',0:'Retained'}),
                     title="Churn Distribution",
                     color_discrete_sequence=['#ef553b','#00cc96'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_raw, x='tenure',
                           color=df_raw['Churn'].map({1:'Churned',0:'Retained'}),
                           title="Tenure Distribution", barmode='overlay',
                           color_discrete_sequence=['#ef553b','#00cc96'])
        st.plotly_chart(fig, use_container_width=True)

    # Model accuracy summary
    st.markdown("### 🤖 Model Accuracy Summary")
    acc_df = pd.DataFrame([{
        'Model': k,
        'Accuracy': f"{v['accuracy']*100:.2f}%",
        'CV Score': f"{v['cv_mean']*100:.2f}% ± {v['cv_std']*100:.2f}%",
        'AUC': f"{v['auc']:.3f}"
    } for k,v in model_results.items()])
    st.dataframe(acc_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    c1,c2 = st.columns(2)
    with c1:
        fig = px.box(df_raw, x=df_raw['Churn'].map({1:'Churned',0:'Retained'}),
                     y='MonthlyCharges', title="Monthly Charges vs Churn",
                     color=df_raw['Churn'].map({1:'Churned',0:'Retained'}),
                     color_discrete_sequence=['#ef553b','#00cc96'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df_raw, x=df_raw['Churn'].map({1:'Churned',0:'Retained'}),
                     y='TotalCharges', title="Total Charges vs Churn",
                     color=df_raw['Churn'].map({1:'Churned',0:'Retained'}),
                     color_discrete_sequence=['#ef553b','#00cc96'])
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df_raw, x='Contract',
                       color=df_raw['Churn'].map({1:'Churned',0:'Retained'}),
                       title="Contract Type vs Churn", barmode='group',
                       color_discrete_sequence=['#ef553b','#00cc96'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Correlation Heatmap")
    num_cols = df_proc.select_dtypes(include=np.number).columns.tolist()
    corr = df_proc[num_cols].corr()
    fig = px.imshow(corr, title="Feature Correlation", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ══════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")

    # Accuracy + CV + AUC cards
    cols = st.columns(3)
    for i,(name,res) in enumerate(model_results.items()):
        cols[i].metric(name,
                       f"{res['accuracy']*100:.2f}%",
                       f"CV: {res['cv_mean']*100:.2f}% | AUC: {res['auc']:.3f}")

    st.markdown("---")

    # ROC Curve — all models
    st.subheader("📈 ROC Curve Comparison")
    fig = go.Figure()
    for name, res in model_results.items():
        fig.add_trace(go.Scatter(x=res['fpr'], y=res['tpr'],
                                 name=f"{name} (AUC={res['auc']:.3f})", mode='lines'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             line=dict(dash='dash', color='gray'), name='Random'))
    fig.update_layout(xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate', title='ROC Curves')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    selected = st.selectbox("Select Model for Details", list(model_results.keys()))
    res = model_results[selected]

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        fig = px.imshow(res['cm'], text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Not Churned','Churned'], y=['Not Churned','Churned'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if selected == 'Random Forest':
            st.subheader("Feature Importance")
            imp = pd.DataFrame({'Feature': feature_cols,
                                'Importance': res['model'].feature_importances_})\
                    .sort_values('Importance', ascending=True).tail(10)
            fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

    # SHAP
    st.markdown("---")
    st.subheader("🔍 SHAP Feature Importance (Model Explainability)")
    with st.spinner("Calculating SHAP values..."):
        try:
            explainer = shap.TreeExplainer(model_results['Random Forest']['model'])
            shap_vals = explainer.shap_values(model_results['Random Forest']['X_test'][:100])
            sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            vals = np.abs(np.array(sv)).mean(0)
            shap_df = pd.DataFrame({'Feature': feature_cols, 'SHAP': vals})\
                        .sort_values('SHAP', ascending=True).tail(10)
            fig = px.bar(shap_df, x='SHAP', y='Feature', orientation='h',
                         title="SHAP Values — Top 10 Features",
                         color='SHAP', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ══════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("🔮 Predict Customer Churn")

    c1,c2,c3 = st.columns(3)
    with c1:
        customer_id      = st.text_input("Customer ID", "CUST-001")
        tenure           = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges  = st.slider("Monthly Charges ($)", 0.0, 150.0, 65.0)
        total_charges    = st.slider("Total Charges ($)", 0.0, 9000.0, float(monthly_charges*tenure))
    with c2:
        gender      = st.selectbox("Gender", ["Male","Female"])
        senior      = st.selectbox("Senior Citizen", ["No","Yes"])
        partner     = st.selectbox("Partner", ["Yes","No"])
        dependents  = st.selectbox("Dependents", ["No","Yes"])
    with c3:
        contract  = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        internet  = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        payment   = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    selected_model = st.selectbox("Choose ML Model", list(model_results.keys()))

    if st.button("🔮 Predict Churn"):
        inp = {
            'gender': 1 if gender=='Male' else 0,
            'SeniorCitizen': 1 if senior=='Yes' else 0,
            'Partner': 1 if partner=='Yes' else 0,
            'Dependents': 1 if dependents=='Yes' else 0,
            'tenure': tenure,
            'PhoneService':1,'MultipleLines':1,
            'InternetService': ['DSL','Fiber optic','No'].index(internet),
            'OnlineSecurity':1,'OnlineBackup':1,'DeviceProtection':1,
            'TechSupport':1,'StreamingTV':1,'StreamingMovies':1,
            'Contract': ['Month-to-month','One year','Two year'].index(contract),
            'PaperlessBilling': 1 if paperless=='Yes' else 0,
            'PaymentMethod': ['Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check'].index(payment),
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        inp_df     = pd.DataFrame([inp])[feature_cols]
        inp_scaled = scaler.transform(inp_df)
        model      = model_results[selected_model]['model']
        pred       = model.predict(inp_scaled)[0]
        prob       = model.predict_proba(inp_scaled)[0][1]

        save_prediction(customer_id, tenure, monthly_charges, total_charges,
                        contract, 'Churn' if pred==1 else 'No Churn', round(prob,3))

        if pred == 1:
            st.error(f"⚠️ This customer is likely to CHURN! (Probability: {prob*100:.1f}%)")
        else:
            st.success(f"✅ This customer will likely STAY! (Churn Probability: {prob*100:.1f}%)")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text':"Churn Probability %"},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':"red" if prob>0.5 else "green"},
                   'steps':[{'range':[0,50],'color':'#d4edda'},
                             {'range':[50,100],'color':'#f8d7da'}]}))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — BULK PREDICT
# ══════════════════════════════════════════════════════════
elif page == "📤 Bulk Predict":
    st.title("📤 Bulk Customer Churn Prediction")
    st.markdown("Upload a CSV file with customer data to predict churn for all customers at once!")

    st.download_button("📥 Download Sample CSV Template",
                       data=pd.DataFrame(columns=['customerID','gender','SeniorCitizen','Partner',
                           'Dependents','tenure','PhoneService','MultipleLines','InternetService',
                           'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                           'StreamingTV','StreamingMovies','Contract','PaperlessBilling',
                           'PaymentMethod','MonthlyCharges','TotalCharges']).to_csv(index=False),
                       file_name="sample_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"**{len(df_up)} customers uploaded**")
        st.dataframe(df_up.head(), use_container_width=True)

        model_choice = st.selectbox("Choose Model", list(model_results.keys()))

        if st.button("🔮 Predict All"):
            try:
                df_proc2 = df_up.copy()
                if 'Churn' in df_proc2.columns:
                    df_proc2.drop('Churn', axis=1, inplace=True)
                le = LabelEncoder()
                cats = [c for c in df_proc2.select_dtypes('object').columns if c != 'customerID']
                for col in cats:
                    df_proc2[col] = le.fit_transform(df_proc2[col].astype(str))
                df_proc2['TotalCharges'] = pd.to_numeric(df_proc2['TotalCharges'], errors='coerce').fillna(0)
                ids = df_proc2['customerID'] if 'customerID' in df_proc2.columns else pd.Series(range(len(df_proc2)))
                X_bulk = df_proc2[feature_cols] if all(c in df_proc2.columns for c in feature_cols) else df_proc2.drop(['customerID'], axis=1, errors='ignore')
                X_bulk_s = scaler.transform(X_bulk)
                model    = model_results[model_choice]['model']
                preds    = model.predict(X_bulk_s)
                probs    = model.predict_proba(X_bulk_s)[:,1]

                result_df = pd.DataFrame({
                    'CustomerID': ids.values,
                    'Churn Prediction': ['Churn' if p==1 else 'No Churn' for p in preds],
                    'Churn Probability': [f"{p*100:.1f}%" for p in probs]
                })
                st.success(f"✅ Done! {preds.sum()} customers predicted to churn out of {len(preds)}")
                st.dataframe(result_df, use_container_width=True)

                # Download results
                csv = result_df.to_csv(index=False)
                st.download_button("📥 Download Results CSV", data=csv,
                                   file_name="churn_predictions.csv", mime="text/csv")

                # Chart
                fig = px.pie(result_df, names='Churn Prediction',
                             title="Bulk Prediction Results",
                             color_discrete_sequence=['#ef553b','#00cc96'])
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 6 — HISTORY
# ══════════════════════════════════════════════════════════
elif page == "📋 History":
    st.title("📋 Prediction History")
    df_hist = get_predictions()
    if df_hist.empty:
        st.info("No predictions yet! Go to Predict page.")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Predictions", len(df_hist))
        c2.metric("Predicted Churns",  len(df_hist[df_hist['prediction']=='Churn']))
        c3.metric("Avg Churn Prob",    f"{df_hist['probability'].mean()*100:.1f}%")

        st.dataframe(df_hist, use_container_width=True)

        csv = df_hist.to_csv(index=False)
        st.download_button("📥 Download History", data=csv,
                           file_name="prediction_history.csv", mime="text/csv")

        fig = px.pie(df_hist, names='prediction', title="Prediction Distribution",
                     color_discrete_sequence=['#ef553b','#00cc96'])
        st.plotly_chart(fig, use_container_width=True)