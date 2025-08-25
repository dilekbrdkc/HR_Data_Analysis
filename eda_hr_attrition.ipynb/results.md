# 📊 Results & Analysis / Risultati e Analisi

> This file presents detailed analysis results and model evaluations for the HR attrition prediction project.


## 🔍 Key Findings / Risultati Chiave

- **Employee Attrition**: Age and job satisfaction emerged as key factors. Employees aged **30–40** with **low job satisfaction** showed a higher risk of attrition.  
  - **Abbandono dei dipendenti**: L’età e la soddisfazione lavorativa sono risultati fattori chiave. I dipendenti tra i **30 e i 40 anni** con **bassa soddisfazione lavorativa** mostrano un rischio maggiore di abbandono.

- **Department-wise Attrition**: The **Sales** department had the highest attrition rate.  
  - **Abbandono per dipartimento**: Il reparto **vendite** ha registrato il tasso di abbandono più elevato.

- **Model Performance**: Without addressing class imbalance, models failed to effectively detect attrition. After applying **SMOTE**, both **Decision Tree** and **Logistic Regression** showed improved recall for attrition cases.  
  - **Prestazioni del modello**: I modelli iniziali non hanno rilevato efficacemente l’abbandono. Dopo l’applicazione di **SMOTE**, sia il modello **Decision Tree** che la **Regressione Logistica** hanno mostrato miglioramenti nel recall dei casi di abbandono.

---

## 1. Exploratory Data Analysis (EDA) / Analisi Esplorativa dei Dati

We explored attrition rates by age, department, overtime, job role, etc. Key observations included:

- High attrition in Sales and Human Resources departments
- Strong correlation between overtime and attrition
- Low job satisfaction often associated with attrition

Abbiamo esaminato i tassi di abbandono per età, reparto, straordinari, ruolo, ecc. Le osservazioni chiave includono:

- Alto tasso di abbandono in Vendite e Risorse Umane
- Forte correlazione tra straordinari e abbandono
- Bassa soddisfazione lavorativa spesso associata all’abbandono

*📌 Suggestion: Include visualizations like heatmaps, bar plots, and box plots here if you have them.*

---


## 🔍 Demographic & Behavioral Insights from Power BI  
## 🔍 Approfondimenti Demografici e Comportamentali da Power BI

These insights were derived using Power BI visualizations.  
Queste informazioni sono state ottenute utilizzando visualizzazioni in Power BI.

- **Attrition by Gender**:  
  - Female employees: **14.80%** attrition  
  - Male employees: **17.01%** attrition  
  - → Male employees appear slightly more likely to leave.  
  - **Abbandono per genere**:  
    - Donne: **14,80%**  
    - Uomini: **17,01%**  
    - → Gli uomini sembrano leggermente più propensi a lasciare l'azienda.

- **Income by Gender**:  
  - Number of high-income employees:  
    - Women: **558**  
    - Men: **882**  
  - **Reddito per genere**:  
    - Donne: **558**  
    - Uomini: **882**

- **Monthly Income Comparison**:  
  - Attrition cases: average monthly income ≈ **4.8K**  
  - Retained employees: average monthly income ≈ **6.8K**  
  - → Lower income levels are linked with higher attrition.  
  - **Confronto reddito mensile**:  
    - Casi di abbandono: ≈ **4.8K**  
    - Dipendenti rimasti: ≈ **6.8K**  
    - → I redditi inferiori sono associati a maggiore abbandono.

- **Business Travel Impact**:  
  - Travel Frequently: **24.91%** attrition  
  - Travel Rarely: **14.96%** attrition  
  - Non-Travel: **0.8%** attrition  
  - → Frequent travel may contribute to burnout and higher attrition.  
  - **Impatto dei viaggi di lavoro**:  
    - Viaggi frequenti: **24,91%**  
    - Viaggi occasionali: **14,96%**  
    - Nessun viaggio: **0,8%**  
    - → I viaggi frequenti possono contribuire al burnout e all'abbandono.

--- 

## 2. Machine Learning Models / Modelli di Apprendimento Automatico

### 2.1 Without SMOTE / Senza SMOTE

| Model               | Accuracy | Precision (Attrition) | Recall (Attrition) | F1 Score (Attrition) |
|---------------------|----------|------------------------|--------------------|----------------------|
| Decision Tree        | 0.77     | 0.16                   | 0.18               | 0.17                 |
| Logistic Regression  | 0.86     | 0.38                   | 0.08               | 0.13                 |

> ❗ These models had high overall accuracy but **very low recall** for attrition cases due to **class imbalance**.
>  
> Questi modelli avevano un'alta accuratezza complessiva, ma un **recall molto basso** per i casi di abbandono a causa dello **sbilanciamento delle classi**.

---

### 2.2 With SMOTE / Con SMOTE

After applying SMOTE to balance the class distribution:

| Model                      | Accuracy | Precision (Attrition) | Recall (Attrition) | F1 Score (Attrition) |
|----------------------------|----------|------------------------|--------------------|----------------------|
| Decision Tree (SMOTE)      | 0.73     | 0.20                   | 0.33               | 0.25                 |
| Logistic Regression (SMOTE)| 0.75     | 0.25                   | 0.44               | 0.32                 |

> ✅ **Recall improved significantly**, making models more useful for identifying potential attrition.
>  
> ✅ Il **recall è migliorato notevolmente**, rendendo i modelli più utili per identificare il rischio di abbandono.

---

## 3. Discussion / Discussione

- SMOTE successfully addressed class imbalance, improving model sensitivity to attrition.
- Logistic Regression with SMOTE showed the **best recall (44%)**, making it suitable for identifying potential leavers.
- While accuracy decreased slightly, this trade-off is acceptable when **detecting rare events** like employee attrition.

- SMOTE ha risolto il problema dello sbilanciamento, migliorando la sensibilità dei modelli.
- La Regressione Logistica con SMOTE ha ottenuto il **miglior recall (44%)**, risultando adatta per identificare i dipendenti a rischio.
- Sebbene l'accuratezza sia leggermente diminuita, questo compromesso è accettabile nella **rilevazione di eventi rari**.

---

## 4. Limitations and Future Work / Limitazioni e Lavori Futuri

- The dataset only covers IBM employees; generalizability to other organizations is uncertain.
- Only basic ML models (Decision Tree, Logistic Regression) were used.  
- Future work can explore:
  - Advanced models (Random Forest, XGBoost)
  - Feature engineering
  - Hyperparameter tuning
  - Better interpretability tools (e.g., SHAP values)

- Il dataset copre solo i dipendenti IBM; la generalizzazione ad altre aziende non è garantita.
- Sono stati utilizzati solo modelli base (Decision Tree, Logistic Regression).  
- I lavori futuri possono esplorare:
  - Modelli avanzati (Random Forest, XGBoost)
  - Feature engineering
  - Ottimizzazione degli iperparametri
  - Strumenti di interpretazione migliori (es. SHAP)

---

*🔗 For full implementation, code, and visualizations, refer to the Jupyter notebooks in the project repository.*  
*🔗 Per l'implementazione completa, il codice e le visualizzazioni, consultare i notebook del progetto.*
