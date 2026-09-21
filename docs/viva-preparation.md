# Academic Viva & Technical Defense Preparation

**Project:** Customer Complaint NLP — CFPB Complaint Categorisation  
**Focus Area:** Taxonomy-Aware Complaint Classification & Task Formulation  

---

## 1. Motivation & Core Problem Formulation

### Q1: Why did you investigate taxonomy normalization instead of just training a larger model or continuing hyperparameter tuning?
**A:**  
Error analysis on the held-out 5,000-record test set revealed that **608 out of 1,522 total errors (39.9%)** occurred between specific pairs of closely related CFPB product categories—most notably between *"Credit reporting"* and *"Credit reporting, credit repair services, or other personal consumer reports"* (314 errors), and between *"Credit card"* and *"Credit card or prepaid card"* (148 errors). 

Rather than blindly tuning regularization constants or adding model parameters to force separation between labels that share identical underlying grievance texts, we examined the ground-truth target taxonomy. Official CFPB documentation confirms that these categories are **historical administrative variants** resulting from the Bureau's April 2017 and 2019 form restructurings. The database preserves submission-time labels without backfilling. Consequently, the observed classification difficulty was largely an artifact of an administrative taxonomy shift rather than linguistic ambiguity in consumer narratives.

---

### Q2: Does merging categories automatically make the model "better"?
**A:**  
**No.** Merging categories fundamentally changes the classification task itself. By reducing the number of target classes (from 18 to 11 or 10), the label granularity decreases and the chance level changes. 

Higher accuracy or F1 on a normalized taxonomy does not indicate that the classifier became smarter; it indicates that the task no longer penalizes the model for failing to separate historically synonymous labels. Therefore, any performance metrics on a normalized taxonomy must be evaluated alongside a rigorous **Information Loss Analysis** that documents the exact product and regulatory distinctions surrendered.

---

### Q3: Did you use category labels or metadata as input features?
**A:**  
**No.** We maintained strict anti-leakage safeguards:
1. The only input feature to the classifier is the **consumer complaint narrative text**.
2. Metadata fields (Product, Sub-product, Issue, State, Company, Date) are never provided as inputs.
3. The TF-IDF vectorizers (237,148 combined word and character n-gram features) were fitted strictly on the 20,000-record training pool prior to any test-set inference.
4. The test set was untouched during both vectorization and model fitting.

---

### Q4: Why is this approach more academically defensible than repeated hyperparameter search?
**A:**  
In applied machine learning, tuning parameters like $C$ or trying dozens of classifiers without diagnosing errors treats the task formulation as an unalterable ground truth. When the target labels contain administrative noise or historical redundancies, optimizing a loss function against conflicting labels leads to overfitting or arbitrary decision boundaries. 

Investigating task formulation addresses the **validity of the classification problem itself**. Disentangling *representational limitations* from *label taxonomy overlap* provides actionable insights into the limits of classical NLP on real-world regulatory data.

---

## 2. Experimental Rigor & Taxonomy Design

### Q5: How did you define the normalized taxonomies to avoid confirmation bias?
**A:**  
To prevent "performance-driven mapping" (where labels are merged simply because doing so artificially inflates test F1), both taxonomy variants were:
1. **Pre-defined and frozen** in version-controlled configuration files (`config/taxonomy_v1_conservative.json` and `config/taxonomy_v2_broad.json`) before executing the experiment.
2. **Grounded in primary CFPB documentation** (the April 24, 2017 *Summary of product and sub-product changes* and official *Past product and issue changes* notices).
3. **Categorized by evidence types**:
   - **Type A:** Direct official CFPB renames and mergers (e.g. *Bank account or service* $\rightarrow$ *Checking or savings account*).
   - **Type B:** Explicit lexical sub-string containment.
   - **Type C:** Empirical confusion patterns observed on the baseline test set.

Neither taxonomy variant was chosen or altered based on test performance.

---

### Q6: What is the distinction between your two taxonomy variants (Conservative vs. Broad)?
**A:**  
The central question distinguishing the two variants is how to handle **Consumer Loan**:
- **v1 Conservative (11 categories):** Keeps `Consumer Loan` as an independent category. The rationale is that in 2017 the CFPB split out vehicle and payday loans, leaving residual consumer loans (typically $1,000–$25,000 installment loans) with distinct regulatory considerations under the Truth in Lending Act (TILA), distinct from short-term small-dollar payday loans.
- **v2 Broad (10 categories):** Merges `Consumer Loan` into `Consumer & Small Dollar Loans`. The rationale is that `Consumer Loan` was the historical parent category from which the 2017 installment and personal loan sub-products were split.

We report both without declaring one "better."

---

### Q7: How do you mathematically dissect the source of error reduction?
**A:**  
We isolate two distinct factors:
$$\text{Total Error Reduction} = \Delta_{\text{task collapse}} + \Delta_{\text{retraining}}$$

1. **Mechanical Task Collapse ($\Delta_{\text{task collapse}}$):** Errors that disappear purely because two labels are mapped to the same target class (evaluated by post-hoc remapping of the 18-category model's predictions). This accounts for **608 errors (39.9%)** in the conservative variant and **635 errors (41.7%)** in the broad variant.
2. **Retraining Effect ($\Delta_{\text{retraining}}$):** The net change in errors when the classifier is retrained with a consolidated objective function, allowing it to optimize decision boundaries without penalizing intra-variant ambiguity.

---

## 3. Information Loss & Practical Trade-offs

### Q8: What specific information is lost in the normalized taxonomies?
**A:**  
We explicitly document four major losses:
1. **Credit Reporting:** Collapses the pre-2019 narrow credit bureau dispute scope with the broader post-2019 scope that includes tenant screening and credit repair companies.
2. **Card Products:** Collapses revolving credit lines (Credit cards, governed by TILA) with stored-value cards (Prepaid cards, governed by EFTA). Prepaid cards serve unbanked consumers and lack credit underwriting.
3. **Banking Accounts:** Collapses historical ancillary banking services (safe deposit boxes, money orders) into retail checking/savings.
4. **Loans (Broad only):** Collapses multi-year installment loans ($5,000+) with two-week payday advances ($300–$500), losing the ability to isolate predatory payday lending complaints.

---

### Q9: If you were deploying this in a financial institution, which model would you deploy?
**A:**  
It depends on the downstream operational objective:
- If the goal is **automated triage to specialized operational teams** (e.g., Credit Bureau Disputes vs. Mortgage Servicing), the **11-category normalized model** is preferable because complaints within the merged clusters go to the same operational department anyway, and the model achieves ~82% accuracy with much higher confidence.
- If the goal is **regulatory reporting to the CFPB** where complaints must match historical federal filings exactly, the **original 18-category model** must be retained, but paired with a confidence threshold routing ambiguous predictions (probabilities $< 0.40$) to human reviewers.
