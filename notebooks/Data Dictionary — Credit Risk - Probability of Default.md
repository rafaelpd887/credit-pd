# Data Dictionary — Credit Risk / Probability of Default

**Dataset:** Default of Credit Card Clients  
**Origin:** Taiwan, 2005  
**Original source:** UCI Machine Learning Repository — Dataset ID 350  
**Objective:** Predict the probability of default (PD).

---

## 1. Dataset overview

| Information | Description |
|---|---|
| Observations | 30,000 clients |
| Predictor variables | 23 |
| Target variable | `Y` |
| Problem type | Binary classification |
| Currency | New Taiwan Dollar (NT$) |
| Missing values | None identified |
| Column data types | All stored as `int64` |

### Target variable

| Value | Meaning | Count | Percentage |
|---|---|---:|---:|
| `0` | No default in the following month | 23,364 | 77.88% |
| `1` | Default in the following month | 6,636 | 22.12% |

**Note:** The positive class (`Y = 1`) represents approximately 22.12% of the dataset. Therefore, accuracy alone is not sufficient to evaluate the model. Metrics such as ROC-AUC, Gini, KS, precision, and recall will also be important.

---

## 2. Client identification and personal characteristics

| Variable | Description | Values / Interpretation |
|---|---|---|
| `LIMIT_BAL` | Amount of credit granted to the client | Monetary value in NT$ |
| `SEX` | Client's sex | `1` = Male; `2` = Female |
| `EDUCATION` | Education level | See table below |
| `MARRIAGE` | Marital status | See table below |
| `AGE` | Client's age | Years |

### `EDUCATION` — Education level

| Code | Meaning | Documentation |
|---:|---|---|
| `1` | Graduate school | UCI and INSEAD |
| `2` | University | UCI and INSEAD |
| `3` | High school | UCI and INSEAD |
| `4` | Others | UCI and INSEAD |
| `5` | Unknown | INSEAD supplementary documentation |
| `6` | Unknown | INSEAD supplementary documentation |
| `0` | Not specified | No definition in the consulted sources |

**Note:** Codes `0`, `5`, and `6` appear in the dataset, but they are not all explained in the official UCI documentation. INSEAD interprets `5` and `6` as unknown, but does not provide a definition for `0`.

### `MARRIAGE` — Marital status

| Code | Meaning | Documentation |
|---:|---|---|
| `1` | Married | UCI and INSEAD |
| `2` | Single | UCI and INSEAD |
| `3` | Others | UCI and INSEAD |
| `0` | Not specified | No definition in the consulted sources |

---

## 3. Payment history — `PAY_*`

These variables represent the client's **repayment status in previous months**.

They are particularly important for credit risk modeling because they provide information about the client's payment behavior over time.

### Correspondence between variables and months

| Variable | Reference month |
|---|---|
| `PAY_1` | September 2005 |
| `PAY_2` | August 2005 |
| `PAY_3` | July 2005 |
| `PAY_4` | June 2005 |
| `PAY_5` | May 2005 |
| `PAY_6` | April 2005 |

**Note:** The original variable is named `PAY_0`, not `PAY_1`. The original sequence skips number 1.

### Meaning of the codes

| Code | Meaning | Interpretation |
|---:|---|---|
| `-2` | No consumption | No consumption during the period |
| `-1` | Pay duly | Payment made on time |
| `0` | Use of revolving credit | Use of revolving credit |
| `1` | Payment delay for 1 month | 1-month payment delay |
| `2` | Payment delay for 2 months | 2-month payment delay |
| `3` | Payment delay for 3 months | 3-month payment delay |
| `4` | Payment delay for 4 months | 4-month payment delay |
| `5` | Payment delay for 5 months | 5-month payment delay |
| `6` | Payment delay for 6 months | 6-month payment delay |
| `7` | Payment delay for 7 months | 7-month payment delay |
| `8` | Payment delay for 8 months | 8-month payment delay |
| `9` | Payment delay for 9 months and above | 9-month delay or longer |

**Important: provenance of these definitions**

- The official UCI documentation specifies `-1 = pay duly` and codes `1` through `9` as the number of months of payment delay.
- The INSEAD teaching page supplements this scale by defining `-2 = no consumption` and `0 = use of revolving credit`.
- The UCI documentation does not explicitly define `-2` and `0` in the consulted section.
- Code `0` should not automatically be interpreted as minimum payment: INSEAD describes it only as the use of revolving credit.

**Methodological note:** Codes `-2`, `-1`, and `0` do not represent months of payment delay. Therefore, they should not automatically be treated as the first levels of a continuous ordinal scale of delinquency.

---

## 4. Bill statement amounts — `BILL_AMT*`

These variables represent the clients' monthly bill statement amounts.

| Variable | Reference month |
|---|---|
| `BILL_AMT1` | September 2005 |
| `BILL_AMT2` | August 2005 |
| `BILL_AMT3` | July 2005 |
| `BILL_AMT4` | June 2005 |
| `BILL_AMT5` | May 2005 |
| `BILL_AMT6` | April 2005 |

| Characteristic | Description |
|---|---|
| Unit | NT$ |
| Information type | Bill statement amount |
| Negative values | Present in the dataset and should not be automatically removed |

Negative values may be associated with credits, adjustments, or other accounting situations. However, the consulted sources do not provide a specific explanation for each occurrence. Their frequency and distribution should be investigated before deciding how to handle them.

---

## 5. Previous payment amounts — `PAY_AMT*`

These variables represent the amounts of previous payments made by the client.

| Variable | Reference month |
|---|---|
| `PAY_AMT1` | September 2005 |
| `PAY_AMT2` | August 2005 |
| `PAY_AMT3` | July 2005 |
| `PAY_AMT4` | June 2005 |
| `PAY_AMT5` | May 2005 |
| `PAY_AMT6` | April 2005 |

| Characteristic | Description |
|---|---|
| Unit | NT$ |
| Information type | Previous payment amount |
| Zero values | Indicate that no payment was recorded during the period, but do not independently explain the reason |

---

## 6. Predictor variables grouped by category

| Group | Variables |
|---|---|
| Credit limit | `LIMIT_BAL` |
| Personal characteristics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Payment history | `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` |
| Bill statement amounts | `BILL_AMT1` to `BILL_AMT6` |
| Previous payment amounts | `PAY_AMT1` to `PAY_AMT6` |
| Target variable | `Y` |

### Statistical nature of the variables

| Type | Variables | Notes |
|---|---|---|
| Continuous / discrete numerical | `LIMIT_BAL`, `AGE`, `BILL_AMT*`, `PAY_AMT*` | Quantitative values |
| Nominal categorical | `SEX`, `MARRIAGE` | Categories without a natural order |
| Categorical with possible partial ordering | `EDUCATION` | Levels 1–3 suggest an order, but `4 = others` has no clear ordinal position |
| Categorical with an ordinal component | `PAY_*` | Delay codes 1–9 have an order; negative codes and zero represent distinct situations |

---

## 7. Data preparation considerations

1. **Undocumented codes:** Initially preserve `EDUCATION = 0` and `MARRIAGE = 0` as unspecified categories, without assigning invented meanings.
2. **Codes supplemented by INSEAD:** Retain the information that `EDUCATION = 5/6` means unknown according to the secondary source.
3. **Payment history:** Do not assume that all `PAY_*` codes form a simple ordinal scale. Evaluate categorical and numerical encodings in separate experiments.
4. **Negative monetary values:** Investigate them before removing or transforming them.
5. **Target imbalance:** Compare models without balancing, with class weights, and, if appropriate, with resampling techniques.
6. **Data leakage prevention:** Perform the train/test split before fitting transformations, encoders, scalers, or balancing techniques.

---

## 8. Sources and traceability

### Official source — UCI Machine Learning Repository

**Default of Credit Card Clients — Dataset ID 350**

https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients

Primary source for dataset identification, variables, and officially documented definitions.

### Supplementary source — INSEAD

**Classification for Credit Card Default — Theos Evgeniou and Spyros Zoumpoulis**

https://inseaddataanalytics.github.io/INSEADAnalytics/CourseSessions/ClassificationProcessCreditCardDefault.html

Supplementary source used to clarify codes that are not fully described in the UCI documentation, particularly `PAY_* = -2`, `PAY_* = 0`, and `EDUCATION = 5/6`.

### Documentation convention used in this project

- **UCI:** Official dataset documentation.
- **INSEAD:** Supplementary interpretation.
- **Not documented:** Values for which no definition was found in the consulted sources.

This distinction is maintained to avoid attributing definitions to UCI that only appear in supplementary documentation.