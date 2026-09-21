# CFPB Consumer Complaint Dataset Documentation

## 1. Dataset Name
**Consumer Complaint Database** (CFPB Consumer Financial Complaints with Published Narratives)

## 2. Official Source
- **Originating Agency**: Consumer Financial Protection Bureau (CFPB), U.S. Federal Government
- **Official Portal**: [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **Data Catalog Entry**: [https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database)
- **Direct Database Archive**: [https://files.consumerfinance.gov/ccdb/complaints.csv.zip](https://files.consumerfinance.gov/ccdb/complaints.csv.zip)

## 3. Download & Acquisition Instructions
1. Navigate to the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
2. To download complaints containing narrative text, use the dataset export or historical narrative archives where consumers provided consent to publish their narrative (`Consumer consent provided?` = "Consent provided").
3. Place the downloaded CSV file in this directory:
   - Full dataset: `data/complaints.csv`
   - Sample dataset: `data/complaints_sample.csv`

## 4. Local File Handling & Git Exclusion
- **Critical Policy**: All CSV files in `data/` (`*.csv`) are strictly excluded from GitHub version control via the project `.gitignore`.
- Raw complaint databases range from hundreds of megabytes to several gigabytes and contain consumer-submitted textual disclosures. They must remain strictly local.

## 5. Actual Inspected Columns & Schema
Based on direct inspection of the downloaded CFPB dataset (`data/complaints.csv`), the file contains **19 columns** and **25,000 complaint records**:

| Column Name | Description | Role in NLP Project |
|---|---|---|
| `Complaint ID` | Unique numeric identifier for the complaint | Primary key / Record Identifier |
| `Date received` | Date the complaint was logged with CFPB | Temporal metadata |
| `Product` | High-level financial product/service category | **Target Label (Categorisation)** |
| `Sub-product` | Detailed sub-category under product | Fine-grained category |
| `Issue` | Reported problem or dispute type | Issue metadata |
| `Sub-issue` | Detailed specification of the issue | Issue metadata |
| `Consumer Complaint` | Unstructured customer narrative text | **Primary Feature (NLP Input)** |
| `Company Public Response` | Public response statement from the financial institution | Metadata |
| `Company` | Name of the responding financial institution | Entity metadata |
| `State` | Two-letter complainant state code | Geographic metadata |
| `ZIP code` | Complainant ZIP code | Geographic metadata |
| `Tags` | Special demographic tags (e.g., Servicemember, Older American) | Demographic metadata |
| `Consumer consent provided?` | Consent status for narrative publication | Audit metadata |
| `Submitted via` | Submission channel (Web, Referral, Phone, etc.) | Channel metadata |
| `Date Sent to Company` | Date CFPB dispatched complaint to company | Workflow metadata |
| `Company Response to Consumer` | Company's formal resolution classification | Resolution metadata |
| `Timely response?` | Whether the company responded within deadline | Service level metadata |
| `Consumer disputed?` | Whether the consumer disputed company response | Outcome metadata |
| `Unnamed: 18` | Trailing empty column delimiter artifact | Excluded |

## 6. Column Mapping Adaptation
Different CFPB export snapshots and historical mirrors occasionally format column names with slight variations. The project loader (`src/data_loader.py`) implements automatic column resolution for:

- **Complaint Text**:
  - `Consumer Complaint` (CFPB narrative archive format)
  - `Consumer complaint narrative` (CFPB official portal export format)
  - `complaint_what_happened` (CFPB API field name)
- **Product Category**:
  - `Product` (standard CFPB column name)
  - `product` (lowercased format)
- **Complaint ID**:
  - `Complaint ID` (standard CFPB column name)
  - `complaint_id` (lowercased format)

The loader maps these columns to standardized internal names:
`text`, `category`, and `complaint_id`.
