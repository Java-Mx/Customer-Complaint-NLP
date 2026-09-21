# Dataset Documentation

## Dataset Name
**Consumer Complaint Database**

## Official Source
- **Provider**: Consumer Financial Protection Bureau (CFPB), a U.S. government agency.
- **Official URL**: [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **Data Catalog URL**: [https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database)

## How to Download
1. Visit the official CFPB complaints portal at:
   `https://www.consumerfinance.gov/data-research/consumer-complaints/`
2. Apply any relevant filters if you wish to download a subset (e.g., complaints with narrative text only, or specific date windows).
3. Click the **Export the data** or **Download data** button and select **CSV format**.
4. Alternatively, use the CFPB public Open Data API or export direct snapshot CSVs as documented on their website.

## Where to Place the Downloaded Dataset
- Place the downloaded CSV file into this directory:
  `data/complaints.csv`
- For local testing and development with smaller slices, save the sampled dataset as:
  `data/complaints_sample.csv`
- **Important**: The raw dataset files (`*.csv`) are ignored by Git via `.gitignore` and should never be committed to the repository due to file size constraints.

## Expected Input Format
- **Format**: Delimited text file (`.csv`), encoded in UTF-8.
- **Primary Text Feature Column**:
  - `Consumer complaint narrative`: Contains the raw unstructured text of the customer's complaint submission.
- **Primary Target / Category Column**:
  - `Product`: The high-level financial product/service category (e.g., *Credit reporting, repair, or other*, *Debt collection*, *Mortgage*, *Credit card or prepaid card*, *Student loan*, *Checking or savings account*).

## Official CFPB Schema Reference
The complete CFPB dataset schema includes the following standard fields:

| Column Name | Description | Role in Pipeline |
|---|---|---|
| `Date received` | Date the complaint was received by CFPB | Metadata |
| `Product` | High-level product category | **Target Label** (Categorisation) |
| `Sub-product` | Granular sub-category of the product | Metadata / Fine-grained label |
| `Issue` | The issue the consumer reported | Context / Optional label |
| `Sub-issue` | Detailed breakdown of the issue | Context |
| `Consumer complaint narrative` | Unstructured text submitted by consumer | **Input Text** (Feature for NLP) |
| `Company public response` | Optional company response text | Metadata |
| `Company` | Name of the financial institution | Metadata |
| `State` | Two-letter state code of complainant | Demographic metadata |
| `ZIP code` | Complainant ZIP code | Demographic metadata |
| `Tags` | Special demographic tags (e.g., Servicemember, Older American) | Metadata |
| `Consumer consent provided?` | Whether consumer consented to publish narrative | Filter (requires 'Consent provided') |
| `Submitted via` | Submission channel (Web, Referral, Phone, Postal mail, Fax) | Metadata |
| `Date sent to company` | Date CFPB forwarded complaint to company | Metadata |
| `Company response to consumer` | Company's formal resolution classification | Metadata |
| `Timely response?` | Whether company responded within deadline | Metadata |
| `Consumer disputed?` | Whether consumer disputed the resolution | Metadata |
| `Complaint ID` | Unique numerical identifier for complaint | Identifier |

## Important Notice
- Only complaints where `Consumer complaint narrative` is non-empty (`Consumer consent provided?` = "Consent provided") are usable for text similarity and NLP classification tasks.
- Do not commit large CSV files to Git.
