# E-Commerce Fulfilment Analysis

A Python data analysis project exploring courier performance, shipping efficiency and refund-related losses for an online marketplace, with a focus on using data to inform business decisions and identify opportunities for operational improvement.

## Project Overview

The analysis uses three datasets containing orders, shipments and refunds to evaluate fulfilment performance and investigate the factors contributing to refund-related losses.

The project is divided into two main areas:

### 1. Exploratory & Statistical Analysis

- Analysing shipping times, order values and refund rates across couriers, regions and product categories.
- Performing a chi-square test to assess whether differences in refund rates between the best and worst-performing product categories are statistically significant.
- Calculating net revenue per parcel, accounting for orders containing multiple shipments.

### 2. Business Analysis & Recommendations

- **Single Courier Strategy:** Comparing delivery speed, delivery rates, returns and refund-related losses to recommend the most suitable courier.
- **Segmented Courier Strategy:** Evaluating performance across regions and product categories to recommend a more targeted courier allocation strategy.
- **Operational Improvements:** Investigating the main reasons for refunds and recommending practical changes to reduce losses.

## Key Findings

- No single courier performed best across all metrics, highlighting trade-offs between delivery speed, delivery success and refund-related losses.
- Courier refund performance varied across regions and product categories, suggesting opportunities for a more targeted allocation strategy.
- Incorrect or damaged items accounted for 45.8% of refunds, while late deliveries accounted for a further 28.9%, highlighting opportunities to improve fulfilment processes.

## Assumptions & Limitations

Refunds were recorded at order level, while shipments were recorded at parcel level. To avoid ambiguous attribution, courier-level refund comparisons were restricted to orders handled by a single courier, representing approximately 90% of all orders.

Some calculations also required assumptions, particularly when allocating order-level revenue across multiple parcels. These assumptions and their implications are explained in the analysis.

## Tools & Libraries

- **Python** — Programming and analysis
- **Pandas** — Data manipulation, merging and aggregation
- **Matplotlib** — Data visualisation
- **SciPy** — Statistical hypothesis testing

## Code & Results

- **`fulfilment_analysis.py`** — Python code for data analysis, statistical testing and visualisations.
- **`business_insights_and_recommendations.pdf`** — Supporting charts, key findings and business recommendations.
- **`sample_data/`** — Small, synthetic datasets included to illustrate the structure of the original data.

**Data note:** The original datasets are not publicly available. The sample files contain fictional data for illustrative purposes only and do not reproduce the findings presented in the report.
