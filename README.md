# Databricks dbt Analytics Project

## Overview
End-to-end analytics project built using **dbt on Databricks**, following modern **Bronze → Silver → Gold** data architecture.

The project uses a **CSV-based customer dataset** loaded into Databricks and models a retail domain including:
- customers
- sales
- returns
- products
- stores
- dates

The final output is a **One Big Table (OBT)** designed for BI tools such as **Tableau** to analyze key business KPIs.

## Tech Stack
- Databricks (Delta Lake)
- dbt Core
- SQL & Jinja
- Git & GitHub

## Key Features
- Bronze–Silver–Gold modeling
- SCD Type 2 using dbt snapshots
- Reusable macros
- Data quality tests
- Secure handling of secrets via environment variables
- Analytics-focused data modeling

## Data Architecture
CSV Source Files
|
v
+----------------+
| Bronze |
| Raw Ingest |
+----------------+
|
v
+-----------------------------+
| Silver |
| - Deduplication |
| - Type casting |
| - Minor transformations |
| - dbt snapshots (SCD 2) |
+-----------------------------+
|
v
+-------------------------------------+
| Gold |
| One Big Table (OBT) |
| Analytics & BI-ready |
+-------------------------------------+


## Layer Details

### Bronze
- Raw ingestion of CSV data into Databricks
- No business logic applied
- Preserves original source structure

### Silver
- Cleans and standardizes Bronze data
- Deduplication and minor transformations
- **dbt snapshot implemented on Customers table**
  - Tracks historical changes (Slowly Changing Dimension Type 2)

### Gold
- Builds a **One Big Table (OBT)**
- Combines customers, sales, returns, products, stores, and dates
- Optimized for BI tools and KPI analysis


## Running the Project

This project is designed to run on **Databricks**.

To execute the models, the following environment variables are required:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

> Credentials are intentionally **not committed** to the repository for security reasons.

Once configured, run:

```bash
dbt build

