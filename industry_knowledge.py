# Industry knowledge base
# Each industry has: typical business problems + common KPIs analysts track
# This context is fed to the AI so it gives industry-specific suggestions

INDUSTRY_KNOWLEDGE = {

    "Retail & E-commerce": {
        "problems": "Sales forecasting, inventory management, customer behavior analysis, cart abandonment, seasonal demand",
        "kpis": [
            "Gross Margin %",
            "Average Order Value (AOV)",
            "Inventory Turnover Ratio",
            "Conversion Rate",
            "Customer Lifetime Value (CLV)",
            "Same-Store Sales Growth",
            "Sell-Through Rate",
        ],
    },

    "Banking & Finance": {
        "problems": "Fraud detection, loan approval analysis, customer retention, credit risk, regulatory compliance",
        "kpis": [
            "Net Interest Margin (NIM)",
            "Non-Performing Asset (NPA) Ratio",
            "Customer Acquisition Cost (CAC)",
            "Return on Assets (ROA)",
            "Cost-to-Income Ratio",
            "Loan-to-Deposit Ratio",
            "Customer Churn Rate",
        ],
    },

    "Healthcare": {
        "problems": "Patient outcome analysis, hospital resource optimization, readmission reduction, cost control",
        "kpis": [
            "Average Length of Stay (ALOS)",
            "Readmission Rate",
            "Bed Occupancy Rate",
            "Patient Satisfaction Score",
            "Mortality Rate",
            "Cost per Patient",
            "Average Wait Time",
        ],
    },

    "Manufacturing": {
        "problems": "Production planning, quality control, cost reduction, equipment downtime, demand forecasting",
        "kpis": [
            "Overall Equipment Effectiveness (OEE)",
            "First Pass Yield (FPY)",
            "Capacity Utilization",
            "Throughput",
            "Scrap Rate",
            "Manufacturing Cost Per Unit",
            "On-Time Delivery Rate",
            "Machine Downtime Rate",
        ],
    },

    "Supply Chain & Logistics": {
        "problems": "Delayed deliveries, high transportation costs, inventory shortages, excess inventory, warehouse inefficiencies",
        "kpis": [
            "On-Time Delivery Rate",
            "Perfect Order Rate",
            "Inventory Turnover",
            "Order Fulfillment Rate",
            "Transportation Cost per Shipment",
            "Stockout Rate",
            "Demand Forecast Accuracy",
        ],
    },

    "Technology & SaaS": {
        "problems": "Customer churn, low product adoption, revenue leakage, poor user engagement",
        "kpis": [
            "Monthly Recurring Revenue (MRR)",
            "Customer Churn Rate",
            "Customer Lifetime Value (CLV)",
            "Customer Acquisition Cost (CAC)",
            "Net Revenue Retention (NRR)",
            "Average Revenue Per User (ARPU)",
            "Feature Adoption Rate",
            "Trial Conversion Rate",
        ],
    },

    "Insurance": {
        "problems": "High claim costs, fraudulent claims, customer retention issues, risk mispricing",
        "kpis": [
            "Loss Ratio",
            "Claim Settlement Time",
            "Customer Retention Rate",
            "Policy Renewal Rate",
            "Fraud Detection Rate",
            "Underwriting Profit Margin",
            "Premium Growth Rate",
        ],
    },



}