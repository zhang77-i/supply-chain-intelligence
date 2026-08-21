# Cost-sensitive Demand Forecasting and Multi-Warehouse Inventory Optimization

An end-to-end supply chain optimization framework integrating demand forecasting, uncertainty quantification, inventory decision-making, and multi-warehouse allocation.

## Architecture

```
Raw Supply Chain Data
        |
        v
SQL Data Pipeline
        |
        v
SKU Demand Forecasting
(Croston-SBA / TSB / LightGBM Quantile)
        |
        v
Demand Uncertainty Modeling
        |
        v
Newsvendor Inventory Optimization
        |
        v
CP-SAT Multi-Warehouse Allocation
```

## Key Components

- Intermittent demand forecasting for long-tail SKUs
- Quantile forecasting for demand risk estimation
- Cost-sensitive inventory optimization
- Multi-warehouse allocation optimization
- Rolling backtesting evaluation

## Project Structure

```
src/
├── forecasting
├── inventory
├── allocation
└── evaluation

sql/
scripts/
tests/
```

## Methods

### Forecasting

- Moving Average baseline
- Croston-SBA
- TSB
- LightGBM quantile regression

### Inventory Optimization

- Newsvendor model
- Safety stock optimization
- Multi-warehouse allocation

### Optimization

- OR-Tools CP-SAT

## Goal

Transform demand prediction into actionable supply chain decisions by connecting forecasting uncertainty with inventory optimization.
