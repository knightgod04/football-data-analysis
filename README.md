# Football Data Analysis: Manchester United (Post-Ferguson Era)

This repository contains a comprehensive statistical analysis of Manchester United’s football performance in the post-Sir Alex Ferguson era, with selected comparisons to Manchester City where relevant.  
The project focuses on applying rigorous statistical methods to football performance data using **R**.

---

## Objectives

- To analyse Manchester United’s on-field performance using advanced statistical techniques
- To study attacking and defensive metrics such as xG, xGA, possession, shooting accuracy, and passing accuracy
- To evaluate managerial and tactical impacts on performance
- To apply formal hypothesis testing and modelling techniques in a football analytics context

---

## Data Description

The analysis uses match-level and season-level football data extracted from structured Excel sheets, including:

- Expected Goals (xG) and Expected Goals Against (xGA)
- Home and Away performance metrics
- Possession, passing accuracy, and shooting accuracy
- Managerial and formation-based summaries
- Match outcomes (Wins, Draws, Losses)

---

## Statistical Methods Used

The project applies the following statistical techniques:

- Exploratory Data Analysis (EDA)
- Shapiro–Wilk test for normality
- Welch’s two-sample t-tests
- Variance tests (F-tests)
- Chi-square tests of independence
- One-way ANOVA
- Kruskal–Wallis test and post-hoc Dunn test
- Two-way ANCOVA
- Multiple Linear Regression (MLR)
- Time series analysis (ACF, AR models, ADF test)
- Poisson and Negative Binomial regression models
- Correlation analysis and diagnostic plots

---

## Tools and Technologies

- **R**
- R packages: `ggplot2`, `dplyr`, `readxl`, `car`, `lmtest`, `forecast`, `tseries`, `MASS`, `FSA`, `corrplot`
- Microsoft Excel (data storage and preprocessing)

---

## Key Insights

- Manchester United’s attacking and defensive performances show significant variation across seasons
- Statistical tests indicate differences in performance metrics when compared across teams and tactical contexts
- Managerial effects and formations play a measurable role in expected goal metrics
- Advanced models such as regression and time series methods provide deeper insight beyond basic summary statistics

---

## Repository Structure
├── data/ # Raw and processed datasets
├── r-scripts/ # R scripts for analysis and modelling
├── notebooks/ # Optional R Markdown / notebooks
├── outputs/ # Plots and result outputs
├── report/ # Methodology and documentation
└── README.md


---

## Author

**Anubhav Roy**  
Statistics student with interests in football analytics, statistical modelling, and data-driven sports analysis.

---

## Notes

This project is academic in nature and is intended to demonstrate the application of statistical reasoning and modelling techniques to real-world football data.
