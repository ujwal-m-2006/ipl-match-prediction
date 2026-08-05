"""IPL Match Prediction & Analytics System.

A production-grade pipeline that ingests Indian Premier League data from the
official iplt20.com feeds (supplemented by Cricsheet for pre-2019 seasons),
stores it in a relational database, engineers features, trains and compares a
suite of machine-learning models, and serves predictions through both a
Streamlit dashboard and a FastAPI service.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
