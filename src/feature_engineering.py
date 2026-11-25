import pandas as pd
import numpy as np


def add_delinquency_features(X):

    X = X.copy()

    delay_columns = [f"PAY_{i}" for i in range(1, 7)]

    # Convert non-positive status codes to zero
    delays = X[delay_columns].clip(lower=0)

    # Maximum delinquency severity
    X["MAX_DELAY"] = delays.max(axis=1)

    # Average delinquency severity
    X["AVG_DELAY"] = delays.mean(axis=1)

    # Number of periods with delinquency
    X["DELAY_COUNT"] = (delays > 0).sum(axis=1)

    # Consecutive delinquent periods from most recent
    X["CONSECUTIVE_DELAYS"] = (
        (delays > 0).cumprod(axis=1).sum(axis=1)
    )

    # Recent vs. older delinquency severity
    X["DELAY_TREND"] = (
        delays.iloc[:, :3].mean(axis=1)
        - delays.iloc[:, 3:].mean(axis=1)
    )

    return X


def add_billing_features(X):

    X = X.copy()

    bill_columns = [f"BILL_AMT{i}" for i in range(1, 7)]

    bills = X[bill_columns]

    # Average bill amount
    X["AVG_BILL"] = bills.mean(axis=1)

    # Maximum bill amount
    X["MAX_BILL"] = bills.max(axis=1)

    # Bill amount variability
    X["BILL_STD"] = bills.std(axis=1, ddof=0)

    # Recent vs. older bill amounts
    X["BILL_TREND"] = (
        bills.iloc[:, :3].mean(axis=1)
        - bills.iloc[:, 3:].mean(axis=1)
    )

    # Number of zero bill amounts
    X["ZERO_BILL_COUNT"] = (bills == 0).sum(axis=1)

    # Number of negative bill amounts
    X["NEGATIVE_BILL_COUNT"] = (bills < 0).sum(axis=1)

    return X


def add_payment_features(X):

    X = X.copy()

    payment_columns = [f"PAY_AMT{i}" for i in range(1, 7)]

    payments = X[payment_columns]

    # Average payment amount
    X["AVG_PAYMENT"] = payments.mean(axis=1)

    # Number of zero payments
    X["ZERO_PAYMENT_COUNT"] = (payments == 0).sum(axis=1)

    # Most recent payment is zero
    X["LAST_PAYMENT_ZERO"] = (
        payments["PAY_AMT1"] == 0
    ).astype(int)

    # Consecutive zero payments from most recent
    X["CONSECUTIVE_ZERO_PAYMENTS"] = (
        (payments == 0).cumprod(axis=1).sum(axis=1)
    )

    # Recent vs. older payment amounts
    X["PAYMENT_TREND"] = (
        payments.iloc[:, :3].mean(axis=1)
        - payments.iloc[:, 3:].mean(axis=1)
    )

    return X


def add_financial_ratio_features(X):

    X = X.copy()

    bill_columns = [f"BILL_AMT{i}" for i in range(1, 7)]

    # Avoid division by non-positive credit limits
    credit_limit = X["LIMIT_BAL"].where(X["LIMIT_BAL"] > 0)

    # Most recent credit utilization
    X["CREDIT_UTILIZATION"] = (
        X["BILL_AMT1"] / credit_limit
    )

    # Average credit utilization
    X["AVG_CREDIT_UTILIZATION"] = (
        X[bill_columns].mean(axis=1) / credit_limit
    )

    # Align payments with previous billing periods
    bills = X[
        [f"BILL_AMT{i}" for i in range(2, 7)]
    ].to_numpy()

    payments = X[
        [f"PAY_AMT{i}" for i in range(1, 6)]
    ].to_numpy()

    # Identify eligible positive bill amounts
    valid_pairs = bills > 0

    # Aggregate matched bills and payments
    total_bills = np.where(
        valid_pairs, bills, 0
    ).sum(axis=1)

    total_payments = np.where(
        valid_pairs, payments, 0
    ).sum(axis=1)

    # Calculate payment-to-bill ratio safely
    X["PAYMENT_TO_BILL_RATIO"] = np.divide(
        total_payments,
        total_bills,
        out=np.full(len(X), np.nan),
        where=total_bills > 0
    )

    return X


def build_features(X):

    X = add_delinquency_features(X)

    X = add_billing_features(X)

    X = add_payment_features(X)

    X = add_financial_ratio_features(X)

    return X