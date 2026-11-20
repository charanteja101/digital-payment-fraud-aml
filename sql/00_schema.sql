-- PaySim Fraud & AML: database schema (SQLite syntax; also valid in PostgreSQL).
-- Load the Kaggle file with:  python sql/load_sqlite.py

DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    transaction_id    INTEGER PRIMARY KEY,
    step              INTEGER NOT NULL,   -- 1 step = 1 hour; 744 steps = 30 days
    type              TEXT NOT NULL,      -- CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    amount            REAL NOT NULL,
    name_orig         TEXT,
    old_balance_orig  REAL,
    new_balance_orig  REAL,
    name_dest         TEXT,               -- names starting with M are merchants
    old_balance_dest  REAL,
    new_balance_dest  REAL,
    is_fraud          INTEGER NOT NULL,
    is_flagged_fraud  INTEGER NOT NULL    -- the simulator's own rule: transfer over 200,000
);

-- Optional: model scores for the test period (exported from your own scoring code, see dashboard/README.md).
DROP TABLE IF EXISTS scored_transactions;
CREATE TABLE scored_transactions (
    row_id             INTEGER PRIMARY KEY,
    step               INTEGER,
    hour_of_day        INTEGER,
    type               TEXT,
    amount             REAL,
    actual_fraud       INTEGER NOT NULL,
    fraud_probability  REAL NOT NULL,
    flagged            INTEGER NOT NULL,
    existing_flag      INTEGER,
    outcome            TEXT NOT NULL      -- True positive / False positive / False negative / True negative
);
