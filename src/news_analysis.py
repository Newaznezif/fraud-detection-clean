import pandas as pd
import re

class NewsData:
    """
    Class to handle loading, cleaning, and analyzing news CSVs.
    Provides utilities for text cleaning, publication counts,
    and basic exploratory analysis.
    """

    def __init__(self, file_path, date_col="date", text_col="headline"):
        self.file_path = file_path
        self.date_col = date_col
        self.text_col = text_col
        self.df = None
        self.load()

    def load(self) -> pd.DataFrame:
        """Load CSV and parse date column."""
        df = pd.read_csv(self.file_path)

        # Validate date column
        if self.date_col not in df.columns:
            raise KeyError(f"Expected column '{self.date_col}' in CSV")

        # Parse and clean date
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = df.dropna(subset=[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)

        # Standardize column name
        df = df.rename(columns={self.date_col: "timestamp"})
        self.df = df
        return df

    def clean_text(self, column=None) -> pd.DataFrame:
        """Clean text column: lowercase, remove special chars, normalize whitespace."""
        if column is None:
            column = self.text_col
        if column not in self.df.columns:
            raise KeyError(f"Expected text column '{column}' in DataFrame")

        df = self.df.copy()
        df[column] = (
            df[column]
            .astype(str)
            .apply(lambda x: re.sub(r"[^A-Za-z0-9\s]", "", x))  # remove special chars
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)  # normalize spaces
        )
        self.df = df
        return df

    def daily_publication_counts(self) -> pd.DataFrame:
        """Count number of articles per day."""
        counts = (
            self.df.groupby(self.df["timestamp"].dt.date)
            .size()
            .reset_index(name="article_count")
        )
        return counts

    def top_publishers(self, publisher_col="publisher", n=10) -> pd.DataFrame:
        """Return top N publishers by article count."""
        if publisher_col not in self.df.columns:
            raise KeyError(f"Expected publisher column '{publisher_col}' in DataFrame")
        return (
            self.df[publisher_col]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "publisher", publisher_col: "count"})
            .head(n)
        )

    def headline_length_summary(self, column=None) -> pd.DataFrame:
        """Return summary stats of headline lengths."""
        if column is None:
            column = self.text_col
        if column not in self.df.columns:
            raise KeyError(f"Expected text column '{column}' in DataFrame")

        lengths = self.df[column].astype(str).apply(len)
        summary = lengths.describe().to_frame().T
        summary.index = ["headline_length"]
        return summary
