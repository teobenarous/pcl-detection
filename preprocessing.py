import os
import re
import argparse
import pandas as pd
from bs4 import BeautifulSoup
import ftfy
from sklearn.model_selection import train_test_split


class PCLCleaner:
    def __init__(self, raw_path, train_ids_path, dev_ids_path, test_path):
        self.raw_path = raw_path
        self.train_ids_path = train_ids_path
        self.dev_ids_path = dev_ids_path
        self.test_path = test_path

        # (1) URLs & Artifacts
        self.url_pattern = re.compile(r'http[s]?\s*:\s*(?://|\*\*)\S+', re.IGNORECASE)
        self.artifact_pattern = re.compile(r'\*\*\d+;\d+;TOOLONG')

        # (2) Mentions & Users
        self.mention_pattern = re.compile(r'(?:^|\s)@\s*[A-Za-z0-9_]+(?:\s*[A-Za-z0-9_]+)*(?:\s*[-:]+)?(?!\S+\.\S+)')
        self.existing_user_pattern = re.compile(r'@?\[USER\][\s:;\-]*')

        # (3) Spaced Punctuation Corruptions
        self.qmark_artifact_pattern = re.compile(r'\?(?:\s*\?)+')
        self.squote_artifact_pattern = re.compile(r"'(?:\s*')+")

        # (4) Dash Normalization
        self.em_dash_pattern = re.compile(r'\s*[-—–]{2,}\s*|\s+[—–]\s+')
        self.compound_hyphen_right_space = re.compile(r'\b([A-Za-z]+)-\s+([A-Za-z]+)\b')
        self.compound_hyphen_left_space = re.compile(r'\b([A-Za-z]+)\s+-([A-Za-z]+)\b')
        self.number_dash_pattern = re.compile(r'\b(\d+)-\s+(?!\d)')

        # (5) Currency, Slashes, and Trailing Junk
        self.currency_suffix_pattern = re.compile(r'([\w\.]+)/-')
        self.word_slash_pattern = re.compile(r'\b([a-zA-Z]+)/([a-zA-Z]+)\b')
        self.trailing_junk_pattern = re.compile(r'[\s@]+$')

        self.whitespace_pattern = re.compile(r'\s+')

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)

        # (0) Initial Cleanup & Formatting
        text = ftfy.fix_text(text)
        text = BeautifulSoup(text, "html.parser").get_text(separator=' ')
        text = self.trailing_junk_pattern.sub('', text)

        # (1) URL & Artifact Masking (before punctuation normalization)
        text = self.url_pattern.sub("[URL]", text)
        text = text.replace("Share URL", '').strip()
        text = self.artifact_pattern.sub('', text)

        # (2) Fix Custom Edge Cases (Currency and Slashes)
        text = self.currency_suffix_pattern.sub(r'\1', text)
        text = self.word_slash_pattern.sub(r'\1-\2', text)

        # (3) Dash Normalization
        text = self.em_dash_pattern.sub(" - ", text)
        text = self.compound_hyphen_right_space.sub(r'\1-\2', text)
        text = self.compound_hyphen_left_space.sub(r'\1-\2', text)
        text = self.number_dash_pattern.sub(r'\1 ', text)

        # (4) Handle Spaced Punctuation Corruptions
        text = self.qmark_artifact_pattern.sub("'", text)
        text = self.squote_artifact_pattern.sub('"', text)

        # (5) Strip Dangling Sentence-Level Quotes
        text = text.strip()
        def strip_quotes(t):
            while t and t[0] in ("'", '"'): t = t[1:].strip()
            while t and t[-1] in ("'", '"'): t = t[:-1].strip()
            return t
        text = strip_quotes(text)

        # (6) Normalize Spacing for Content Quotes
        text = re.sub(r'\s*"\s*', ' " ', text)
        text = re.sub(r'("\s*)+', '"', text)

        # (7) User Masking
        text = self.mention_pattern.sub("[USER] ", text)
        text = self.existing_user_pattern.sub("[USER] ", text)

        # (8) Detokenization: Punctuation, Clitics, and Apostrophes
        text = re.sub(r'\s+([?.!,:;])', r'\1', text)

        # (8b) Unified Clitic, Contraction, and Apostrophe Fix
        def fix_all_clitics(t):
            # (A) "do n't" -> "don't"
            t = re.sub(r"\b(\w+)\s+n't\b", r"\1n't", t, flags=re.IGNORECASE)
            # (B) Spaced contractions: "you ' re", "I 'm", "Can ' t" -> "you're", "I'm", "Can't"
            t = re.sub(r"\b(\w+)\s*'\s*(re|ve|ll|d|m|s|t)\b", r"\1'\2", t, flags=re.IGNORECASE)
            # (C) Dangling possessive apostrophes: "migrants ' " -> "migrants'"
            t = re.sub(r"\b(\w+s)\s+'(?!\w)", r"\1'", t, flags=re.IGNORECASE)
            # (D) Spaced years: " ' 90" -> "'90"
            t = re.sub(r"(?:^|\s)'\s+(\d{2,4})\b", r" '\1", t)
            return t

        text = fix_all_clitics(text)

        # (8c) Separate words from quotes
        text = re.sub(r"([a-zA-Z]{2,})'([a-zA-Z]{3,})\b", r"\1' \2", text)

        # (8d) Parenthesis spacing
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)

        # (9) Final Formatting
        text = re.sub(r"'\s+([^']+?)\s+'", r"'\1'", text)
        text = self.whitespace_pattern.sub(' ', text)
        text = text.strip()

        return text

    def process_and_save(self, output_dir, seed):
        os.makedirs(output_dir, exist_ok=True)
        print("Starting Preprocessing Pipeline...")

        print(f"Loading raw data from {self.raw_path}...")
        df_raw = pd.read_csv(
            self.raw_path,
            sep='\t',
            header=None,
            names=["par_id", "art_id", "keyword", "country", "text", "label"],
            quoting=3,
            skiprows=4
        )

        df_raw["par_id"] = df_raw["par_id"].astype(str)

        # (1) Clean text & calculate word count BEFORE dropping NaNs or short texts
        print("Cleaning raw text (HTML removal, Unicode fixing, Entity Masking)...")
        # Fill NaNs temporarily just so clean_text works
        df_raw["text"] = df_raw["text"].fillna('')
        df_raw["clean_text"] = df_raw["text"].apply(self.clean_text)
        df_raw["word_count"] = df_raw["clean_text"].apply(lambda x: len(x.split()))

        # (2) Inject contextual metadata
        print("Injecting Contextual Metadata ([KEYWORD]...[COUNTRY]...[SEP])...")
        df_raw["keyword"] = df_raw["keyword"].fillna("unknown")
        df_raw["country"] = df_raw["country"].fillna("unknown")
        df_raw["text"] = "[KEYWORD] " + df_raw["keyword"] + " [COUNTRY] " + df_raw["country"] + " [SEP] " + df_raw["clean_text"]

        # Helper function to extract splits based on ID files
        def get_split_df(split_path, split_name):
            print(f"Extracting {split_name} split from {split_path}...")
            df_split_ids = pd.read_csv(split_path, sep=',', header=0)
            df_split_ids["par_id"] = df_split_ids["par_id"].astype(str)
            df_split = pd.merge(df_split_ids[["par_id"]], df_raw, on="par_id", how="inner")
            if "label" in df_split.columns:
                df_split["label"] = df_split["label"].astype(int).apply(lambda x: 0 if x < 2 else 1)
            return df_split

        # (3) Extract and save official dev split (UNFILTERED)
        df_train_official = get_split_df(self.train_ids_path, "train")
        df_dev_official = get_split_df(self.dev_ids_path, "dev")
        official_dev_out = os.path.join(output_dir, "dev_external.csv")
        df_dev_official[['par_id', 'text', 'label']].to_csv(official_dev_out, index=False)
        print(f"Saved dev_external.csv ({len(df_dev_official)} rows) to {official_dev_out}")

        # (4) Internal splits (FILTERED)
        print("\nCombining official train and dev sets for internal restructuring...")
        df_combined = pd.concat([df_train_official, df_dev_official], ignore_index=True)

        # Filter < 5 words sentences purely for internal splits
        initial_combined_len = len(df_combined)
        df_combined = df_combined[df_combined['word_count'] >= 5].copy()
        print(f"Dropped {initial_combined_len - len(df_combined)} noisy rows for internal splitting.")
        print(f"Total combined records for internal splits: {len(df_combined)}")

        # 80% train, 20% pool
        train_df, pool_df = train_test_split(
            df_combined,
            test_size=0.2,
            random_state=seed,
            stratify=df_combined["label"]
        )

        # Cut pool in half -> 10% internal dev, 10% internal test
        internal_dev_df, internal_test_df = train_test_split(
            pool_df,
            test_size=0.5,
            random_state=seed,
            stratify=pool_df["label"]
        )

        # Print statistics
        print("\nNew Split Distributions:")
        for name, df in [
            ("Train", train_df), ("Internal Dev", internal_dev_df), ("Internal Test", internal_test_df)
        ]:
            counts = df["label"].value_counts().to_dict()
            print(f"{name:>15}: {len(df)} rows | Class 0: {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")

        # Save files
        train_df[["par_id", "text", "label"]].to_csv(os.path.join(output_dir, "train.csv"), index=False)
        internal_dev_df[["par_id", "text", "label"]].to_csv(os.path.join(output_dir, "dev_internal.csv"), index=False)
        internal_test_df[["par_id", "text", "label"]].to_csv(os.path.join(output_dir, "test_internal.csv"), index=False)
        print("\nSaved train.csv, dev_internal.csv, and test_internal.csv")

        # (5) Process official test set (UNFILTERED)
        if os.path.exists(self.test_path):
            print(f"\nProcessing official test set from {self.test_path}...")
            df_test = pd.read_csv(
                self.test_path,
                sep='\t',
                header=None,
                names=["par_id", "art_id", "keyword", "country", "text"],
                quoting=3
            )

            df_test["text"] = df_test["text"].fillna('')
            df_test["clean_text"] = df_test["text"].apply(self.clean_text)
            df_test["keyword"] = df_test["keyword"].fillna("unknown")
            df_test["country"] = df_test["country"].fillna("unknown")

            df_test["text"] = "[KEYWORD] " + df_test["keyword"] + " [COUNTRY] " + df_test["country"] + " [SEP] " + df_test["clean_text"]

            out_file = os.path.join(output_dir, "test_external.csv")
            df_test[["par_id", 'text']].to_csv(out_file, index=False)
            print(f"  -> Saved test_external.csv ({len(df_test)} rows) to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="PCL Data Preprocessing Pipeline")
    parser.add_argument("--raw_file", type=str, default="data/dontpatronizeme_pcl.tsv")
    parser.add_argument("--train_ids", type=str, default="data/train_semeval_parids-labels.csv")
    parser.add_argument("--dev_ids", type=str, default="data/dev_semeval_parids-labels.csv")
    parser.add_argument("--test_file", type=str, default="data/task4_test.tsv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cleaner = PCLCleaner(
        args.raw_file,
        args.train_ids,
        args.dev_ids,
        args.test_file
    )
    cleaner.process_and_save(args.output_dir, args.seed)

if __name__ == "__main__":
    main()
