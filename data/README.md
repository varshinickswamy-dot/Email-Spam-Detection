# Dataset setup

Raw corpora and processed email splits are intentionally excluded from Git:
the email messages can contain personal information, and generated CSV splits
contain their text. The trained model and aggregate evaluation reports are
included, so running the web app does not require downloading either corpus.

## Rebuild the email model

Download these Apache SpamAssassin Public Corpus archives from
<https://spamassassin.apache.org/old/publiccorpus/> and place them in
`data/raw/spamassassin/`:

- `20030228_easy_ham.tar.bz2`
- `20030228_easy_ham_2.tar.bz2`
- `20030228_hard_ham.tar.bz2`
- `20030228_spam.tar.bz2`
- `20030228_spam_2.tar.bz2`

Then run `scripts/inspect_email_corpus.py` followed by
`scripts/train_email_model.py`. The generated normalized corpus and train,
validation, and test CSV files remain local and are ignored by Git.

## Rebuild the legacy SMS model

Download the UCI SMS Spam Collection from
<https://archive.ics.uci.edu/dataset/228/sms+spam+collection> and place its
`SMSSpamCollection` text file at `data/raw/SMSSpamCollection`. Then run
`scripts/train_model.py`. This SMS corpus is a separate task and is never used
by the email scanner.

Check each dataset's source and terms before redistributing its contents.
