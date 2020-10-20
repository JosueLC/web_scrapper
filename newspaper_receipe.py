import argparse
import logging
from urllib.parse import urlparse
import pandas as pd
import hashlib
import re
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('spanish'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_data(filename):
    logger.info('Reading file {}'.format(filename))
    return pd.read_csv(filename)

def _extract_newspaper_uid(filename):
    logger.info('Extracting newspaper UID')
    newspaper_uid = filename.split('_')[0]
    logger.info('Newspaper UID detected: {}'.format(newspaper_uid))
    return newspaper_uid

def _add_newspaper_uid_column(df, newspaper_uid):
    logger.info('Filling newspaper UID column with {}'.format(newspaper_uid))
    df['newspaper_uid'] = newspaper_uid
    return df

def _extract_host(df):
    logger.info('Extracting host from urls')
    df['host'] = df['url'].apply(lambda url: urlparse(url).netloc)
    return df

def _fill_missing_titles(df):
    logger.info('Filling missing titles')
    missing_titles_mask = df['title'].isna()
    missing_titles = (df[missing_titles_mask]['url']
                        .str.extract(r'(?P<missing_titles>[^/]+)$')
                        .applymap(lambda title: title.split('-'))
                        .applymap(lambda title_word_list: ' '.join(title_word_list))
                     )
    df.loc[missing_titles_mask, 'title'] = missing_titles.loc[:, 'missing_titles']
    return df

def _generate_uids_for_row(df):
    logger.info('Generating uids for each row')
    uids = (df
            .apply(lambda row: hashlib.md5(bytes(row['url'].encode())), axis=1)
            .apply(lambda hashobject: hashobject.hexdigest())
    )
    df['uid'] = uids
    return df.set_index('uid')

def _strip_body(df):
    logger.info('Clean invisibles characters from body')
    stripped_body = (df
                        .apply(lambda row: row['body'], axis=1)
                        .apply(lambda body: re.sub(r'(\n|\r)+',r'',body))
                    )
    df['body'] = stripped_body
    return df

def _tokenize_column(df, column_name):
    logger.info('Tokenizing from column {}'.format(column_name))
    df['n_tokens_'+column_name] = (df
                                    .dropna()
                                    .apply(lambda row: nltk.word_tokenize(row[column_name]), axis=1)
                                    .apply(lambda tokens: list(filter(lambda token: token.isalpha(), tokens)))
                                    .apply(lambda tokens: list(map(lambda token: token.lower(), tokens)))
                                    .apply(lambda word_list: list(filter(lambda word: word not in stop_words, word_list)))
                                    .apply(lambda valid_word_list: len(valid_word_list))
                                  )
    return df

def _remove_duplicates_entries(df,column_name):
    logger.info('Removing duplicates values in {}'.format(column_name))
    return df.drop_duplicates(subset=[column_name], keep='first')

def _drop_rows_with_missing_data(df):
    logger.info('Dropping rows with missing values')
    return df.dropna()

def _save_data(df, filename):
    clean_filename = 'clean_{}'.format(filename)
    logger.info('Saving data at location: {}'.format(clean_filename))
    df.to_csv(clean_filename)

def main(filename):
    logger.info('Starting cleaning process')

    df = _read_data(filename)
    newspaper_uid = _extract_newspaper_uid(filename)
    df = _add_newspaper_uid_column(df, newspaper_uid)
    df = _extract_host(df)
    df = _fill_missing_titles(df)
    df = _generate_uids_for_row(df)
    df = _strip_body(df)
    df = _tokenize_column(df, 'title')
    df = _tokenize_column(df, 'body')
    df = _remove_duplicates_entries(df,'title')
    df = _drop_rows_with_missing_data(df)
    _save_data(df, filename)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'filename',
        help='The path to the dirty data',
        type=str
    )
    args = parser.parse_args()
    df = main(args.filename)
    print(df)