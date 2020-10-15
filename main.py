import argparse
from common import config
import logging

import news_page_objects as news

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _news_scraper(news_site_uid):
    host = config()['news_sites'][news_site_uid]['url']
    logger.info('Beginning scrapper for {}'.format(host))
    homepage = news.HomePage(news_site_uid, host)

    for link in homepage.article_links:
        print(link)

if __name__ == "__main__":
    
    news_sites = list(config()['news_sites'].keys())

    parser = argparse.ArgumentParser()
    parser.add_argument('news_site',
                        help='The sites that you want to scrape',
                        type=str,
                        choices=news_sites
    )
    args = parser.parse_args()
    _news_scraper(args.news_site)

