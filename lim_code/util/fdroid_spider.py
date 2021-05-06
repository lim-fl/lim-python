import scrapy
import sys
import time
import os
from scrapy.http import Request

class FdroidSpider(scrapy.Spider):
    name = 'f-droid-spider'
    start_urls = ['https://f-droid.org/en/packages/index.html']

    if (not os.path.exists('apks')):
        os.makedirs('apks')

    custom_settings = {
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 0.25,        
    }

    def parse(self, response):
        base_url = 'https://f-droid.org/en/packages'

        for href in response.css('a.package-header ::attr("href")').extract():
            yield Request(
                url=response.urljoin(href),
                callback=self.parse_package
            )

        for next_page in response.css('.nav.next > a'):
            yield response.follow(next_page, self.parse)

    def parse_package(self, response):
        apk_url = response.css('p.package-version-download > a ::attr("href")').extract_first();

        yield Request(
            url=apk_url,
            callback=self.save_apk
        )

    def save_apk(self, response):
        path = 'apks/' + response.url.split('/')[-1]

        if (os.path.exists(path)):
            return

        self.logger.info('Saving APK %s as %s', response.url, path)

        with open(path, 'wb') as f:
            f.write(response.body)