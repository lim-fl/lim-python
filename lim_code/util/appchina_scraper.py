#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Get the category of an Appchina app.
"""

import requests
from scrapy.selector import Selector
from mtranslate import translate

def details(package_name):
    url = 'http://www.appchina.com/app/' + package_name
    try:
        response = requests.get(url)
        return parse_content(response)
    except Exception as e:
        raise ValueError('Could not scrape Appchina: {}'.format(e))


def parse_content(response):
    hxs = Selector(response)
    try:
        item = {}
        # https://github.com/wanganhong/Spider/blob/77fd4510149a783d066fe0a133bbe082897a395b/ApkSpider/ApkSpider/spiders/appchina.py
        c = hxs.xpath('//div[@class="breadcrumb centre-content"]/a[3]/text()').extract()[0]
        item['category'] = translate(c)
        return item
    except Exception as e :
        raise ValueError('Could not scrape Appchina: {}'.format(e))