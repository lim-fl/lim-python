#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Get the category of an Anzhi app.
"""

from bs4 import BeautifulSoup
import requests

# https://github.com/ArionHill/pcapcatch/blob/2e7e071786a219a192fef1a3aa06aae69f275287/PcapCatch/anZhiLink.py
def details(package_name):
    url = 'http://www.anzhi.com/pkg/' + package_name
    try:
        html = requests.get(url).text
    except Exception as e:
        raise ValueError('Could not scrape Anzhi: {}'.format(e))

    soup = BeautifulSoup(html, 'html.parser')
    info = {}
    divs = soup.select('div')

    for i in divs:
        try:
            if i['class'][0] == 'detail_line':
                spans = i.select('span')
                for j in spans:
                    if j['class'][0] == 'app_detail_version':
                        info['version'] = j.string
                        name = i.select('h3')
                        for j in name:
                            info['name'] = j.string
            if i['class'][0] == 'app_detail_infor':
                infor = i.select('p')
                for j in infor:
                    info['intro'] += str(j).replace('"', r'\"')
                info['intro'] = str(j).replace('"', r'\"')
                # exit(0)
        except KeyError:
            pass

    uls = soup.select('ul')
    for i in uls:
        try:
            if i['id'] == 'detail_line_ul':
                lis = i.select('li')
                for j in lis:
                    if '分类' in j.string:
                        info['category'] = j.string
                    elif '下载' in j.string:
                        info['downloads'] = j.string
                    elif '时间' in j.string:
                        info['updatetime'] = j.string
                    elif '大小' in j.string:
                        info['size'] = j.string
                    elif '系统' in j.string:
                        info['system'] = j.string
                    elif '资费' in j.string:
                        info['price'] = j.string
                    elif '作者' in j.string:
                        info['composer'] = j.string
                    elif '软件语言' in j.string:
                        info['language'] = j.string
        except KeyError as e:
            pass
        except Exception as e:
            raise ValueError('Could not scrape Anzhi: {}'.format(e))

    return info
