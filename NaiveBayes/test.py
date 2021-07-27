import random

import feedparser

ny = feedparser.parse('http://feed.read.org.cn/')
print(ny['entries'])