import random

import feedparser

ny = feedparser.parse('https://newyork.craigslist.org/')
print(ny['entries'])