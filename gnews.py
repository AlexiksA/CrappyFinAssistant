"""
You'll have to decode the RSS URLs by yourself...
I'm not sure how okay it would be to commit a file using
undocumented google endpoints :?

Google News RSS items don't link to the article. They link to an opaque
https://news.google.com/rss/articles/<token> URL, and turning that into the
publisher's real URL is left as an exercise. The rest of the project only
needs one function from here: decode_google_news_url(url) -> str.

How it should be done, roughly:

1. The last path segment is a URL-safe base64 token. Pad it, decode it, and
   strip the protobuf-looking framing (a short fixed prefix and suffix).
   Older tokens hold the real URL in plain sight after a length byte, and
   you're done.
2. Newer tokens (they decode to something starting with "AU_yqL") hide the
   URL. Fetch the article page on news.google.com, and read the signature
   and timestamp attributes it embeds (data-n-a-sg, data-n-a-ts).
3. POST those, plus the token, to the DotsSplashUi batchexecute endpoint
   using the "Fbv4je" RPC, and dig the URL out of the "garturlres" entry in
   the response.
4. Expect the EU consent wall (a SOCS cookie gets you past it), a browser
   User-Agent, rate limits, and breakage whenever Google changes the
   payload. Fall back to following redirects if the tokens are missing.

Until you write that, this returns the input URL unchanged, so downstream
stages will scrape the Google interstitial instead of the article.
"""


def decode_google_news_url(source_url: str) -> str:
    # TODO: implement, see the module docstring.
    return source_url
