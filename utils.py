from six import unichr
from html.entities import name2codepoint as n2cp
import re
import multiprocessing
import itertools
import numpy as np

RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)

ARTICLE_MIN_WORDS = 50
"""Ignore shorter articles (after full preprocessing)."""

# default thresholds for lengths of individual tokens
TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15

RE_P0 = re.compile(r'<!--.*?-->', re.DOTALL | re.UNICODE)
"""Comments."""
RE_P1 = re.compile(r'<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
"""Footnotes."""
RE_P2 = re.compile(r'(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$', re.UNICODE)
"""Links to languages."""
RE_P3 = re.compile(r'{{([^}{]*)}}', re.DOTALL | re.UNICODE)
"""Template."""
RE_P4 = re.compile(r'{{([^}]*)}}', re.DOTALL | re.UNICODE)
"""Template."""
RE_P5 = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)
"""Remove URL, keep description."""
RE_P6 = re.compile(r'\[([^][]*)\|([^][]*)\]', re.DOTALL | re.UNICODE)
"""Simplify links, keep description."""
RE_P7 = re.compile(r'\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
"""Keep description of images."""
RE_P8 = re.compile(r'\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
"""Keep description of files."""
RE_P9 = re.compile(r'<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)
"""External links."""
RE_P10 = re.compile(r'<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)
"""Math content."""
RE_P11 = re.compile(r'<(.*?)>', re.DOTALL | re.UNICODE)
"""All other tags."""
RE_P12 = re.compile(r'(({\|)|(\|-(?!\d))|(\|}))(.*?)(?=\n)', re.UNICODE)
"""Table formatting."""
RE_P13 = re.compile(r'(?<=(\n[ ])|(\n\n)|([ ]{2})|(.\n)|(.\t))(\||\!)([^[\]\n]*?\|)*', re.UNICODE)
"""Table cell formatting."""
RE_P14 = re.compile(r'\[\[Category:[^][]*\]\]', re.UNICODE)
"""Categories."""
RE_P15 = re.compile(r'\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)
"""Remove File and Image templates."""
RE_P16 = re.compile(r'\[{2}(.*?)\]{2}', re.UNICODE)
"""Capture interlinks text and article linked"""
RE_P17 = re.compile(
    r'(\n.{0,4}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=)|(scope=))(.*))|'
    r'(^.{0,2}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=))(.*))',
    re.UNICODE
)
"""Table markup"""
IGNORED_NAMESPACES = [
    'Wikipedia', 'Category', 'File', 'Portal', 'Template',
    'MediaWiki', 'User', 'Help', 'Book', 'Draft', 'WikiProject',
    'Special', 'Talk'
]


def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.

    Returns
    -------
    str
        Unicode version of `text`.

    """
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)


def safe_unichr(intval):
    """Create a unicode character from its integer value. In case `unichr` fails, render the character
    as an escaped `\\U<8-byte hex value of intval>` string.

    Parameters
    ----------
    intval : int
        Integer code of character

    Returns
    -------
    string
        Unicode string of character

    """
    try:
        return unichr(intval)
    except ValueError:
        # ValueError: unichr() arg not in range(0x10000) (narrow Python build)
        s = "\\U%08x" % intval
        # return UTF16 surrogate pair
        return s.decode('unicode-escape')

def decode_htmlentities(text):
    def substitute_entity(match):
        try:
            ent = match.group(3)
            if match.group(1) == "#":
                # decoding by number
                if match.group(2) == '':
                    # number is in decimal
                    return safe_unichr(int(ent))
                elif match.group(2) in ['x', 'X']:
                    # number is in hex
                    return safe_unichr(int(ent, 16))
            else:
                # they were using a name
                cp = n2cp.get(ent)
                if cp:
                    return safe_unichr(cp)
                else:
                    return match.group()
        except Exception:
            # in case of errors, return original input
            return match.group()

    return RE_HTML_ENTITY.sub(substitute_entity, text)

def filter_wiki(raw, promote_remaining=True, simplify_links=True):
    """Filter out wiki markup from `raw`, leaving only text.

    Parameters
    ----------
    raw : str
        Unicode or utf-8 encoded string.
    promote_remaining : bool
        Whether uncaught markup should be promoted to plain text.
    simplify_links : bool
        Whether links should be simplified keeping only their description text.

    Returns
    -------
    str
        `raw` without markup.

    """
    # parsing of the wiki markup is not perfect, but sufficient for our purposes
    # contributions to improving this code are welcome :)
    text = to_unicode(raw, 'utf8', errors='ignore')
    text =decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    return remove_markup(text, promote_remaining, simplify_links)


def remove_template(s):
    """Remove template wikimedia markup.

    Parameters
    ----------
    s : str
        String containing markup template.

    Returns
    -------
    str
        Сopy of `s` with all the `wikimedia markup template <http://meta.wikimedia.org/wiki/Help:Template>`_ removed.

    Notes
    -----
    Since template can be nested, it is difficult remove them using regular expressions.

    """
    # Find the start and end position of each template by finding the opening
    # '{{' and closing '}}'
    n_open, n_close = 0, 0
    starts, ends = [], [-1]
    in_template = False
    prev_c = None
    for i, c in enumerate(s):
        if not in_template:
            if c == '{' and c == prev_c:
                starts.append(i - 1)
                in_template = True
                n_open = 1
        if in_template:
            if c == '{':
                n_open += 1
            elif c == '}':
                n_close += 1
            if n_open == n_close:
                ends.append(i)
                in_template = False
                n_open, n_close = 0, 0
        prev_c = c

    # Remove all the templates
    starts.append(None)
    return ''.join(s[end + 1:start] for end, start in zip(ends, starts))


def remove_file(s):
    """Remove the 'File:' and 'Image:' markup, keeping the file caption.

    Parameters
    ----------
    s : str
        String containing 'File:' and 'Image:' markup.

    Returns
    -------
    str
        Сopy of `s` with all the 'File:' and 'Image:' markup replaced by their `corresponding captions
        <http://www.mediawiki.org/wiki/Help:Images>`_.

    """
    # The regex RE_P15 match a File: or Image: markup
    for match in re.finditer(RE_P15, s):
        m = match.group(0)
        caption = m[:-2].split('|')[-1]
        s = s.replace(m, caption, 1)
    return s

def remove_markup(text, promote_remaining=True, simplify_links=True):
    """Filter out wiki markup from `text`, leaving only text.

    Parameters
    ----------
    text : str
        String containing markup.
    promote_remaining : bool
        Whether uncaught markup should be promoted to plain text.
    simplify_links : bool
        Whether links should be simplified keeping only their description text.

    Returns
    -------
    str
        `text` without markup.

    """
    text = re.sub(RE_P2, '', text)  # remove the last list (=languages)
    # the wiki markup is recursive (markup inside markup etc)
    # instead of writing a recursive grammar, here we deal with that by removing
    # markup in a loop, starting with inner-most expressions and working outwards,
    # for as long as something changes.
    text = remove_template(text)
    text = remove_file(text)
    iters = 0
    while True:
        old, iters = text, iters + 1
        text = re.sub(RE_P0, '', text)  # remove comments
        text = re.sub(RE_P1, '', text)  # remove footnotes
        text = re.sub(RE_P9, '', text)  # remove outside links
        text = re.sub(RE_P10, '', text)  # remove math content
        text = re.sub(RE_P11, '', text)  # remove all remaining tags
        text = re.sub(RE_P14, '', text)  # remove categories
        text = re.sub(RE_P5, '\\3', text)  # remove urls, keep description

        if simplify_links:
            text = re.sub(RE_P6, '\\2', text)  # simplify links, keep description only
        # remove table markup
        text = text.replace("!!", "\n|")  # each table head cell on a separate line
        text = text.replace("|-||", "\n|")  # for cases where a cell is filled with '-'
        text = re.sub(RE_P12, '\n', text)  # remove formatting lines
        text = text.replace('|||', '|\n|')  # each table cell on a separate line(where |{{a|b}}||cell-content)
        text = text.replace('||', '\n|')  # each table cell on a separate line
        text = re.sub(RE_P13, '\n', text)  # leave only cell content
        text = re.sub(RE_P17, '\n', text)  # remove formatting lines

        # remove empty mark-up
        text = text.replace('[]', '')
        # stop if nothing changed between two iterations or after a fixed number of iterations
        if old == text or iters > 2:
            break

    if promote_remaining:
        text = text.replace('[', '').replace(']', '')  # promote all remaining markup to plain text

    return text


class InputQueue(multiprocessing.Process):
    """Populate a queue of input chunks from a streamed corpus.

    Useful for reading and chunking corpora in the background, in a separate process,
    so that workers that use the queue are not starved for input chunks.

    """
    def __init__(self, q, corpus, chunksize, maxsize, as_numpy):
        """
        Parameters
        ----------
        q : multiprocessing.Queue
            Enqueue chunks into this queue.
        corpus : iterable of iterable of (int, numeric)
            Corpus to read and split into "chunksize"-ed groups
        chunksize : int
            Split `corpus` into chunks of this size.
        as_numpy : bool, optional
            Enqueue chunks as `numpy.ndarray` instead of lists.

        """
        super(InputQueue, self).__init__()
        self.q = q
        self.maxsize = maxsize
        self.corpus = corpus
        self.chunksize = chunksize
        self.as_numpy = as_numpy

    def run(self):
        it = iter(self.corpus)
        while True:
            chunk = itertools.islice(it, self.chunksize)
            if self.as_numpy:
                # HACK XXX convert documents to numpy arrays, to save memory.
                # This also gives a scipy warning at runtime:
                # "UserWarning: indices array has non-integer dtype (float64)"
                wrapped_chunk = [[np.asarray(doc) for doc in chunk]]
            else:
                wrapped_chunk = [list(chunk)]

            if not wrapped_chunk[0]:
                self.q.put(None, block=True)
                break

            try:
                qsize = self.q.qsize()
            except NotImplementedError:
                qsize = '?'
            self.q.put(wrapped_chunk.pop(), block=True)


def chunkize_serial(iterable, chunksize, as_numpy=False, dtype=np.float32):
    """Yield elements from `iterable` in "chunksize"-ed groups.

    The last returned element may be smaller if the length of collection is not divisible by `chunksize`.

    Parameters
    ----------
    iterable : iterable of object
        An iterable.
    chunksize : int
        Split iterable into chunks of this size.
    as_numpy : bool, optional
        Yield chunks as `np.ndarray` instead of lists.

    Yields
    ------
    list OR np.ndarray
        "chunksize"-ed chunks of elements from `iterable`.

    Examples
    --------
    .. sourcecode:: pycon

        >>> print(list(grouper(range(10), 3)))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """
    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[np.array(doc, dtype=dtype) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        # memory opt: wrap the chunk and then pop(), to avoid leaving behind a dangling reference
        yield wrapped_chunk.pop()


def chunkize(corpus, chunksize, maxsize=0, as_numpy=False):
    """Split `corpus` into fixed-sized chunks, using :func:`~gensim.utils.chunkize_serial`.

    Parameters
    ----------
    corpus : iterable of object
        An iterable.
    chunksize : int
        Split `corpus` into chunks of this size.
    maxsize : int, optional
        If > 0, prepare chunks in a background process, filling a chunk queue of size at most `maxsize`.
    as_numpy : bool, optional
        Yield chunks as `np.ndarray` instead of lists?

    Yields
    ------
    list OR np.ndarray
        "chunksize"-ed chunks of elements from `corpus`.

    Notes
    -----
    Each chunk is of length `chunksize`, except the last one which may be smaller.
    A once-only input stream (`corpus` from a generator) is ok, chunking is done efficiently via itertools.

    If `maxsize > 0`, don't wait idly in between successive chunk `yields`, but rather keep filling a short queue
    (of size at most `maxsize`) with forthcoming chunks in advance. This is realized by starting a separate process,
    and is meant to reduce I/O delays, which can be significant when `corpus` comes from a slow medium
    like HDD, database or network.

    If `maxsize == 0`, don't fool around with parallelism and simply yield the chunksize
    via :func:`~gensim.utils.chunkize_serial` (no I/O optimizations).

    Yields
    ------
    list of object OR np.ndarray
        Groups based on `iterable`

    """
    assert chunksize > 0

    if maxsize > 0:
        q = multiprocessing.Queue(maxsize=maxsize)
        worker = InputQueue(q, corpus, chunksize, maxsize=maxsize, as_numpy=as_numpy)
        worker.daemon = True
        worker.start()
        while True:
            chunk = [q.get(block=True)]
            if chunk[0] is None:
                break
            yield chunk.pop()
    else:
        for chunk in chunkize_serial(corpus, chunksize, as_numpy=as_numpy):
            yield chunk