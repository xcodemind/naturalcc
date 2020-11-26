import os
from multiprocessing import Pool, cpu_count

try:
    from dataset.csn_feng import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR,
        RETRIEVAL_RAW_DATA_DIR, RETRIEVAL_SPLITTER,
        LOGGER,
    )
except ImportError:
    from .. import (
        LANGUAGES, MODES,
        RAW_DATA_DIR, LIBS_DIR, FLATTEN_DIR,
        RETRIEVAL_RAW_DATA_DIR, RETRIEVAL_SPLITTER,
        LOGGER,
    )

if __name__ == '__main__':
    lang = 'ruby'
    filename = os.path.join(RETRIEVAL_RAW_DATA_DIR['train'], lang, 'train.txt')

    lines = []
    with open(filename, 'r', encoding='utf8') as reader:
        for line in reader:
            line = line.strip().split(RETRIEVAL_SPLITTER)
            """
            [
                '1', 
                'https://github.com/epuber-io/epuber/blob/4d736deb3f18c034fc93fcb95cfb9bf03b63c252/lib/epuber/server/handlers.rb#L50-L55', 
                'Epuber.Server.handle_bade', 
                '@param [ String ] file_path path to bade file to render', 
                "def handle_bade ( file_path ) [ 200 , self . class . render_bade ( file_path ) ] rescue => e env [ 'sinatra.error' ] = e ShowExceptions . new ( self ) . call ( env ) end"
            ]
            """
            if len(line) != 5:
                continue
            lines.append(line)
