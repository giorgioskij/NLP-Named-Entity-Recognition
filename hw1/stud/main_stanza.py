import stanza

stanza.download('en')
nlp = stanza.Pipeline('en')
doc = nlp('Barack Obama was born in Hawaii.')
