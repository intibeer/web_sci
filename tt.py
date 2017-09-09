
ep_token = [b'a', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as', b'as']


#ep_token = [str(token) for token in ep_token]
#ep_token = " ".join(ep_token)
#ep_token = ep_token.decode('utf-8')
#ep_token = nlp(ep_token)
#ep_tokens_vec = ep_token.vector
#ep_token = bytearray(ep_token).decode('ascii')
ep_token = b' '.join(ep_token).decode("ascii")

print(ep_token)