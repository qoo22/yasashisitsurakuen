import sys, subprocess, os
from gradio_client import Client

name, prompt, seed, length = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tok = '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'
c = Client('sanchit-gandhi/musicgen-streaming', token=tok, verbose=False, download_files=False)
res = c.predict(text_prompt=prompt, audio_length_in_s=length, play_steps_in_s=2, seed=seed,
                api_name='/generate_audio')
url = res['url']
out = 'music/%s.mp3' % name
r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', url,
                    '-ac', '1', '-ar', '32000', '-b:a', '96k', out])
if r.returncode == 0 and os.path.exists(out):
    print('OK %s -> %d bytes' % (out, os.path.getsize(out)))
else:
    print('FAIL', name)
