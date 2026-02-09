import os, sys
print('PYTHON:', sys.executable)
print('OPENAI_API_KEY set:', bool(os.getenv('OPENAI_API_KEY')))
try:
    import requests
except Exception as e:
    print('requests import failed:', e)
    sys.exit(0)
key=os.getenv('OPENAI_API_KEY')
if not key:
    print('NO_KEY')
    sys.exit(0)
import traceback
url='https://api.openai.com/v1/models'
print('Probing', url)
try:
    r=requests.get(url, headers={'Authorization':f'Bearer {key}'}, timeout=10)
    print('status', r.status_code)
    txt=r.text
    print('text_start:', txt[:500])
except Exception as e:
    print('probe exception:', e)
    traceback.print_exc()
