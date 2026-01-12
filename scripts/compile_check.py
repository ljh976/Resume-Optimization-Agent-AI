import py_compile, glob

files = glob.glob('core/*.py') + ['app.py']
ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print('OK', f)
    except Exception as e:
        ok = False
        print('ERROR', f, e)
if ok:
    print('All files compiled successfully')
else:
    print('Some files failed to compile')
