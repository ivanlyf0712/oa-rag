#!/usr/bin/env python3
import importlib.util, os, py_compile, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP = os.path.join(ROOT, 'apps', 'oa', 'app.py')
SEARCH = os.path.join(ROOT, 'apps', 'oa', 'search.py')
TICKET4 = os.path.join(ROOT, '.scratch', 'oa-rag-refinement', 'issues', '04-contract-streamlit-ui.md')


def check(cond, label):
    print(('PASS' if cond else 'FAIL') + ' - ' + label)
    if not cond:
        raise SystemExit(1)


def main():
    py_compile.compile(SEARCH, doraise=True)
    py_compile.compile(APP, doraise=True)
    try:
        import pandas  # noqa: F401
        import streamlit  # noqa: F401
    except Exception as exc:
        print(f'FAIL - missing runtime dependency in venv: {exc}')
        raise SystemExit(1)
    spec = importlib.util.spec_from_file_location('oa_app', APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    check(hasattr(mod, 'main'), 'OA app exposes main()')
    check(hasattr(mod, '_render_search'), 'OA app exposes search view')
    check(hasattr(mod, '_render_browser'), 'OA app exposes browse view')
    check(hasattr(mod, '_render_dashboard'), 'OA app exposes dashboard view')
    src = open(APP, encoding='utf-8').read()
    check('Onyx Chat' not in src, 'OA app contains no Onyx Chat surface')
    check(os.path.exists(TICKET4), 'Ticket 4 issue file exists')
    txt = open(TICKET4, encoding='utf-8').read()
    check('[x] The app loads into contract search, browser, and dashboard views only.' in txt, 'Ticket 4 app-view criterion marked done')
    check('[x] No Onyx Chat, iframe chat embed, or chat-specific navigation remains in the UI.' in txt, 'Ticket 4 no-chat criterion marked done')
    print('OA requirements check completed successfully.')


if __name__ == '__main__':
    main()
