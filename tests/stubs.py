"""Recording stand-ins for streamlit and plotly, used ONLY by tests because those packages cannot be installed in
the build sandbox. They let the dashboard code run so that logic/NameError/data-connection bugs are caught.
They do NOT validate real rendering or the real Streamlit/Plotly APIs."""
import sys
import types
from unittest.mock import MagicMock

LOG = []        # every string passed to st.markdown / st.info / st.error / st.caption ... (for assertions)


def install():
    st = MagicMock(name="streamlit")

    def rec(*a, **k):
        LOG.extend(str(x) for x in a if isinstance(x, str))
    for n in ("markdown", "info", "error", "warning", "success", "caption", "write", "code", "download_button",
              "progress", "exception"):
        getattr(st, n).side_effect = rec
    n_of = lambda spec: spec if isinstance(spec, int) else len(spec)
    def col():
        c = MagicMock()
        c.text_input.return_value = ""
        c.multiselect.side_effect = lambda label, options, default=None, **k: list(default if default is not None else options)
        c.selectbox.side_effect = lambda label, options, **k: list(options)[0]
        return c
    st.columns.side_effect = lambda spec, **k: [col() for _ in range(n_of(spec))]
    st.tabs.side_effect = lambda labels, **k: [MagicMock() for _ in labels]
    st.radio.side_effect = lambda label, options, **k: list(options)[0]
    st.selectbox.side_effect = lambda label, options, **k: list(options)[0]
    st.multiselect.side_effect = lambda label, options, default=None, **k: list(default if default is not None else options)
    st.text_input.return_value = ""
    st.button.return_value = False
    st.checkbox.return_value = False
    st.cache_data.side_effect = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    st.plotly_chart.side_effect = lambda *a, **k: None
    st.dataframe.side_effect = lambda *a, **k: None
    sys.modules["streamlit"] = st
    for name in ("plotly", "plotly.express", "plotly.graph_objects"):
        sys.modules[name] = MagicMock(name=name)
    sys.modules["plotly"].express = sys.modules["plotly.express"]
    sys.modules["plotly"].graph_objects = sys.modules["plotly.graph_objects"]
    return st
