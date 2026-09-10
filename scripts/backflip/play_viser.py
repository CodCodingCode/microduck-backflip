"""mjlab play with the viser viewer, patched so zero-range command sliders don't assert."""
import sys
import viser
_orig = viser.GuiApi.add_slider
def _patched(self, label, min, max, step, initial_value, *a, **k):
    lo, hi = min, max
    initial_value = lo if initial_value < lo else (hi if initial_value > hi else initial_value)
    return _orig(self, label, lo, hi, step, initial_value, *a, **k)
viser.GuiApi.add_slider = _patched
import mjlab_microduck.tasks  # register microduck tasks
from mjlab.scripts.play import main
sys.exit(main())
