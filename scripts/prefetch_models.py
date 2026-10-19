"""Download the tools-env models ahead of time (Whisper large-v3, ECAPA, vocal separator, de-reverb / de-noise)."""
from audio_separator.separator import Separator

from clean_refs import STEPS
from prep_voice import SEP_MODEL, ecapa
from vl_common import CACHE, whisper_model

whisper_model()
print("whisper large-v3 ready")
ecapa()
print("ecapa ready")
for model in [SEP_MODEL] + [m for m, _ in STEPS]:
    Separator(model_file_dir=str(CACHE / "separator"), output_dir=str(CACHE / "tmp")).load_model(model_filename=model)
    print(f"separator {model} ready")
