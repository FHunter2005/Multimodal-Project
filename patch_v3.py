"""
Applies the pretrained-model patch to eye_follower_v3.py in-place.
Run once: python patch_v3.py
"""
from pathlib import Path

src = Path('eye_follower_v3.py').read_text()

OLD = '''                    def _train():
                        global training_in_progress
                        fit_calibration(calib_features, calib_targets, calib_weights)
                        stretcher.fit(calib_features, calib_targets)
                        kalman.reset()
                        heatmap.reset()
                        _calib_event.set()
                        training_in_progress = False
                        print("Calibration complete!  Press 'd' for drift fix.")'''

NEW = '''                    def _train():
                        global training_in_progress, model_x, model_y
                        import pickle
                        from pathlib import Path as _P
                        pkl = _P('pretrained_model.pkl')
                        if pkl.exists():
                            print("[Pretrain] Loading pretrained model …")
                            with open(pkl, 'rb') as _f:
                                _d = pickle.load(_f)
                            model_x = _d['model_x']
                            model_y = _d['model_y']
                            import pretrain_and_finetune as _ptf
                            _ptf.model_x = model_x
                            _ptf.model_y = model_y
                            _ptf.finetune(calib_features, calib_targets,
                                          calib_weights, n_extra_estimators=60)
                            model_x = _ptf.model_x
                            model_y = _ptf.model_y
                        else:
                            print("[Pretrain] pretrained_model.pkl not found "
                                  "— training from scratch.")
                            fit_calibration(calib_features, calib_targets,
                                            calib_weights)
                        stretcher.fit(calib_features, calib_targets)
                        kalman.reset()
                        heatmap.reset()
                        _calib_event.set()
                        training_in_progress = False
                        print("Calibration complete!  Press 'd' for drift fix.")'''

if OLD not in src:
    print("Pattern not found — already patched or file differs. No changes made.")
else:
    Path('eye_follower_v3.py').write_text(src.replace(OLD, NEW, 1))
    print("Patched eye_follower_v3.py successfully.")
