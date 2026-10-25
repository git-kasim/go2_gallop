"""Isaac Sim 6.1: 未保存のシーンの確認を出さずに終了する。

実行: isaacsim_send.py --ff --file quit_isaacsim.py
"""
import carb.settings
import omni.kit.app

settings = carb.settings.get_settings()
settings.set("/app/file/ignoreUnsavedOnExit", True)
settings.set("/app/file/ignoreUnsavedStage", True)
print("quitting")
omni.kit.app.get_app().post_quit(0)
