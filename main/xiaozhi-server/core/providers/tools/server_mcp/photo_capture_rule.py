from __future__ import annotations

from plugins_func.register import Action, ActionResponse


class ServerPhotoCaptureRule:
    TAKE_PHOTO_TOOL = "xiaozhi_take_photo"
    DENY_MESSAGE = "请先问用户可以拍照吗，得到肯定答复后再拍。"

    def __init__(self, conn) -> None:
        self.conn = conn

    def before_execute(self, actual_tool_name: str) -> tuple[ActionResponse | None, bool]:
        if actual_tool_name != self.TAKE_PHOTO_TOOL:
            return None, False

        if not bool(getattr(self.conn, "_server_photo_capture_granted", False)):
            return (
                ActionResponse(
                    action=Action.RESPONSE,
                    response=self.DENY_MESSAGE,
                ),
                False,
            )

        self.conn._server_photo_capture_granted = False
        return None, True

    def restore_after_failure(self, actual_tool_name: str, restore_grant: bool) -> None:
        if actual_tool_name == self.TAKE_PHOTO_TOOL and restore_grant:
            self.conn._server_photo_capture_granted = True
