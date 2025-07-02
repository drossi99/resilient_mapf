import sys
import customtkinter as ctk
from src.view.state import AppState
from src.view.layout_builder import build_layout
from src.view.handlers import configure_handlers

TITLE = "MAPF Solver"
WINDOW_SIZE = "1700x1200"

class MainGUIController:
    def __init__(self):
        self.console_redirector = None

        try:
            self.root = ctk.CTk()
            self.state = AppState(self.root)
            self.root.title(TITLE)
            self.root.geometry(WINDOW_SIZE)

            self.console_textbox = build_layout(self.root, self.state)
            self._redirect_stdout()

            configure_handlers(self.root, self.state, self.console_textbox)
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        except Exception as e:
            print(f"Error during controller init: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _redirect_stdout(self):
        try:
            if self.console_textbox is None:
                return

            from src.view.console_redirector import ThreadSafeConsoleRedirector

            self.console_redirector = ThreadSafeConsoleRedirector(
                self.console_textbox,
                self.root
            )

            # Only redirect if textbox is ready
            if self.console_textbox.winfo_exists():
                sys.stdout = self.console_redirector
                sys.stderr = self.console_redirector
            else:
                print("Console_textbox not ready, postponing redirect")
                # Schedule redirect for later
                self.root.after(100, self._delayed_redirect)

        except Exception as e:
            print(f"Error setting up stdout redirect: {e}")
            # Continue without redirect rather than crash

    def _delayed_redirect(self):
        try:
            if self.console_textbox and self.console_textbox.winfo_exists():
                sys.stdout = self.console_redirector
                sys.stderr = self.console_redirector
        except Exception as e:
            print(f"Delayed redirect failed: {e}")

    def on_closing(self):
        try:
            if self.console_redirector:
                self.console_redirector.restore()

            self.root.destroy()

        except Exception as e:
            print(f"Error during closing: {e}")
            # Force close
            try:
                self.root.quit()
            except:
                pass

    def run(self):
        try:
            print("Starting main loop...")
            self.root.mainloop()
        except Exception as e:
            print(f">Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.console_redirector:
                self.console_redirector.restore()